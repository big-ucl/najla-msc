from enum import Enum
from pathlib import Path
from typing import Self

import numpy as np
import polars as pl
from pyproj import Transformer


def convert_excel_to_parquet(data_path: Path, *files: list[Path]) -> list[Path]:
    new_files = []

    for file in files:
        df = pl.read_excel(data_path / file)

        new_file = file.with_suffix(".parquet")
        df.write_parquet(data_path / new_file)
        new_files.append(new_file)

    return new_files


def read_from_parquet(path: Path, schema: dict = None) -> pl.DataFrame:
    schema = {} if schema is None else schema

    df = pl.read_parquet(path)
    return pl.DataFrame(df, schema_overrides=schema)


def bng_to_lat_long(
    df: pl.DataFrame, eastings_col: str, northings_col: str
) -> tuple[np.ndarray, np.ndarray]:
    BNG_EPSG_CODE = 27700
    LAT_LONG_EPSG_CODE = 4326
    transformer = Transformer.from_crs(BNG_EPSG_CODE, LAT_LONG_EPSG_CODE)

    return transformer.transform(df[eastings_col], df[northings_col])


class _PolarsEnum(Enum):
    @classmethod
    def names(cls):
        return [member.value for member in cls]

    @classmethod
    def polars_enum(cls):
        return pl.Enum(cls)


class Purpose(_PolarsEnum):
    MISSING = "Missing"
    NOT_ASKED = "Not asked"
    HOME = "Home"
    WORK = "Work - Usual workplace"
    WORK_DELIVERY = "Work - Delivery/loading"
    WORK_OTHER = "Work - Other"
    ENTERTAINMENT = "Entertainment/recreation"
    SHOPPING_FOOD = "Shopping - Food"
    PERSONAL_BUSINESS = "Personal business / use services"
    EDUCATION = "Education (as a pupil)"
    HOTEL = "Hotel / holiday home"
    ESCORT_WORK = "Drop off/pick up someone to/from work"
    ESCORT_SCHOOL = "Drop off/pick up someone to/from school"
    ESCORT_HEALTH = "Drop off/pick up someone to/from health visit"
    WORSHIP = "Worship or religious observance"
    OTHER = "Other"
    HEALTH = "Health or medical visit"
    ESCORT_OTHER = "Drop off/pick up someone to/from other place"
    SPORT = "Participate in Sport"
    LEISURE = "Leisure trip - enjoyment"
    SOCIAL_VISIT = "Visit friends/relatives at home"
    SOCIAL_OTHER = "Other Social"
    SHOPPING_OTHER = "Shopping - Other"


class LandUse(_PolarsEnum):
    MISSING = "Missing"
    NOT_ASKED = "Not asked"
    RESIDENTIAL = "Residential"
    OFFICE = "Office"
    FACTORY = "Factory/warehouse"
    SCHOOL = "School/College"
    SHOPS = "Shops"
    PUBLIC_BUILDING = "Public Buildings"
    OPEN_SPACE = "Open space"
    MYSTERY = "MYSTERY LAND USE"
    WORSHIP = "Place of worship"
    OTHER = "Other"
    HOSPITAL = "Hospital"
    GP = "GP/Dentist/Other health service"


HH_PERSON_SCHEMA = pl.Schema(
    {
        "hh_id": pl.String,
        "person_id": pl.String,
        "year": pl.Int64,
        "loc_work_locid": pl.String,
        "loc_work_lat": pl.Float64,
        "loc_work_lon": pl.Float64,
        "loc_home_locid": pl.String,
        "loc_home_lat": pl.Float64,
        "loc_home_lon": pl.Float64,
    }
)

TRIP_SCHEMA = pl.Schema(
    {
        "hh_id": pl.String,
        "person_id": pl.String,
        "trip_id": pl.String,
        "trip_number": pl.Int64,
        "year": pl.Int64,
        "mode": pl.Int64,
        "duration": pl.Int64,
        "distance": pl.Float64,
        "purpose": Purpose.polars_enum(),
        "purpose_dest": Purpose.polars_enum(),
        "land_use": LandUse.polars_enum(),
        "start_time": pl.Int64,
        "end_time": pl.Int64,
        "loc_origin_locid": pl.String,
        "loc_origin_lat": pl.Float64,
        "loc_origin_lon": pl.Float64,
        "loc_dest_locid": pl.String,
        "loc_destination_lat": pl.Float64,
        "loc_destination_lon": pl.Float64,
    }
)


def check_schema(df: pl.DataFrame, schema: pl.Schema) -> pl.DataFrame:
    df_items = set(df.schema.items())
    schema_items = set(schema.items())

    difference = df_items ^ schema_items

    if difference:
        raise ValueError(
            f"Schemas do not match:\nExpected: {schema}\nGot:      {df.schema}\nDifferent elements: {difference}"
        )

    return df


class ActivityDataset:
    name: str
    hh_person_df: pl.DataFrame
    trip_df: pl.DataFrame

    _HHP_FILENAME = "hh_person_df.parquet"
    _TRIP_FILENAME = "trip_df.parquet"

    def __init__(self, name: str, hh_person_df: pl.DataFrame, trip_df: pl.DataFrame):
        self.name = name
        self.hh_person_df = check_schema(hh_person_df, HH_PERSON_SCHEMA)
        self.trip_df = check_schema(trip_df, TRIP_SCHEMA)

    def save(self, path: Path, dir_name: str = None) -> Path:
        dir_name = dir_name if dir_name is not None else self.name

        dataset_dir = path / dir_name
        dataset_dir.mkdir(exist_ok=True)

        hh_path = dataset_dir / self._add_file_prefix(self.name, self._HHP_FILENAME)
        trip_path = dataset_dir / self._add_file_prefix(self.name, self._TRIP_FILENAME)

        self.hh_person_df.write_parquet(hh_path)
        self.trip_df.write_parquet(trip_path)

        return dataset_dir

    @classmethod
    def load(cls, path: Path, dir_name: str, name: str = None) -> Self:
        name = dir_name if name is None else dir_name
        dataset_dir = path / dir_name

        hh_path = dataset_dir / cls._add_file_prefix(name, cls._HHP_FILENAME)
        trip_path = dataset_dir / cls._add_file_prefix(name, cls._TRIP_FILENAME)

        hh_person_df = check_schema(pl.read_parquet(hh_path), HH_PERSON_SCHEMA)
        trip_df = check_schema(pl.read_parquet(trip_path), TRIP_SCHEMA)

        return ActivityDataset(name, hh_person_df, trip_df)

    @classmethod
    def exists_on_disk(cls, path: Path, dir_name: str, name: str = None) -> bool:
        name = dir_name if name is None else dir_name
        dataset_dir = path / dir_name

        hh_path = dataset_dir / cls._add_file_prefix(name, cls._HHP_FILENAME)
        trip_path = dataset_dir / cls._add_file_prefix(name, cls._TRIP_FILENAME)

        return dataset_dir.exists() and hh_path.exists() and trip_path.exists()

    @classmethod
    def _add_file_prefix(cls, name: str, filename: str):
        name = name if not name else name + "_"
        return name + filename
