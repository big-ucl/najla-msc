from enum import IntFlag, auto
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


class _PolarsEnum(IntFlag):
    @classmethod
    def names(cls):
        return [cls.description(member) for member in cls]

    @classmethod
    def polars_enum(cls):
        return pl.UInt32

    @classmethod
    def description(cls, member: Self):
        return cls(member).name


class Purpose(_PolarsEnum):
    MISSING = auto()
    NOT_ASKED = auto()
    HOME = auto()
    WORK = auto()
    WORK_DELIVERY = auto()
    WORK_OTHER = auto()
    ENTERTAINMENT = auto()
    SHOPPING_FOOD = auto()
    PERSONAL_BUSINESS = auto()
    EDUCATION = auto()
    HOTEL = auto()
    ESCORT_WORK = auto()
    ESCORT_SCHOOL = auto()
    ESCORT_HEALTH = auto()
    WORSHIP = auto()
    OTHER = auto()
    HEALTH = auto()
    ESCORT_OTHER = auto()
    SPORT = auto()
    LEISURE = auto()
    SOCIAL_VISIT = auto()
    SOCIAL_OTHER = auto()
    SHOPPING_OTHER = auto()

    @classmethod
    def description(cls, member: Self):
        descriptions = {
            cls.MISSING: "Missing",
            cls.NOT_ASKED: "Not asked",
            cls.HOME: "Home",
            cls.WORK: "Work - Usual workplace",
            cls.WORK_DELIVERY: "Work - Delivery/loading",
            cls.WORK_OTHER: "Work - Other",
            cls.ENTERTAINMENT: "Entertainment/recreation",
            cls.SHOPPING_FOOD: "Shopping - Food",
            cls.PERSONAL_BUSINESS: "Personal business / use services",
            cls.EDUCATION: "Education (as a pupil)",
            cls.HOTEL: "Hotel / holiday home",
            cls.ESCORT_WORK: "Drop off/pick up someone to/from work",
            cls.ESCORT_SCHOOL: "Drop off/pick up someone to/from school",
            cls.ESCORT_HEALTH: "Drop off/pick up someone to/from health visit",
            cls.WORSHIP: "Worship or religious observance",
            cls.OTHER: "Other",
            cls.HEALTH: "Health or medical visit",
            cls.ESCORT_OTHER: "Drop off/pick up someone to/from other place",
            cls.SPORT: "Participate in Sport",
            cls.LEISURE: "Leisure trip - enjoyment",
            cls.SOCIAL_VISIT: "Visit friends/relatives at home",
            cls.SOCIAL_OTHER: "Other Social",
            cls.SHOPPING_OTHER: "Shopping - Other",
        }

        return descriptions[member]


class LandUse(_PolarsEnum):
    MISSING = auto()
    NOT_ASKED = auto()
    RESIDENTIAL = auto()
    OFFICE = auto()
    FACTORY = auto()
    SCHOOL = auto()
    SHOPS = auto()
    PUBLIC_BUILDING = auto()
    OPEN_SPACE = auto()
    MYSTERY = auto()
    WORSHIP = auto()
    OTHER = auto()
    HOSPITAL = auto()
    GP = auto()

    @classmethod
    def description(cls, member: Self):
        descriptions = {
            cls.MISSING: "Missing",
            cls.NOT_ASKED: "Not asked",
            cls.RESIDENTIAL: "Residential",
            cls.OFFICE: "Office",
            cls.FACTORY: "Factory/warehouse",
            cls.SCHOOL: "School/College",
            cls.SHOPS: "Shops",
            cls.PUBLIC_BUILDING: "Public Buildings",
            cls.OPEN_SPACE: "Open space",
            cls.MYSTERY: "MYSTERY LAND USE",
            cls.WORSHIP: "Place of worship",
            cls.OTHER: "Other",
            cls.HOSPITAL: "Hospital",
            cls.GP: "GP/Dentist/Other health service",
        }

        return descriptions[member]


HH_PERSON_SCHEMA = pl.Schema({
    "hh_id": pl.String,
    "person_id": pl.String,
    "year": pl.Int64,
    "loc_work_loc_id": pl.String,
    "loc_work_lat": pl.Float64,
    "loc_work_lon": pl.Float64,
    "loc_home_loc_id": pl.String,
    "loc_home_lat": pl.Float64,
    "loc_home_lon": pl.Float64,
})

TRIP_SCHEMA = pl.Schema({
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
    "loc_origin_loc_id": pl.String,
    "loc_origin_lat": pl.Float64,
    "loc_origin_lon": pl.Float64,
    "loc_dest_loc_id": pl.String,
    "loc_destination_lat": pl.Float64,
    "loc_destination_lon": pl.Float64,
})


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
