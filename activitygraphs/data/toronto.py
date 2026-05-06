from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd
import polars as pl

from activitygraphs.base import CRS, LOCATIONS_COLUMNS, LOCATIONS_SCHEMA, USER_JOURNEY_SCHEMA, Mode, Purpose
from activitygraphs.config import TorontoDataConfig
from activitygraphs.network import NA, NetworkData, build_special_locations
from activitygraphs.utils import (
    DataFrameStore,
    add_lon_lat_from_centroid,
    check_schema,
    gdf_to_polars,
    get_project_root,
)

TORONTO_CMA = 535

MODE_MAP = {
    "AIR_OR_HSR": Mode.OTHER,
    "BICYCLING": Mode.CYCLE,
    "BUS": Mode.BUS,
    "CAR": Mode.CAR,
    "FERRY": Mode.BOAT,
    "LIGHT_RAIL": Mode.TRAIN,
    "SUBWAY": Mode.TRAIN,
    "TRAIN": Mode.TRAIN,
    "TRAM": Mode.TRAMWAY,
    "TROLLEYBUS": Mode.BUS,
    "UNKNOWN": Mode.UNKNOWN,
    "WALKING": Mode.WALK,
}

MANUAL_MODE_MAP = {
    "walk": Mode.WALK,
    "car-park-off-street-res": Mode.CAR,
    "car-park-off-street-non-res": Mode.CAR,
    "transit": Mode.BUS,
    "bike": Mode.CYCLE,
    "car-park-on-street-res": Mode.CAR,
    "car-park-on-street-non-res": Mode.CAR,
    "train": Mode.TRAIN,
    "taxi": Mode.VEH_PASS,
    "train-transit": Mode.TRAIN,
    "golf_cart": Mode.OTHER,
}

PURPOSE_MAP = {
    str(Purpose.UNKNOWN): Purpose.UNKNOWN,
    "home": Purpose.HOME,
    "entertainment-visit": Purpose.ENTERTAINMENT,
    "excercice-recreation": Purpose.LEISURE_OTHER,
    "work": Purpose.WORK_MAIN,
    "shopping-grocerie": Purpose.SHOP,
    "pick-drop": Purpose.ESCORT,
    "appointment-visit": Purpose.PERSONAL,
    "personal-business": Purpose.PERSONAL,
    "school": Purpose.STUDY,
    "dog_walk": Purpose.LEISURE_OTHER,
    "church": Purpose.LEISURE_OTHER,
    "shopping": Purpose.SHOP,
    "golf": Purpose.LEISURE_OTHER,
    "walk_dog": Purpose.LEISURE_OTHER,
    "shunting": Purpose.ESCORT,
    "return_home": Purpose.HOME,
    "volunteering": Purpose.LEISURE_OTHER,
    "vacation": Purpose.LONG_DISTANCE_TRIP,
    "volunteer": Purpose.LEISURE_OTHER,
    "visit_family": Purpose.VISIT,
    "lunch": Purpose.ENTERTAINMENT,
    "travel": Purpose.LONG_DISTANCE_TRIP,
    "religious": Purpose.LEISURE_OTHER,
    "going_home": Purpose.HOME,
    "parcel_deliveries in brampton": Purpose.WORK_OTHER,
    "library": Purpose.LEISURE_OTHER,
    "cat_sitting for a friend": Purpose.VISIT,
    "coffee": Purpose.ENTERTAINMENT,
    "family_visit": Purpose.VISIT,
    "walk": Purpose.LEISURE_OTHER,
    "traveling": Purpose.LONG_DISTANCE_TRIP,
    # More can be added
}


@dataclass(frozen=True)
class TorontoInputs:
    raw_journeys_df: pl.DataFrame
    raw_person_df: pl.DataFrame

    metropolitan_areas_gdf: gpd.GeoDataFrame
    census_tracts_gdf: gpd.GeoDataFrame
    dissemination_areas_gdf: gpd.GeoDataFrame


class TorontoData(NetworkData, DataFrameStore):
    def __init__(
        self,
        inputs: TorontoInputs,
        locations_gdf: gpd.GeoDataFrame,
        user_journeys_df: pl.DataFrame,
        filters: list[str] | None = None,
    ):
        super().__init__(user_journeys_df, locations_gdf, filters)
        self.inputs = inputs

    def _copy(self, filters: list[str] | None = None):
        return TorontoData(self.inputs, self._locations_gdf, self._user_journeys_df, filters)

    @classmethod
    def load(cls, cfg: TorontoDataConfig, project_root: Path | None = None, name: str | None = None) -> "TorontoData":
        project_root, data_dir = cls._dirs(cfg, project_root, name)
        toronto_inputs = load_files(cfg, project_root)

        if data_dir.exists():
            locations_gdf = gpd.read_parquet(data_dir / "locations_gdf.parquet")
            user_journeys_df = pl.read_parquet(data_dir / "user_journeys_df.parquet", schema=USER_JOURNEY_SCHEMA)

            return cls(toronto_inputs, locations_gdf, user_journeys_df)
        else:
            data = build_toronto_data(toronto_inputs)
            data.save(cfg, project_root, name)

            return data

    def save(self, cfg: TorontoDataConfig, project_root: Path | None = None, name: str | None = None):
        project_root, data_dir = self._dirs(cfg, project_root, name)

        data_dir.mkdir(parents=True, exist_ok=True)
        self.locations_gdf.to_parquet(data_dir / "locations_gdf.parquet")
        self.user_journeys_df.write_parquet(data_dir / "user_journeys_df.parquet")


def load_files(cfg: TorontoDataConfig, project_root: Path | None = None) -> TorontoInputs:
    project_root = get_project_root(project_root)

    raw_data_dir = project_root / cfg.paths.raw
    raw_journeys_df = pl.read_parquet(raw_data_dir / "trips.parquet")
    raw_persons_df = pl.read_parquet(raw_data_dir / "indivs.parquet")

    boundaries_dir = project_root / cfg.inputs.boundaries.directory
    boundary_cma = gpd.read_file(boundaries_dir / cfg.inputs.boundaries.metropolitan_areas)
    boundary_ct = gpd.read_file(boundaries_dir / cfg.inputs.boundaries.census_tracts)
    boundary_da = gpd.read_file(boundaries_dir / cfg.inputs.boundaries.dissemination_areas)

    return TorontoInputs(raw_journeys_df, raw_persons_df, boundary_cma, boundary_ct, boundary_da)


def build_toronto_data(inputs: TorontoInputs) -> TorontoData:
    locations_gdf = build_toronto_locations(inputs)
    user_journeys_df = build_toronto_journeys(inputs, locations_gdf)

    return TorontoData(inputs, locations_gdf, user_journeys_df)


def build_toronto_locations(inputs: TorontoInputs) -> gpd.GeoDataFrame:
    special_locations = build_special_locations()
    subsector_locations = build_subsector_locations(inputs)
    return gpd.GeoDataFrame(
        pd.concat(
            [
                special_locations,
                subsector_locations,
            ],
            ignore_index=True,
        ),
        crs=CRS,
    )


def build_toronto_journeys(inputs: TorontoInputs, locations_gdf: gpd.GeoDataFrame) -> pl.DataFrame:
    trips = inputs.raw_journeys_df
    persons = inputs.raw_person_df

    # Replace app_id with user_id
    app_to_user_id_map = persons.select("app_id", "person_id").unique()
    user_journeys_df = trips.join(app_to_user_id_map, on="app_id")

    # Replace transport modes
    user_journeys_df = user_journeys_df.with_columns(
        pl.col("predicted_modes").replace_strict(MODE_MAP, default=Mode.UNKNOWN),
        pl.col("manual_mode").replace_strict(MANUAL_MODE_MAP, default=None),
    ).with_columns(
        pl
        .when(pl.col("manual_mode").is_not_null())
        .then(pl.col("manual_mode"))
        .otherwise("predicted_modes")
        .alias("leg_mode")
    )

    # Rank trip legs by start time
    leg_rank_expr = pl.col("start_fmt_time_section").rank("dense").over("app_id", "trip_id") - 1

    # Replace trip purposes
    purpose_expr = (
        pl.col("manual_purpose").fill_null(Purpose.UNKNOWN).replace_strict(PURPOSE_MAP, default=Purpose.OTHER)
    )

    user_journeys_df = user_journeys_df.select(
        user_id="person_id",
        journey_id="trip_id",
        leg_id=leg_rank_expr.cast(pl.Int8),
        leg_mode=pl.col("leg_mode").cast(pl.Categorical),
        leg_line=pl.lit(None, dtype=pl.String),
        dep_day=pl.col("start_fmt_time_section").str.to_date(format="%+"),
        dep_time=pl.col("start_fmt_time_section").str.to_time(format="%+"),
        dep_loc_id=pl.col("start_loc_CT_section"),
        arr_loc_id=pl.col("end_loc_CT_section"),
        arr_purpose=purpose_expr,
    )

    # Add departure purposes (purpose of previous trip)
    unique_trips = (
        user_journeys_df
        .sort("user_id", "dep_day", "dep_time")
        .unique(["user_id", "journey_id"], keep="first")
        .select(
            "journey_id",
            "arr_purpose",
            previous_journey_id=pl.col("journey_id").shift(1).over(["user_id", "dep_day"], order_by="dep_time"),
        )
    )

    dep_purposes = unique_trips.join(
        unique_trips.select("journey_id", dep_purpose="arr_purpose"),
        left_on="previous_journey_id",
        right_on="journey_id",
    ).drop("arr_purpose", "previous_journey_id")

    user_journeys_df = user_journeys_df.join(dep_purposes, on="journey_id", how="left").with_columns(
        pl.col("arr_purpose").cast(pl.Categorical),
        pl.col("dep_purpose").fill_null(Purpose.UNKNOWN).cast(pl.Categorical),
    )

    # Replace null loc_ids with NA
    user_journeys_df = user_journeys_df.with_columns(
        pl.col("dep_loc_id").fill_null(NA), pl.col("arr_loc_id").fill_null(NA)
    )

    return user_journeys_df.select(USER_JOURNEY_SCHEMA.keys()).sort(
        "user_id",
        "dep_day",
        "dep_time",
    )


def build_subsector_locations(inputs: TorontoInputs) -> gpd.GeoDataFrame:
    toronto_cma = inputs.metropolitan_areas_gdf.query(f"CMAUID == '{TORONTO_CMA}'")
    utm_crs = toronto_cma.estimate_utm_crs()

    boundaries_gdf = inputs.census_tracts_gdf
    boundaries_gdf = boundaries_gdf.to_crs(utm_crs).sjoin(toronto_cma.to_crs(utm_crs))
    boundaries_gdf = boundaries_gdf.drop(columns=["index_right"]).reset_index(drop=True).to_crs(CRS)

    boundaries_gdf = boundaries_gdf.rename(columns={"CTUID": "loc_id", "CTNAME": "loc_name"})
    boundaries_gdf["type"] = "subsector"
    boundaries_gdf = add_lon_lat_from_centroid(boundaries_gdf, index_col="loc_id")

    # noinspection PyTypeChecker
    gdf: gpd.GeoDataFrame = boundaries_gdf[LOCATIONS_COLUMNS].copy()

    return check_schema(gdf, LOCATIONS_SCHEMA)
