from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import cached_property
from typing import Self, TypeVar

import geopandas as gpd
import polars as pl

from activitygraphs import utils
from activitygraphs.base import (
    CRS,
    LOCATIONS_SCHEMA,
    USER_JOURNEY_SCHEMA,
    PTNodeType,
)
from activitygraphs.routing import TravelTimeCalculator
from activitygraphs.utils import check_schema

NA_LON, NA_LAT = 6.1709475192397605, 46.24348817355701
NA, NA_SOURCE, NA_SINK = "NA", "NA_SOURCE", "NA_SINK"

type PTLayerBuilder = Callable[[pl.DataFrame, PTNodeType], tuple[pl.DataFrame, pl.DataFrame]]
type TravelTimeFactory = float | pl.Expr | pl.DataFrame | TravelTimeCalculator | Callable[[str, str], float]
TDataFrame = TypeVar("TDataFrame", pl.DataFrame, gpd.GeoDataFrame)


class NetworkData(ABC):
    _user_journeys_df: pl.DataFrame
    _locations_gdf: gpd.GeoDataFrame

    def __init__(
        self, user_journeys_df: pl.DataFrame, locations_gdf: gpd.GeoDataFrame, filters: list[str] | None = None
    ):
        self._user_journeys_df = check_schema(user_journeys_df, USER_JOURNEY_SCHEMA)
        self._locations_gdf = check_schema(locations_gdf, LOCATIONS_SCHEMA)

        self._type_filters = None if filters is None else filters.copy()

    @cached_property
    def user_journeys_df(self) -> pl.DataFrame:
        dep_filtered = self._filter_loc_types(self._user_journeys_df, loc_id_col="dep_loc_id")
        return self._filter_loc_types(dep_filtered, loc_id_col="arr_loc_id")

    @cached_property
    def locations_gdf(self) -> gpd.GeoDataFrame:
        return self._filter_loc_types(self._locations_gdf)

    @cached_property
    def locations_df(self) -> pl.DataFrame:
        return utils.gdf_to_polars(self.locations_gdf)

    @cached_property
    def aggregated_journeys(self) -> pl.DataFrame:
        group_keys = ["user_id", "journey_id"]
        grouped_journeys = self._user_journeys_df.group_by(group_keys, maintain_order=True)

        modes = grouped_journeys.agg(modes="leg_mode")
        trips = grouped_journeys.agg(pl.all().gather([0, -1])).select(
            "user_id",
            "journey_id",
            (pl.col("leg_id").list.last() + 1).alias("num_legs"),
            pl.col("dep_day").list.first(),
            pl.col("dep_time").list.first(),
            pl.col("dep_purpose").list.first(),
            pl.col("dep_loc_id").list.first(),
            pl.col("arr_loc_id").list.last(),
            pl.col("arr_purpose").list.last(),
        )

        trips_with_modes = trips.join(modes, on=group_keys)

        filtered_trips = self._filter_loc_types(trips_with_modes, loc_id_col="dep_loc_id")
        filtered_trips = self._filter_loc_types(filtered_trips, loc_id_col="arr_loc_id")

        return filtered_trips

    @cached_property
    def location_visits(self) -> pl.DataFrame:
        departures = self.aggregated_journeys.select("user_id", purpose="dep_purpose", loc_id="dep_loc_id")
        arrivals = self.aggregated_journeys.select("user_id", purpose="arr_purpose", loc_id="arr_loc_id")
        all_visits = pl.concat([departures, arrivals])

        return all_visits.group_by("user_id", "purpose", "loc_id").agg(num_visits=pl.len()).sort("user_id")

    @cached_property
    def location_visits_by_purpose(self) -> pl.DataFrame:
        return self.location_visits.group_by("user_id", "purpose").agg(pl.col("loc_id").unique()).sort("user_id")

    @cached_property
    def home_locations(self) -> pl.DataFrame:
        return self.location_visits_by_purpose.filter(purpose="od_lieu_domicile").with_columns(
            pl.col("loc_id").list.first()
        )

    @cached_property
    def work_locations(self) -> pl.DataFrame:
        return self.location_visits_by_purpose.filter(purpose="od_lieu_travail").with_columns(
            pl.col("loc_id").list.len()
        )

    @cached_property
    def edu_locations(self) -> pl.DataFrame:
        return self.location_visits_by_purpose.filter(purpose="od_lieu_etude").with_columns(pl.col("loc_id").list.len())

    @cached_property
    def user_ids(self) -> pl.Series:
        return self.location_visits["user_id"].unique().sort()

    def with_filter(self, loc_types: str | list[str]) -> Self:
        filters = [loc_types] if isinstance(loc_types, str) else loc_types
        return self._copy(filters)

    def _filter_loc_types(self, df: TDataFrame, loc_id_col: str = "loc_id") -> TDataFrame:
        if self._type_filters is None:
            return df

        locations_df = utils.gdf_to_polars(self._locations_gdf)

        valid_loc_ids = locations_df.filter(pl.col("type").is_in(self._type_filters)).select(
            pl.col("loc_id").alias(loc_id_col)
        )

        if isinstance(df, pl.DataFrame):
            return df.join(valid_loc_ids, on=loc_id_col)

        return df.merge(valid_loc_ids.to_pandas(), on=loc_id_col)

    @abstractmethod
    def _copy(self, filters: list[str] | None = None): ...


def build_special_locations() -> gpd.GeoDataFrame:
    na_location = {
        "loc_id": "NA",
        "loc_name": "NA",
        "type": "na",
        "lon": NA_LON,
        "lat": NA_LAT,
    }

    return gpd.GeoDataFrame(
        [na_location],
        geometry=gpd.points_from_xy([NA_LON], [NA_LAT]),
        crs=CRS,
    )
