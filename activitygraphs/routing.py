import abc
from collections.abc import Iterable

import geopandas as gpd
import numpy as np
import polars as pl
import requests
from pypolyline.cutil import encode_coordinates

from activitygraphs.network import Mode
from activitygraphs.utils import check_schema

MAX_COORDS = 10_000
PRECISION = 5

MODE_TO_OSRM_PROFILE_MAP = {
    Mode.CAR: "car",
    Mode.MOTORCYCLE: "car",
    Mode.TAXI: "car",
    Mode.VEH_PASS: "car",
    Mode.WALK: "walk",
    Mode.CYCLE: "cycle",
}


class Router(abc.ABC):
    def travel_times(self, edge_df: pl.DataFrame, locations_gdf: gpd.GeoDataFrame) -> pl.DataFrame:
        check_schema(edge_df, pl.Schema({"orig_loc_id": pl.String, "dest_loc_id": pl.String}), ignore_extra_cols=True)
        check_schema(locations_gdf, {"loc_id": "object", "lon": "float64", "lat": "float64"}, ignore_extra_cols=True)

        locations_df = pl.DataFrame(locations_gdf[["loc_id", "lon", "lat"]])
        origins = locations_df.select(pl.all().name.prefix("orig_"))
        destinations = locations_df.select(pl.all().name.prefix("dest_"))

        edge_coords_df = (
            edge_df.select("orig_loc_id", "dest_loc_id")
            .join(origins, on="orig_loc_id")
            .join(destinations, on="dest_loc_id")
        )

        result = self._travel_times(edge_coords_df)
        check_schema(
            result, pl.Schema({"orig_loc_id": pl.String, "dest_loc_id": pl.String, "travel_time_min": pl.Float64})
        )

        return edge_df.join(result, on=["orig_loc_id", "dest_loc_id"], how="left")

    @abc.abstractmethod
    def _travel_times(self, edge_coords_df: pl.DataFrame) -> pl.DataFrame:
        pass


class OSRMRouter(Router):
    def __init__(self, url: str, mode: Mode, max_coordinates: int = MAX_COORDS, precision: int = PRECISION):
        if mode not in MODE_TO_OSRM_PROFILE_MAP.keys():
            raise ValueError(f"Invalid mode: {mode}, accepted: {MODE_TO_OSRM_PROFILE_MAP.keys()}")

        self.url = url
        self.mode = mode
        self.profile = MODE_TO_OSRM_PROFILE_MAP[mode]
        self.max_coordinates = max_coordinates
        self.precision = precision

    def _travel_times(self, edge_df: pl.DataFrame):
        locations = (
            pl.concat([
                edge_df.select(loc_id="orig_loc_id", lon="orig_lon", lat="orig_lat"),
                edge_df.select(loc_id="dest_loc_id", lon="dest_lon", lat="dest_lat"),
            ])
            .unique("loc_id")
            .sort("loc_id")
            .with_row_index()
        )

        coordinates = locations.select("lon", "lat").iter_rows()
        durations = self.table(coordinates)

        edge_indices = edge_df.join(locations.select(orig_index="index", orig_loc_id="loc_id"), on="orig_loc_id").join(
            locations.select(dest_index="index", dest_loc_id="loc_id"), on="dest_loc_id"
        )
        edge_indices_np = edge_indices.select("orig_index", "dest_index").to_numpy()
        travel_times = durations[edge_indices_np[:, 0], edge_indices_np[:, 1]]

        return edge_df.select("orig_loc_id", "dest_loc_id").with_columns(travel_time_min=travel_times)

    def table(self, coordinates: Iterable[tuple[float, float]]) -> np.ndarray:
        coordinates = list(coordinates)

        if len(coordinates) > self.max_coordinates:
            raise ValueError(f"Too many coordinates: {len(coordinates)}, maximum: {self.max_coordinates}")

        polyline = encode_coordinates(coordinates, self.precision)
        full_url = f"{self.url}/table/v1/{self.profile}/polyline({polyline})"
        response = requests.get(full_url)

        if not response.ok:
            raise ConnectionError(f"OSRM routing service error: status code={response.status_code}")

        json = response.json()
        if json["code"] != "Ok":
            raise ConnectionError(f"OSRM routing error:\nstatus code={json['code']}\nmessage={json['message']}")

        travel_times_min = np.array(json["durations"]) / 60
        return travel_times_min


class CachedRouter(Router):
    def __init__(self, router: Router):
        self.router = router
        self.cache: dict[tuple[str, str], float] = {}

    def _travel_times(self, edge_df) -> pl.DataFrame:
        # TODO

        return self.router._travel_times(edge_df)


def compute_osrm_travel_times(edge_df: pl.DataFrame, locations_df: gpd.GeoDataFrame | pl.DataFrame):
    pass
