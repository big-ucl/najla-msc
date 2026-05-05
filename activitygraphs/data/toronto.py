from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import geopandas as gpd
import polars as pl

from activitygraphs.base import CRS, LOCATIONS_COLUMNS, LOCATIONS_SCHEMA, USER_JOURNEY_SCHEMA
from activitygraphs.config import DataConfig, TorontoDataConfig
from activitygraphs.network import NetworkData
from activitygraphs.utils import DataFrameStore, add_lon_lat_from_centroid, check_schema, get_project_root

TORONTO_CMA = 535


class ZoneType(StrEnum):
    CT = "CT"
    DA = "DA"


@dataclass(frozen=True)
class TorontoInputs:
    raw_journeys_df: pl.DataFrame

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

    boundaries_dir = project_root / cfg.inputs.boundaries.directory
    cma = gpd.read_file(boundaries_dir / cfg.inputs.boundaries.metropolitan_areas)
    x = boundaries_dir / cfg.inputs.boundaries.metropolitan_areas

    ct = gpd.read_file(boundaries_dir / cfg.inputs.boundaries.census_tracts)
    da = gpd.read_file(boundaries_dir / cfg.inputs.boundaries.dissemination_areas)

    return TorontoInputs(raw_journeys_df, cma, ct, da)


def build_toronto_data(inputs: TorontoInputs) -> TorontoData:
    locations_gdf = build_toronto_locations(inputs)
    user_journeys_df = build_toronto_journeys(inputs, locations_gdf)

    return TorontoData(inputs, locations_gdf, inputs.raw_journeys_df)


def build_toronto_locations(inputs: TorontoInputs) -> gpd.GeoDataFrame:
    subsector_locations = build_subsector_locations(inputs)
    return subsector_locations


def build_toronto_journeys(inputs: TorontoInputs, locations_gdf: gpd.GeoDataFrame) -> pl.DataFrame:
    return inputs.raw_journeys_df  # TODO Pre-process raw_journeys


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
