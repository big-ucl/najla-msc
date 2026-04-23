from collections.abc import Mapping
from pathlib import Path
from typing import TypeVar

import geopandas as gpd
import pandas as pd
import polars as pl
import torch

from activitygraphs.base import CRS, EDGE_LIST_SCHEMA, LOCATIONS_SCHEMA

PandasSchema = Mapping[str, str]
TDataFrame = TypeVar("TDataFrame", gpd.GeoDataFrame, pl.DataFrame)
TSchema = TypeVar("TSchema", pl.Schema, PandasSchema)


def get_project_root(project_root: Path | None = None) -> Path:
    project_root: Path = project_root if project_root is not None else Path(".")
    return project_root


def check_schema(df: TDataFrame, schema: TSchema, ignore_extra_cols: bool = False) -> TDataFrame:
    """Checks that a DataFrame matches provided Schema.

    Args:
        df (pl.DataFrame | gpd.GeoDataFrame): the dataframe to be checked
        schema (pl.Schema | dict): the schema to check against
        ignore_extra_cols (bool): ignore extra columns not in the schema, defaults to False

    Raises:
        ValueError: if the DataFrame does not match the schema

    Returns:
        pl.DataFrame: the valid DataFrame
    """

    if isinstance(df, pl.DataFrame) and isinstance(schema, pl.Schema):
        return check_polars_schema(df, schema, ignore_extra_cols=ignore_extra_cols)

    if (isinstance(df, gpd.GeoDataFrame) or isinstance(df, pd.DataFrame)) and isinstance(schema, dict):
        return check_geopandas_schema(df, schema, ignore_extra_cols=ignore_extra_cols)

    raise ValueError(f"Invalid types: Got {type(df)=}, {type(schema)=}")


def check_polars_schema(df: pl.DataFrame, schema: pl.Schema, ignore_extra_cols: bool) -> pl.DataFrame:
    df_items = set(df.schema.items())
    schema_items = set(schema.items())

    missing = schema_items - df_items
    extra = df_items - schema_items

    if missing or (extra and not ignore_extra_cols):
        raise ValueError(
            f"Schemas do not match:\nMissing elements: {missing}\n"
            + (f"Extra elements: {extra}" if not ignore_extra_cols else "")
        )

    return df


def check_geopandas_schema(gdf: gpd.GeoDataFrame, schema: PandasSchema, ignore_extra_cols: bool) -> gpd.GeoDataFrame:
    def dtypes_match(dtype: str, expected: str) -> bool:
        return (dtype == "str" and expected == "object") or dtype == expected

    if not ignore_extra_cols and set(schema.keys()) != set(gdf.columns):
        raise ValueError(
            f"Columns do not match schema. \nGot: {sorted(gdf.columns)}\nExpected: {sorted(schema.keys())}"
        )

    if ignore_extra_cols and not set(schema.keys()).issubset(set(gdf.columns)):
        missing = set(schema.keys()).difference(set(gdf.columns))

        raise ValueError(f"Columns do not match schema. \nGot: {sorted(schema.keys())}\nMissing columns: {missing}")

    # noinspection PyTypeChecker
    mismatches = [
        (col, dtype, schema[col]) for col, dtype in gdf.dtypes.items() if col in schema and dtypes_match(dtype, schema)
    ]
    if mismatches:
        errors = [f"\n\t`{col}`: got {dtype}, expected {expected}" for col, dtype, expected in mismatches]
        raise ValueError("Invalid types: " + "".join(errors))

    return gdf


def check_geometry_shapes(geometry: gpd.GeoSeries, *shapes: str) -> gpd.GeoSeries:
    geom_types: pd.Series = geometry.geom_type
    is_valid = geom_types.isin(shapes)

    if not is_valid.all():
        invalid = list(geom_types[~is_valid].unique())
        raise ValueError(f"Invalid shapes in geometry. Permitted: {shapes}, found {invalid}")

    return geometry


def check_shape(tensor: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """
    Checks that a Tensor has a given shape, ignores size of dimension if size is set to -1.

    Args:
        tensor (torch.Tensor): the tensor
        shape (tuple[int, ...]): a tuple of dimension sizes

    Raises:
        ValueError: if the shape is invalid

    Returns:
        torch.Tensor: the checked tensor
    """
    if len(tensor.shape) != len(shape):
        raise ValueError(f"Tensor has invalid number of dims, got shape {tensor.shape}, expected {shape}")

    for i, (t, s) in enumerate(zip(tensor.shape, shape)):
        if t != s and s != -1:
            raise ValueError(
                f"Tensor has invalid shape along dimension {i}, got shape {tensor.shape}, expected {shape}"
            )

    return tensor


def gdf_to_polars(gdf: gpd.GeoDataFrame) -> pl.DataFrame:
    return pl.DataFrame(gdf.drop(columns=["geometry"]))


def convert_locations_to_point_geometry(locations_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    check_schema(locations_gdf, LOCATIONS_SCHEMA)

    locations_gdf = locations_gdf.copy()
    locations_gdf.set_geometry(gpd.points_from_xy(locations_gdf["lon"], locations_gdf["lat"], crs=CRS), inplace=True)

    return locations_gdf


def extract_unique_loc_ids(*edge_dfs: TDataFrame) -> list[str]:
    loc_dfs = []
    for edge_df in edge_dfs:
        edge_df = edge_df if isinstance(edge_df, pl.DataFrame) else gdf_to_polars(edge_df)
        edge_df = check_schema(edge_df, EDGE_LIST_SCHEMA, ignore_extra_cols=True)

        loc_dfs.append(edge_df["orig_loc_id"])
        loc_dfs.append(edge_df["dest_loc_id"])

    return pl.concat(loc_dfs).unique().to_list()


def convert_excel_to_parquet(data_path: Path, *files: Path) -> list[Path]:
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
