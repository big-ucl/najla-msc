from collections.abc import Mapping
from typing import TypeVar

import geopandas as gpd
import pandas as pd
import polars as pl
import torch

PandasSchema = Mapping[str, str]
TDataFrame = TypeVar("TDataFrame", gpd.GeoDataFrame, pl.DataFrame)
TSchema = TypeVar("TSchema", pl.Schema, PandasSchema)


def check_schema(df: TDataFrame, schema: pl.Schema) -> TDataFrame:
    """Checks that a DataFrame matches provided Schema.

    Args:
        df (pl.DataFrame | gpd.GeoDataFrame): the dataframe to be checked
        schema (pl.Schema | dict): the schema to check against

    Raises:
        ValueError: if the DataFrame does not match the schema

    Returns:
        pl.DataFrame: the valid DataFrame
    """

    if isinstance(df, pl.DataFrame) and isinstance(schema, pl.Schema):
        return check_polars_schema(df, schema)

    if (isinstance(df, gpd.GeoDataFrame) or isinstance(df, pd.DataFrame)) and isinstance(schema, dict):
        return check_geopandas_schema(df, schema)

    raise ValueError(f"Invalid types: Got {type(df)=}, {type(schema)=}")


def check_polars_schema(df: pl.DataFrame, schema: pl.Schema) -> pl.DataFrame:
    df_items = set(df.schema.items())
    schema_items = set(schema.items())

    missing = schema_items - df_items
    extra = df_items - schema_items

    if missing or extra:
        raise ValueError(f"Schemas do not match:\nMissing elements: {missing}\nExtra elements: {extra}")

    return df


def check_geopandas_schema(gdf: gpd.GeoDataFrame, schema: PandasSchema) -> gpd.GeoDataFrame:
    if set(schema.keys()) != set(gdf.columns):
        raise ValueError(
            f"Columns do not match schema. \nGot: {sorted(schema.keys())}\nExpected: {sorted(gdf.columns)}"
        )

    # noinspection PyTypeChecker
    mismatches = [(col, dtype, schema[col]) for col, dtype in gdf.dtypes.items() if dtype != schema[col]]
    if mismatches:
        errors = [f"\n\t`{col}`: got {dtype}, expected {expected}" for col, dtype, expected in mismatches]
        raise ValueError("Invalid types: " + "".join(errors))

    return gdf


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
