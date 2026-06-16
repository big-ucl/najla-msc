"""
Schema validation, geometry helpers, and I/O utilities.

This module provides low-level helper functions used throughout the project for:

  - Schema validation: ``check_schema``, ``check_polars_schema``,
    ``check_geopandas_schema`` — verify that DataFrames have the expected
    columns and data types before further processing.

  - Geometry helpers: ``check_geometry_shapes``,
    ``convert_locations_to_point_geometry``, ``add_lon_lat_from_centroid`` —
    validate and convert geometric objects between formats.

  - PyTorch tensor validation: ``check_shape`` — ensures tensors have the
    expected dimensionality.

  - I/O helpers: ``gdf_to_polars``, ``extract_unique_loc_ids``,
    ``convert_excel_to_parquet``, ``read_from_parquet`` — common file-read
    and format-conversion operations.

  - Data infrastructure: ``DataFrameStore`` abstract mixin — provides a
    standard cache-directory layout for network data classes.

  - ``invert_mapping``: invert a one-to-many dict to a many-to-one dict.
"""

from abc import ABC
from collections.abc import Mapping
from pathlib import Path
from typing import Hashable, TypeVar

import geopandas as gpd
import pandas as pd
import polars as pl
import torch

from activitygraphs.base import CRS, EDGE_LIST_SCHEMA, LOCATIONS_SCHEMA
from activitygraphs.config import DataConfig

# Type alias for a Pandas/GeoPandas schema represented as a plain dict
# mapping column names to dtype strings (e.g. {"loc_id": "object", "lon": "float64"}).
PandasSchema = Mapping[str, str]

# Generic TypeVar for functions that accept either a Polars or a GeoPandas DataFrame.
# This allows type-checkers to infer that the return type matches the input type.
TDataFrame = TypeVar("TDataFrame", gpd.GeoDataFrame, pl.DataFrame)

# Generic TypeVar for functions that accept either a Polars Schema or a PandasSchema dict.
TSchema = TypeVar("TSchema", pl.Schema, PandasSchema)


def get_project_root(project_root: Path | None = None) -> Path:
    """
    Description: Return a valid Path for the project root directory.  If the
    caller passes an explicit path it is returned unchanged; otherwise the
    current working directory (Path(".")) is used as a sensible default.

    This helper ensures that all downstream code that needs a project root
    always has a non-None Path to work with.

    Input:
      - project_root (Path | None): optional explicit path to the project root.
                                    Pass None (or omit the argument) to use the
                                    current working directory.

    Output:
      - (Path): the resolved project root as a Path object.
    """
    # Use the provided path if given, otherwise fall back to the current directory.
    project_root: Path = project_root if project_root is not None else Path(".")
    return project_root


def check_schema(df: TDataFrame, schema: TSchema, ignore_extra_cols: bool = False) -> TDataFrame:
    """
    Description: Validate that a DataFrame's columns and data types match a
    given schema definition.  Dispatches to either ``check_polars_schema`` or
    ``check_geopandas_schema`` depending on the types of ``df`` and ``schema``.

    This is the main entry point for schema validation used throughout the project.
    Calling it at the start of a function is a defensive programming technique that
    catches data errors early with a clear error message.

    Input:
      - df (pl.DataFrame | gpd.GeoDataFrame): the DataFrame whose schema is to
                                              be validated.
      - schema (pl.Schema | dict[str, str]): the expected schema.  Use a
                                             ``pl.Schema`` for Polars DataFrames
                                             and a plain dict for GeoPandas/Pandas.
      - ignore_extra_cols (bool): if True, columns present in ``df`` but absent
                                  from ``schema`` are silently ignored.  If False
                                  (default), extra columns raise a ValueError.

    Output:
      - (TDataFrame): the input ``df`` unchanged (returned for chaining).

    Raises:
      - ValueError: if the df/schema type combination is unsupported, or if the
                    schema validation fails.
    """

    # Polars DataFrame + Polars Schema → use the Polars-specific validator.
    if isinstance(df, pl.DataFrame) and isinstance(schema, pl.Schema):
        return check_polars_schema(df, schema, ignore_extra_cols=ignore_extra_cols)

    # GeoPandas/Pandas DataFrame + plain dict → use the GeoPandas-specific validator.
    if (isinstance(df, gpd.GeoDataFrame) or isinstance(df, pd.DataFrame)) and isinstance(schema, dict):
        return check_geopandas_schema(df, schema, ignore_extra_cols=ignore_extra_cols)

    # Unsupported combination of types — raise a clear error.
    raise ValueError(f"Invalid types: Got {type(df)=}, {type(schema)=}")


def check_polars_schema(df: pl.DataFrame, schema: pl.Schema, ignore_extra_cols: bool) -> pl.DataFrame:
    """
    Description: Validate a Polars DataFrame against a Polars Schema by comparing
    the set of (column_name, dtype) pairs.  Missing columns (in schema but not in
    df) always raise an error.  Extra columns (in df but not in schema) raise an
    error only when ``ignore_extra_cols=False``.

    Input:
      - df (pl.DataFrame): the Polars DataFrame to validate.
      - schema (pl.Schema): the expected schema — an ordered mapping of column
                            names to Polars dtypes.
      - ignore_extra_cols (bool): if True, columns in ``df`` that are not listed
                                  in ``schema`` do not raise an error.

    Output:
      - (pl.DataFrame): the input ``df`` unchanged (returned for chaining).

    Raises:
      - ValueError: if there are missing columns/dtypes or (when not ignored) extra ones.
    """
    # Convert each schema to a set of (column_name, dtype) pairs for set arithmetic.
    df_items = set(df.schema.items())       # Actual (name, dtype) pairs from the DataFrame.
    schema_items = set(schema.items())      # Expected (name, dtype) pairs from the schema.

    # Columns/types in the expected schema but absent from the DataFrame.
    missing = schema_items - df_items
    # Columns/types present in the DataFrame but not in the expected schema.
    extra = df_items - schema_items

    if missing or (extra and not ignore_extra_cols):
        raise ValueError(
            f"Schemas do not match:\nMissing elements: {missing}\n"
            + (f"Extra elements: {extra}" if not ignore_extra_cols else "")
        )

    return df


def check_geopandas_schema(gdf: gpd.GeoDataFrame, schema: PandasSchema, ignore_extra_cols: bool) -> gpd.GeoDataFrame:
    """
    Description: Validate a GeoPandas or Pandas DataFrame against a column-name to
    dtype-string mapping.  Checks both that the right columns are present and that
    each column's dtype matches the expected string (handling the "str"/"object"
    Pandas equivalence).

    Input:
      - gdf (gpd.GeoDataFrame): the GeoPandas (or Pandas) DataFrame to validate.
      - schema (PandasSchema): dict mapping column names to expected Pandas dtype
                               strings, e.g. {"loc_id": "object", "lon": "float64"}.
      - ignore_extra_cols (bool): if True, columns in ``gdf`` not listed in
                                  ``schema`` are silently accepted.

    Output:
      - (gpd.GeoDataFrame): the input ``gdf`` unchanged (returned for chaining).

    Raises:
      - ValueError: if required columns are missing, extra columns exist (when not
                    ignored), or column dtypes do not match the expected values.
    """

    def dtypes_match(dtype: str, expected: str) -> bool:
        """
        Description: Check whether a Pandas dtype string matches the expected schema
        dtype string.  Handles the Pandas quirk where Python ``str`` columns are
        stored as dtype "object".

        Input:
          - dtype (str): actual dtype string from the DataFrame (e.g. "object").
          - expected (str): expected dtype string from the schema (e.g. "object").

        Output:
          - (bool): True if the dtypes are considered equivalent.
        """
        # "str" and "object" are the same type in Pandas — treat them as equal.
        return (dtype == "str" and expected == "object") or dtype == expected

    # When not ignoring extra columns, the column sets must match exactly.
    if not ignore_extra_cols and set(schema.keys()) != set(gdf.columns):
        raise ValueError(
            f"Columns do not match schema. \nGot: {sorted(gdf.columns)}\nExpected: {sorted(schema.keys())}"
        )

    # When ignoring extra columns, all schema columns must at least be present.
    if ignore_extra_cols and not set(schema.keys()).issubset(set(gdf.columns)):
        # Find which schema columns are missing from the DataFrame.
        missing = set(schema.keys()).difference(set(gdf.columns))

        raise ValueError(f"Columns do not match schema. \nGot: {sorted(schema.keys())}\nMissing columns: {missing}")

    # noinspection PyTypeChecker
    # Build a list of columns whose actual dtype does not match the expected dtype.
    mismatches = [
        (col, dtype, schema[col]) for col, dtype in gdf.dtypes.items() if col in schema and dtypes_match(dtype, schema)
    ]
    if mismatches:
        # Format all mismatches into a single readable error message.
        errors = [f"\n\t`{col}`: got {dtype}, expected {expected}" for col, dtype, expected in mismatches]
        raise ValueError("Invalid types: " + "".join(errors))

    return gdf


def check_geometry_shapes(geometry: gpd.GeoSeries, *shapes: str) -> gpd.GeoSeries:
    """
    Description: Validate that every geometry object in a GeoSeries belongs to
    one of the permitted Shapely geometry type strings.  This is used to guard
    functions that only work with specific geometry kinds (e.g. a function that
    needs point geometries should call this with shapes="Point").

    Input:
      - geometry (gpd.GeoSeries): the GeoSeries of Shapely geometry objects to check.
      - *shapes (str): one or more permitted geometry type strings, e.g. "Point",
                       "Polygon", "MultiPolygon".  Any geometry whose type is not
                       in this list causes a ValueError.

    Output:
      - (gpd.GeoSeries): the input ``geometry`` unchanged (returned for chaining).

    Raises:
      - ValueError: lists the invalid geometry types that were found.
    """
    # Extract the geometry type name (e.g. "Point", "Polygon") for each geometry.
    geom_types: pd.Series = geometry.geom_type
    # Boolean Series: True for geometries whose type is in the permitted set.
    is_valid = geom_types.isin(shapes)

    if not is_valid.all():
        # Collect the unique invalid type names found in the Series.
        invalid = list(geom_types[~is_valid].unique())
        raise ValueError(f"Invalid shapes in geometry. Permitted: {shapes}, found {invalid}")

    return geometry


def check_shape(tensor: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """
    Description: Assert that a PyTorch tensor has the expected shape.  A dimension
    size of -1 in the ``shape`` argument is treated as a wildcard — any actual size
    is accepted for that dimension.  This makes it easy to check batch dimensions
    without knowing the exact batch size at call time.

    Example:
        check_shape(x, (-1, 128))   # x must be 2-D with exactly 128 columns.
        check_shape(y, (32, -1, 3)) # y must be 3-D, first dim=32, last dim=3.

    Input:
      - tensor (torch.Tensor): the tensor whose shape is to be validated.
      - shape (tuple[int, ...]): expected shape.  Use -1 for "any size allowed
                                 in this dimension".

    Output:
      - (torch.Tensor): the input ``tensor`` unchanged (returned for chaining).

    Raises:
      - ValueError: if the number of dimensions does not match, or if any
                    non-wildcard dimension has the wrong size.
    """
    # First check that the number of dimensions is correct.
    if len(tensor.shape) != len(shape):
        raise ValueError(f"Tensor has invalid number of dims, got shape {tensor.shape}, expected {shape}")

    # Then check each dimension individually; skip wildcards (s == -1).
    for i, (t, s) in enumerate(zip(tensor.shape, shape)):
        # t = actual size of dimension i; s = expected size (-1 = don't care).
        if t != s and s != -1:
            raise ValueError(
                f"Tensor has invalid shape along dimension {i}, got shape {tensor.shape}, expected {shape}"
            )

    return tensor


def gdf_to_polars(gdf: gpd.GeoDataFrame) -> pl.DataFrame:
    """
    Description: Convert a GeoPandas GeoDataFrame to a Polars DataFrame by
    dropping the geometry column.  The geometry column contains Shapely objects
    that Polars cannot represent natively, so it must be removed before conversion.
    All other attribute columns are preserved.

    Input:
      - gdf (gpd.GeoDataFrame): the GeoDataFrame to convert.  Must have a column
                                named "geometry" (the standard GeoPandas geometry
                                column name).

    Output:
      - (pl.DataFrame): Polars DataFrame with all columns of ``gdf`` except
                        "geometry".
    """
    # Drop the geometry column (Shapely objects) and wrap the rest in a Polars DataFrame.
    return pl.DataFrame(gdf.drop(columns=["geometry"]))


def convert_locations_to_point_geometry(locations_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Description: Replace the geometry column of a locations GeoDataFrame with
    Point geometries constructed from the ``lon`` and ``lat`` columns.  The
    original geometry may be polygons (zone boundaries); this function replaces
    them with point geometries so that edge-to-edge distance calculations and
    map marker rendering work correctly.

    The input is validated against LOCATIONS_SCHEMA before any modification.

    Input:
      - locations_gdf (gpd.GeoDataFrame): locations table matching LOCATIONS_SCHEMA
                                          (must have "lon" and "lat" columns).

    Output:
      - (gpd.GeoDataFrame): a copy of ``locations_gdf`` where the geometry column
                            contains EPSG:4326 Point objects built from lon/lat.
    """
    # Validate that the input has all required LOCATIONS_SCHEMA columns/types.
    check_schema(locations_gdf, LOCATIONS_SCHEMA)

    # Work on a copy so the original is not modified in place.
    locations_gdf = locations_gdf.copy()
    # Build Point geometries from the lon/lat columns and set them as the active geometry.
    locations_gdf.set_geometry(gpd.points_from_xy(locations_gdf["lon"], locations_gdf["lat"], crs=CRS), inplace=True)

    return locations_gdf


def extract_unique_loc_ids(*edge_dfs: TDataFrame) -> list[str]:
    """
    Description: Gather all unique location IDs that appear in one or more
    edge-list DataFrames.  Each edge connects an origin (``orig_loc_id``) to
    a destination (``dest_loc_id``); this function collects every ID from
    both columns across all provided DataFrames and returns a deduplicated list.

    This is useful when building the graph node set: you need to know every
    location that participates in at least one edge.

    Input:
      - *edge_dfs (pl.DataFrame | gpd.GeoDataFrame): one or more edge-list
                  DataFrames, each conforming to EDGE_LIST_SCHEMA (must have at
                  least ``orig_loc_id`` and ``dest_loc_id`` columns).

    Output:
      - (list[str]): sorted list of unique location ID strings.
    """
    # Accumulate Polars Series of loc_ids from each DataFrame's origin and destination columns.
    loc_dfs = []
    for edge_df in edge_dfs:
        # Convert GeoDataFrames to Polars (drops geometry) so we can use Polars operations.
        edge_df = edge_df if isinstance(edge_df, pl.DataFrame) else gdf_to_polars(edge_df)
        # Validate that the required columns are present (extra columns are fine).
        edge_df = check_schema(edge_df, EDGE_LIST_SCHEMA, ignore_extra_cols=True)

        # Collect origin and destination ID columns separately.
        loc_dfs.append(edge_df["orig_loc_id"])
        loc_dfs.append(edge_df["dest_loc_id"])

    # Concatenate all ID Series, deduplicate, and return as a Python list.
    return pl.concat(loc_dfs).unique().to_list()


def convert_excel_to_parquet(data_path: Path, *files: Path) -> list[Path]:
    """
    Description: Read one or more Excel (.xlsx) files located under ``data_path``
    and save each as a Parquet file with the same stem but a ``.parquet`` suffix.
    Parquet is much faster to read than Excel and is the preferred format for
    processed data in this project.

    Input:
      - data_path (Path): directory that contains the Excel files (and where the
                          output Parquet files will be written).
      - *files (Path): relative file paths (relative to ``data_path``) of the
                       Excel files to convert.

    Output:
      - (list[Path]): list of the new file paths (relative to ``data_path``) for
                      the created Parquet files, in the same order as ``files``.
    """
    # Accumulate output file paths as they are created.
    new_files = []

    for file in files:
        # Read the entire Excel file into a Polars DataFrame.
        df = pl.read_excel(data_path / file)

        # Build the output path: same name as input but with .parquet extension.
        new_file = file.with_suffix(".parquet")
        # Write the Polars DataFrame to parquet format.
        df.write_parquet(data_path / new_file)
        new_files.append(new_file)

    return new_files


def read_from_parquet(path: Path, schema: dict = None) -> pl.DataFrame:
    """
    Description: Read a Parquet file from disk into a Polars DataFrame, with an
    optional dict of column-level dtype overrides.  Use ``schema`` to cast specific
    columns to a different dtype after loading (e.g. to force a column to
    ``pl.Categorical()`` that was saved as ``pl.String``).

    Input:
      - path (Path): filesystem path to the Parquet file to read.
      - schema (dict | None): optional mapping of column name → Polars dtype to
                              override the file's stored dtype.  Columns not listed
                              here keep their original dtype.  Pass None (or omit)
                              to keep all original dtypes.

    Output:
      - (pl.DataFrame): the loaded DataFrame with any requested dtype overrides
                        applied.
    """
    # Default to an empty dict so the schema_overrides kwarg below is always valid.
    schema = {} if schema is None else schema

    # Read the Parquet file using Polars' native reader.
    df = pl.read_parquet(path)
    # Re-wrap with schema_overrides to apply any requested dtype casts.
    return pl.DataFrame(df, schema_overrides=schema)


def add_lon_lat_from_centroid(
    gdf: gpd.GeoDataFrame, index_col: str, lon_name="lon", lat_name="lat"
) -> gpd.GeoDataFrame:
    """
    Description: Add longitude and latitude columns to a GeoDataFrame by computing
    the centroid of each polygon geometry.  To get an accurate centroid, the
    polygons are first re-projected to a local UTM coordinate system (measured in
    metres), the centroid is computed there, and then the result is converted back
    to WGS-84 (EPSG:4326) longitude/latitude degrees.

    This is used to derive a representative point for each zone (e.g. a census
    tract or administrative subsector) so it can be placed on a map or used in
    travel-time calculations.

    Input:
      - gdf (gpd.GeoDataFrame): GeoDataFrame with polygon geometries and an index
                                column specified by ``index_col``.
      - index_col (str): name of the column used as the row identifier when joining
                         back the centroid coordinates (e.g. "loc_id").
      - lon_name (str): name to give the new longitude column. Defaults to "lon".
      - lat_name (str): name to give the new latitude column. Defaults to "lat".

    Output:
      - (gpd.GeoDataFrame): copy of ``gdf`` with two new columns (``lon_name``
                            and ``lat_name``) containing WGS-84 centroid coordinates.
                            The temporary "centroid" helper column is dropped.
    """
    # Work on a copy so the original GeoDataFrame is not modified.
    gdf = gdf.copy()

    # Automatically determine the best UTM projection for this data extent.
    # UTM uses metres, which gives accurate centroid calculations.
    projected_crs = gdf.estimate_utm_crs()
    # Project to UTM, compute the polygon centroid for each row, then back to WGS-84.
    centroids = gdf.to_crs(projected_crs).set_index(index_col).centroid.to_crs(CRS)
    # Wrap the centroid GeoSeries in a GeoDataFrame so we can join it.
    centroids = gpd.GeoDataFrame(centroids, columns=["centroid"])

    # Join the centroid Points back to the original table on index_col.
    gdf = gdf.join(centroids, on=index_col)
    # Extract the X (longitude) and Y (latitude) coordinates from the centroid Point.
    gdf[lon_name] = gdf["centroid"].x
    gdf[lat_name] = gdf["centroid"].y

    # Remove the temporary centroid geometry column — lon/lat columns contain the info.
    return gdf.drop(columns=["centroid"])


class DataFrameStore(ABC):
    """
    Description: Abstract mixin class that provides a standard cache-directory
    naming convention for concrete data-store subclasses (e.g. network graph
    data classes that cache their processed DataFrames to disk).

    When a class inherits from DataFrameStore and calls ``_dirs()``, it receives:
      - the project root Path (for constructing absolute file paths elsewhere)
      - a ``data_dir`` Path: ``<project_root>/<processed>/<ClassName>[-<name>]/``

    The optional ``name`` argument allows the same class to cache multiple
    named variants (e.g. different date ranges or geographic subsets).

    This class is abstract (ABC) so it cannot be instantiated directly — only
    concrete subclasses that implement any required abstract methods can be used.
    """

    @classmethod
    def _dirs(cls, cfg: DataConfig, project_root: Path | None = None, name: str | None = None) -> tuple[Path, Path]:
        """
        Description: Compute the project root and the class-specific data
        cache directory.  The cache directory is named after the concrete
        class that calls this method, with an optional suffix.

        Input:
          - cfg (DataConfig): dataset configuration providing the ``paths.processed``
                              directory relative to the project root.
          - project_root (Path | None): absolute path to the repository root.
                                        If None, the current working directory is used.
          - name (str | None): optional suffix appended to the directory name to
                               distinguish multiple cached variants.  If None, no
                               suffix is added.

        Output:
          - (tuple[Path, Path]): a 2-tuple of:
              [0] project_root (Path): the resolved project root directory.
              [1] data_dir (Path): the cache directory for this class instance
                                   (``<project_root>/<processed>/<ClassName>[-<name>]``).
        """
        # Use the provided root or fall back to the current working directory.
        project_root: Path = project_root if project_root is not None else Path(".")
        # Build an optional suffix string for the directory name.
        suffix = "" if name is None else f"-{name}"
        # Construct the full cache directory path: processed/<ClassName>[-name]
        data_dir = project_root / cfg.paths.processed / f"{cls.__name__}{suffix}"

        return project_root, data_dir


def invert_mapping(mapping: dict[Hashable, list[Hashable]]) -> dict[Hashable, Hashable]:
    """
    Description: Invert a one-to-many dictionary (where each key maps to a *list*
    of values) to produce a many-to-one dictionary (where each former value maps
    back to its original key).

    Example:
        mapping = {"A": [1, 2], "B": [3, 4]}
        invert_mapping(mapping)  →  {1: "A", 2: "A", 3: "B", 4: "B"}

    This function is used when, for example, you have a mapping from zone types
    to lists of zone IDs and you need the reverse lookup (zone ID → zone type).

    Input:
      - mapping (dict[Hashable, list[Hashable]]): the one-to-many mapping to invert.
                Each key must map to a list (or other iterable) of values.  Values
                must be unique across all lists — if any value appears under more
                than one key a ValueError is raised.

    Output:
      - (dict[Hashable, Hashable]): the inverted many-to-one mapping.

    Raises:
      - ValueError: if any value appears more than once across the input lists,
                    since the inversion would be ambiguous.
    """
    # The result dict maps each value back to the key it came from.
    inversion = {}

    for k, vs in mapping.items():
        for v in vs:
            # Guard against duplicate values — each value must map to exactly one key.
            if v in inversion:
                raise ValueError(f"Duplicate values in mapping for {k=}: v1={v} and v2={inversion[v]}")

            # Record this value → key mapping in the inversion dict.
            inversion[v] = k

    return inversion
