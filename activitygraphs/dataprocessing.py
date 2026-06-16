"""
Build Network Graphs and per-user PyG tensors from NetworkData.

This module is the main data-processing pipeline that takes a ``NetworkData`` object
(containing cleaned survey data) and produces the PyTorch Geometric (PyG) graphs and
tensors used for model training. The pipeline has three main stages:

  A. **Network Graph construction**: Build a spatial contiguity graph of subsectors,
     enriched with census statistics (population, jobs), Overture POI counts, and land-use
     fractions. Nodes are subsectors; edges connect adjacent or nearby subsectors.

  B. **Spatial demographics**: For each user, create per-node binary indicators
     (is_home, is_work, is_edu, is_visited) and stack them into tensors of shape
     [n_users, n_nodes, n_features].

  C. **PyG graph creation**: Convert the network graph and per-user annotations into
     PyTorch Geometric ``Data`` objects. Each user gets their own PyG graph with node
     features, edge features, labels (is_visited), and positional encodings.

Results are cached to disk as Parquet and Pickle files to avoid recomputation.
"""

import pickle
from pathlib import Path
from typing import Callable, cast

import city2graph as c2g
import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch_geometric as pyg
import torch_geometric.transforms as T
from joblib import Parallel, delayed


from activitygraphs.base import CRS
from activitygraphs.config import (
    DataConfig,
    GenevaStatsInputs,
    StatsInputs,
    TorontoStatsInputs,
    TorontoDataConfig,
    GenevaDataConfig,
)
from activitygraphs.data.geneva import GenevaData
from activitygraphs.data.overture import Overture
from activitygraphs.data.statistics import (
    add_geneva_population_job_statistics,
    add_toronto_population_job_statistics,
)
from activitygraphs.data.toronto import TorontoData
from activitygraphs.network import NetworkData
from activitygraphs.utils import get_project_root

# Columns from the network nodes GeoDataFrame that should NOT be included as node
# features in the PyG graph. These are either metadata (names, geometry), labels
# (is_visited), or per-user columns that would leak information.
COLS_EXCLUDED_FROM_FEATURES = [
    "loc_name",          # Human-readable location name: not useful as a numeric feature
    "type",              # Location type string (e.g. "subsector"): not numeric
    "lon",               # Longitude: redundant with geometry; excluded to avoid data leakage
    "lat",               # Latitude: same reason as lon
    "is_work",           # Per-user work indicator: user-specific, excluded from shared graph
    "is_edu",            # Per-user education indicator: user-specific
    "is_visited",        # Target label: excluded from features to prevent leakage
    "purpose",           # Visit purposes string: user-specific metadata
    "geometry",          # Shapely geometry object: not a numeric tensor feature
    "original_geometry", # Backup geometry column sometimes added during spatial joins
]

# Single source of truth for the per-user spatial feature columns, in the order they are
# appended after the network node features in the concatenated node feature matrix `x`.
# `is_home` must stay first so that the derived home-column index (see ActivityDataset)
# remains valid for `extract_is_home` / the conditional baseline.
SPATIAL_FEATURE_NAMES = ["is_home"]
# Note: is_work and is_edu are computed but currently excluded from SPATIAL_FEATURE_NAMES
# to keep the feature set minimal; only is_home is used as the spatial conditioning signal.

# =========================================
# A. Network graph
# =========================================

# Type alias for the callable that builds the network graph for a specific dataset.
# Takes (locations GeoDataFrame, Overture data, statistics config) and returns
# a (nodes GeoDataFrame, edges GeoDataFrame) tuple.
type NetworkGraphBuilder = Callable[
    [gpd.GeoDataFrame, Overture, StatsInputs], tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
]


def load_network_graph(
    data: NetworkData,
    cfg: DataConfig,
    build_network_graph: NetworkGraphBuilder,
    project_root: Path | None = None,
    name: str = "NetworkGraph",
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Description: Load the Network Graph from a Parquet cache if it already exists,
    otherwise build it from scratch and save it to disk. This avoids recomputing
    the expensive spatial join, census enrichment, and POI aggregation on every run.

    Cached files are stored at ``<project_root>/<cfg.paths.processed>/<name>/nodes.parquet``
    and ``.../<name>/edges.parquet``.

    Input:
      - data (NetworkData): Dataset instance (GenevaData or TorontoData) providing locations.
      - cfg (DataConfig): Configuration object with file paths and input settings.
      - build_network_graph (NetworkGraphBuilder): Dataset-specific callable that builds
        the node/edge GeoDataFrames. E.g. ``build_gva_network_graph`` or
        ``build_toronto_network_graph``.
      - project_root (Path | None): Root directory of the project. Defaults to current dir.
      - name (str): Name of the subdirectory under ``processed/`` used for caching.

    Output:
      - (tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]): ``(network_nodes, network_edges)``
        where ``network_nodes`` has one row per subsector (with census + POI + land-use
        features) and ``network_edges`` has one row per spatial adjacency edge.
    """
    # Resolve the project root (falls back to "." if not specified)
    root = get_project_root(project_root)
    # Directory where nodes and edges Parquet files will be cached
    network_path = root / cfg.paths.processed / name
    # Individual file paths for the two GeoDataFrames
    nodes_path = network_path / "nodes.parquet"
    edges_path = network_path / "edges.parquet"

    # If both cache files exist, load and return them immediately
    if nodes_path.exists() and edges_path.exists():
        network_nodes = gpd.read_parquet(nodes_path)
        network_edges = gpd.read_parquet(edges_path)

        return network_nodes, network_edges

    # Download or load Overture POI and land-use data for the study area
    overture = Overture.load(data.locations_gdf, cfg.inputs.overture, root)

    # Filter the dataset to subsector-level locations only (the finest spatial unit)
    subsector_data = data.with_filter("subsector")
    # Build the graph using the dataset-specific builder function
    network_nodes, network_edges = build_network_graph(subsector_data.locations_gdf, overture, cfg.inputs.statistics)

    # Save to Parquet for future runs
    network_path.mkdir(parents=True, exist_ok=True)
    network_nodes.to_parquet(nodes_path)
    network_edges.to_parquet(edges_path)

    return network_nodes, network_edges


def load_gva_network_graph(
    gva_data: GenevaData,
    cfg: DataConfig,
    project_root: Path | None = None,
    name: str = "NetworkGraph",
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Description: Convenience wrapper around ``load_network_graph`` that pre-binds the
    Geneva-specific graph builder (``build_gva_network_graph``). Handles island subsectors
    that are not contiguous with the mainland via a kNN connection step.

    Input:
      - gva_data (GenevaData): Geneva dataset instance providing subsector locations.
      - cfg (DataConfig): Geneva configuration with file paths and statistics settings.
      - project_root (Path | None): Root directory of the project.
      - name (str): Cache subdirectory name. Defaults to ``"NetworkGraph"``.

    Output:
      - (tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]): ``(network_nodes, network_edges)``
        for the Geneva study area.
    """
    return load_network_graph(gva_data, cfg, build_gva_network_graph, project_root, name)


def load_toronto_network_graph(
    toronto_data: TorontoData,
    cfg: DataConfig,
    project_root: Path | None = None,
    name: str = "NetworkGraph",
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Description: Convenience wrapper around ``load_network_graph`` that pre-binds the
    Toronto-specific graph builder (``build_toronto_network_graph``).

    Input:
      - toronto_data (TorontoData): Toronto dataset instance providing census tract locations.
      - cfg (DataConfig): Toronto configuration with file paths and statistics settings.
      - project_root (Path | None): Root directory of the project.
      - name (str): Cache subdirectory name. Defaults to ``"NetworkGraph"``.

    Output:
      - (tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]): ``(network_nodes, network_edges)``
        for the Toronto CMA study area.
    """
    return load_network_graph(toronto_data, cfg, build_toronto_network_graph, project_root, name)


def build_gva_network_graph(
    locations: gpd.GeoDataFrame,
    overture: Overture,
    stats_cfg: StatsInputs,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Description: Build the Geneva spatial contiguity graph enriched with census statistics,
    Overture POI counts, and land-use fractions. The graph captures which subsectors share
    a border (contiguity). Because some Geneva subsectors are physical islands (e.g. in the
    lake), they have no contiguous neighbours and are connected via a kNN graph instead.

    Input:
      - locations (gpd.GeoDataFrame): Subsector-level locations for Geneva (polygon geometry).
        Must have a ``loc_id`` column.
      - overture (Overture): Loaded Overture Maps data (POIs and land use).
      - stats_cfg (StatsInputs): Path configuration for the Geneva census statistics files.
        Expected to be a ``GenevaStatsInputs`` instance.

    Output:
      - (tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]): ``(nodes_gdf, edges_gdf)`` in the
        project CRS. Nodes have demographic and spatial features; edges have a ``weight``
        column (contiguity or kNN distance).
    """
    # Project to a local UTM CRS for accurate area and distance calculations
    utm_crs = locations.estimate_utm_crs()
    locations = locations.to_crs(utm_crs)

    # Narrow the type so IDEs know we have GenevaStatsInputs-specific attributes
    stats_cfg: GenevaStatsInputs

    # Enrich locations with population and job density from Geneva census data
    network_locations = add_geneva_population_job_statistics(locations, stats_cfg)
    # Add POI counts (one column per Overture category, normalised by area)
    network_locations = overture.add_poi_counts(network_locations)
    # Add land-use fractions (one column per Overture subtype, as fraction of area)
    network_locations = overture.add_land_uses(network_locations)
    # Set loc_id as the index so city2graph uses it as node ID
    network_locations = network_locations.set_index("loc_id")

    # Build the contiguity graph: nodes = subsector centroids, edges = shared borders
    nodes, edges = c2g.contiguity_graph(network_locations, set_point_nodes=True)

    # Connect disconnected subsectors to main graph
    # These subsectors are physically separated from the mainland (e.g. islands in the lake)
    # and would have no edges in a pure contiguity graph.
    island_loc_ids = [
        "subsector-174",
        "subsector-141",
        "subsector-10",
        "subsector-325",
    ]
    # These subsectors border the lake but are effectively separated from the main cluster
    cross_lake_loc_ids = ["subsector-243", "subsector-261", "subsector-54"]
    # Separate island nodes from the mainland
    island_nodes = nodes[nodes.index.isin(island_loc_ids)]
    # Mainland = all nodes that are neither islands nor cross-lake subsectors
    mainland_nodes = nodes[~nodes.index.isin(island_loc_ids) & ~nodes.index.isin(cross_lake_loc_ids)]

    # Build kNN (k=3) edges from each island to its 3 nearest mainland subsectors
    _, island_edges = c2g.knn_graph(island_nodes, k=3, target_gdf=mainland_nodes)
    # Reset index so we can access "source" and "target" as regular columns
    island_edges = island_edges.reset_index()
    # The source/target columns in knn_graph output are tuples; extract the second element (loc_id)
    island_edges["source"] = island_edges["source"].str[1]
    island_edges["target"] = island_edges["target"].str[1]
    # Restore the multi-index format expected by city2graph
    island_edges = island_edges.set_index(["source", "target"])

    # Convert back to project CRS (WGS84 / EPSG:4326)
    nodes = nodes.to_crs(CRS)
    # Merge contiguity edges with the island kNN edges
    edges: gpd.GeoDataFrame = pd.concat([edges, island_edges])
    edges = edges.to_crs(CRS)

    return nodes, edges


def build_toronto_network_graph(
    locations: gpd.GeoDataFrame,
    overture: Overture,
    stats_cfg: StatsInputs,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Description: Build the Toronto spatial contiguity graph enriched with census statistics,
    Overture POI counts, and land-use fractions. Unlike Geneva, Toronto census tracts form
    a single contiguous area within the CMA, so no island-correction step is needed.

    Input:
      - locations (gpd.GeoDataFrame): Census-tract (subsector) locations for Toronto (polygon).
        Must have a ``loc_id`` column.
      - overture (Overture): Loaded Overture Maps data (POIs and land use).
      - stats_cfg (StatsInputs): Path configuration for the Toronto census CSV files.
        Expected to be a ``TorontoStatsInputs`` instance.

    Output:
      - (tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]): ``(nodes_gdf, edges_gdf)`` in the
        project CRS. Nodes have demographic and spatial features; edges represent
        contiguous borders between census tracts.
    """
    # Project to local UTM for accurate spatial calculations
    utm_crs = locations.estimate_utm_crs()
    locations = locations.to_crs(utm_crs)

    # Narrow the type for IDE type checking
    stats_cfg: TorontoStatsInputs

    # Enrich with population and job density from Canadian census data
    network_locations = add_toronto_population_job_statistics(locations, stats_cfg)
    # Add POI density columns from Overture Maps (normalised by area)
    network_locations = overture.add_poi_counts(network_locations)
    # Add land-use fraction columns from Overture Maps
    network_locations = overture.add_land_uses(network_locations)
    # Use loc_id as the graph node identifier
    network_locations = network_locations.set_index("loc_id")

    # Build spatial contiguity graph (shared-border adjacency)
    nodes, edges = c2g.contiguity_graph(network_locations, set_point_nodes=True)

    # Convert back to project CRS
    nodes = nodes.to_crs(CRS)
    # Wrap edges in pd.concat for consistency (no island edges needed for Toronto)
    edges: gpd.GeoDataFrame = pd.concat([edges])
    edges = edges.to_crs(CRS)

    return nodes, edges


# =========================================
# B. Spatial demographics
# =========================================


def add_user_cols(
    user_id: str,
    network_nodes: gpd.GeoDataFrame,
    location_visits: pl.DataFrame,
    home_locations: pl.DataFrame,
    work_locations: pl.DataFrame,
    edu_locations: pl.DataFrame,
) -> gpd.GeoDataFrame:
    """
    Description: Annotate a copy of the shared network nodes GeoDataFrame with
    per-user binary indicator columns and a ``purpose`` metadata column. This
    produces a user-specific node feature matrix for building per-user PyG graphs.

    The indicators are integer-valued (0 or 1):
      - ``is_home``: 1 if this node is the user's home location.
      - ``is_work``: 1 if the user visited this node for work purposes.
      - ``is_edu``: 1 if the user visited this node for education purposes.
      - ``is_visited``: 1 if the user visited this node at all (any purpose).
      - ``purpose``: comma-joined string listing all activity purposes at this node.

    Input:
      - user_id (str): The user whose indicators should be computed.
      - network_nodes (gpd.GeoDataFrame): Shared network node features (one row per subsector).
        Must have ``loc_id`` as index or as a column.
      - location_visits (pl.DataFrame): All users' visit records. Columns: user_id, purpose,
        loc_id, num_visits. Filtered to this user internally.
      - home_locations (pl.DataFrame): All users' home locations. Columns: user_id, loc_id.
      - work_locations (pl.DataFrame): All users' work locations. Columns: user_id, loc_id.
      - edu_locations (pl.DataFrame): All users' education locations. Columns: user_id, loc_id.

    Output:
      - (gpd.GeoDataFrame): A copy of ``network_nodes`` with the five additional columns.
        Indexed by ``loc_id`` and sorted alphabetically.
    """
    # Work on a copy to avoid mutating the shared network_nodes GeoDataFrame
    network_nodes = network_nodes.copy().reset_index()

    # Extract this user's visited, home, work, and edu location IDs as Python lists
    user_visits = location_visits.filter(user_id=user_id)["loc_id"].to_list()
    home_location = home_locations.filter(user_id=user_id)["loc_id"].to_list()
    work_location = work_locations.filter(user_id=user_id)["loc_id"].to_list()
    edu_location = edu_locations.filter(user_id=user_id)["loc_id"].to_list()

    # Binary indicator: 1 if this node is in the user's home location list, else 0
    network_nodes["is_home"] = network_nodes["loc_id"].isin(home_location).astype(int)
    # Binary indicator: 1 if this node is one of the user's work locations
    network_nodes["is_work"] = network_nodes["loc_id"].isin(work_location).astype(int)
    # Binary indicator: 1 if this node is one of the user's education locations
    network_nodes["is_edu"] = network_nodes["loc_id"].isin(edu_location).astype(int)
    # Binary indicator: 1 if the user visited this node at all (any purpose)
    network_nodes["is_visited"] = network_nodes["loc_id"].isin(user_visits).astype(int)

    # Build a "purpose" string column: join all activity purposes at this location for this user
    visit_purposes = (
        location_visits
        .filter(user_id=user_id)
        .select("loc_id", pl.col("purpose"))
        .group_by("loc_id")
        .agg(pl.col("purpose").str.join(", "))  # E.g. "home, work" if both purposes recorded
        .to_pandas()
    )
    # Left-join purpose strings onto nodes (unvisited nodes get NaN)
    network_nodes = network_nodes.merge(visit_purposes, on="loc_id", how="left")
    # Restore loc_id as the index and sort deterministically
    network_nodes = network_nodes.set_index("loc_id").sort_index()

    return network_nodes


def create_spatial_demographics(data: NetworkData) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Description: Build two 3-D tensors that encode per-user spatial information over all
    network nodes. These tensors are used by the model to condition predictions on each
    user's known activity locations.

      - ``spatial_features``: Contains ``is_home`` (and potentially is_work, is_edu) indicators.
        Appended to the shared node feature matrix ``data.x`` during model forward pass.
      - ``spatial_labels``: Contains ``is_visited`` (binary), the prediction target. Stored
        as ``data.y`` in each per-user PyG graph.

    Input:
      - data (NetworkData): Dataset with home_locations, work_locations, edu_locations,
        location_visits, user_ids, and locations_df attributes populated.

    Output:
      - (tuple[torch.Tensor, torch.Tensor]):
          - spatial_features: Float32 tensor of shape [n_users, n_nodes, n_features].
            ``n_features`` equals ``len(SPATIAL_FEATURE_NAMES)`` (currently 1: is_home).
          - spatial_labels: Float32 tensor of shape [n_users, n_nodes, 1].
            Entry [u, n, 0] = 1.0 if user u visited node n, else 0.0.
    """
    # Create a location ordering: row index "loc_order" ensures consistent node ordering
    locations = data.locations_df.select("loc_id").with_row_index("loc_order")
    # Cross-join: every (user_id, loc_id) combination, lazily evaluated
    all_users_and_locs = data.user_ids.to_frame().join(locations, how="cross").lazy()

    # Tensor dimensions
    n_users = len(data.user_ids)   # Number of unique users in the filtered dataset
    n_locs = len(locations)        # Number of network nodes (subsectors)

    # Build spatial features (i.e. will be appended to PyG data.x)
    feature_df = (
        all_users_and_locs
        .pipe(add_indicator_column, data.home_locations, "is_home")   # 1 if user's home node
        .pipe(add_indicator_column, data.work_locations, "is_work")   # 1 if user's work node
        .pipe(add_indicator_column, data.edu_locations, "is_edu")     # 1 if user's edu node
        .sort("user_id", "loc_order")  # Sort so rows align with node ordering
        .drop("loc_order")             # Remove the helper ordering column
        .collect()                     # Materialise the lazy DataFrame
    )

    # Only include the feature columns defined in SPATIAL_FEATURE_NAMES (currently ["is_home"])
    feature_cols = SPATIAL_FEATURE_NAMES
    # Reshape flat [n_users * n_locs, n_features] tensor to [n_users, n_locs, n_features]
    spatial_features = torch.tensor(feature_df.select(feature_cols).to_numpy(), dtype=torch.float32).reshape(
        n_users, n_locs, len(feature_cols)
    )

    # Build spatial demographics labels (i.e. will form PyG data.y)
    label_df = (
        all_users_and_locs
        .pipe(add_indicator_column, data.location_visits, "is_visited")  # 1 if user visited
        .sort("user_id", "loc_order")
        .drop("loc_order")
        .collect()
    )

    # Target columns: only "is_visited" (binary prediction target)
    label_cols = ["is_visited"]
    # Shape: [n_users, n_locs, 1]
    spatial_labels = torch.tensor(label_df.select(label_cols).to_numpy(), dtype=torch.float32).reshape(
        n_users, n_locs, len(label_cols)
    )

    return spatial_features, spatial_labels


def create_home_distances(data, network_nodes: gpd.GeoDataFrame) -> torch.Tensor:
    """
    Description: Build a 3-D tensor of Euclidean distances (in metres) from each user's
    home location to every other network node. Uses centroid-to-centroid straight-line
    distances in a local UTM projection for accuracy.

    This tensor can be used as an additional spatial feature to help the model learn
    distance-decay effects (e.g. people tend to visit locations closer to home more often).

    Input:
      - data: Any NetworkData instance with ``home_locations`` (user_id -> loc_id) and
        ``user_ids`` attributes.
      - network_nodes (gpd.GeoDataFrame): Network nodes indexed by loc_id, with Point or
        Polygon geometry. The geometry centroid is used as the node's coordinate.

    Output:
      - (torch.Tensor): Float32 tensor of shape [n_users, n_nodes, 1].
        Entry [u, n, 0] = Euclidean distance in metres from user u's home to node n.
    """
    # Project to UTM for Euclidean distance calculations in metres
    nodes = network_nodes.sort_index().to_crs(network_nodes.estimate_utm_crs())
    # Get (x, y) coordinate pairs for all node centroids as a pandas DataFrame
    coords = nodes.geometry.get_coordinates()

    # Build a dict: user_id -> home loc_id for O(1) lookup inside the loop
    home_by_user = dict(data.home_locations.iter_rows())

    # Tensor dimensions
    n_users, n_nodes = len(data.user_ids), len(coords)
    # Pre-allocate the output array with zeros; shape [n_users, n_nodes, 1]
    distances = np.zeros((n_users, n_nodes, 1), dtype=np.float32)

    # For each user, compute distance from their home node to every other node
    for user_idx, user_id in enumerate(data.user_ids):
        # Look up this user's home location ID
        home_loc_id = home_by_user[user_id]
        # Euclidean norm between every node's coordinate and the home coordinate
        # coords.loc[home_loc_id] is the (x, y) row for the home node
        distances[user_idx, :, 0] = np.linalg.norm(coords - coords.loc[home_loc_id], axis=1)

    # Convert from numpy to PyTorch tensor
    return torch.from_numpy(distances)


def add_indicator_column(feature_df: pl.LazyFrame, indicator_df: pl.DataFrame, col_name: str):
    """
    Description: Add a boolean indicator column to a LazyFrame by left-joining against
    a reference DataFrame. The column is ``True`` for rows whose (user_id, loc_id) pair
    appears in ``indicator_df``, and ``False`` for all other rows (missing values filled).

    This is a functional helper used in a ``pipe`` chain inside ``create_spatial_demographics``.

    Input:
      - feature_df (pl.LazyFrame): The all-users-and-locations cross-join LazyFrame.
        Must have columns: user_id, loc_id.
      - indicator_df (pl.DataFrame): Table of positive cases (user_id, loc_id pairs that
        should get True). E.g. home_locations, work_locations, or location_visits.
      - col_name (str): Name of the new boolean column to add. E.g. ``"is_home"``.

    Output:
      - (pl.LazyFrame): Same as ``feature_df`` but with the new boolean column appended.
        Null values (no match in indicator_df) are replaced with False.
    """
    # Build a three-column lazy frame: user_id, loc_id, col_name=True for all positive cases
    true_indicators = indicator_df.select("user_id", "loc_id", pl.lit(True).alias(col_name)).unique().lazy()

    # Left join: rows without a match get null in col_name, then fill null with False
    return feature_df.join(true_indicators, on=["user_id", "loc_id"], how="left").with_columns(
        pl.col(col_name).fill_null(False)
    )


# =========================================
# C. Individual demographics
# =========================================


def create_individual_demographics(data: NetworkData) -> torch.Tensor:
    """
    Description: Build a 2-D tensor of individual-level demographic features for each user.
    These are user-level attributes (not per-node) such as household size, car ownership,
    or transit pass. For datasets that don't include demographics (e.g. Geneva), a dummy
    tensor of ones is returned so the model can still run without special-casing.

    Input:
      - data (NetworkData): Dataset with a ``users_df`` attribute that contains demographic
        columns beyond ``user_id`` and ``home_loc_id``.

    Output:
      - (torch.Tensor): Float32 tensor of shape [n_users, n_demo_features].
        If no demographic columns exist, shape is [n_users, 1] filled with ones.
    """
    # Drop the ID columns; only keep the numeric demographic feature columns
    indiv_demographics = data.users_df.drop("user_id", "home_loc_id")

    # Geneva has no individual demographics; return a dummy all-ones tensor
    if len(indiv_demographics) == 0:
        return torch.ones((len(data.user_ids), 1), dtype=torch.float32)

    # Convert the remaining demographic columns to a float32 tensor
    return torch.tensor(indiv_demographics.to_numpy(), dtype=torch.float32)


# =========================================
# PyG / Torch creation
# =========================================


def convert_to_torch(
    data: NetworkData, network_nodes: gpd.GeoDataFrame, network_edges: gpd.GeoDataFrame
) -> tuple[pyg.data.Data | pyg.data.HeteroData, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Description: Convert the network graph GeoDataFrames and NetworkData into the PyTorch
    Geometric (PyG) objects and tensors required by the model. This function orchestrates
    the full tensor-building pipeline.

    Input:
      - data (NetworkData): Filtered dataset (subsector level) providing user/location data.
      - network_nodes (gpd.GeoDataFrame): Enriched subsector nodes (one row per node).
        Columns include census stats, POI counts, land-use fractions.
      - network_edges (gpd.GeoDataFrame): Spatial adjacency edges (one row per edge).
        Must include a ``weight`` column.

    Output:
      - (tuple of 5 elements):
          1. network_graph (pyg.data.Data or HeteroData): Shared graph with node features
             (census + POI + land-use) and edge features (weight). No per-user data here.
          2. spatial_features (torch.Tensor): Shape [n_users, n_nodes, n_feat].
             Per-user home/work/edu indicators.
          3. spatial_labels (torch.Tensor): Shape [n_users, n_nodes, 1].
             Binary is_visited label per user per node.
          4. demographics (torch.Tensor): Shape [n_users, n_demo_features].
             Individual demographic attributes per user.
          5. distances (torch.Tensor): Shape [n_users, n_nodes, 1].
             Euclidean distance from each user's home to every node (metres).
    """
    # Compute per-user spatial feature and label tensors
    spatial_features, spatial_labels = create_spatial_demographics(data)
    # Compute per-user demographic feature tensor
    demographics = create_individual_demographics(data)
    # Compute per-user distance-from-home tensor
    distances = create_home_distances(data, network_nodes)

    # Select node feature columns (exclude metadata and per-user columns)
    node_feature_cols = [col for col in network_nodes.columns if col not in COLS_EXCLUDED_FROM_FEATURES]

    # Convert the GeoDataFrames to a PyG Data object with node and edge features
    network_graph = c2g.gdf_to_pyg(
        network_nodes,
        network_edges,
        node_feature_cols=node_feature_cols,  # Shared census + POI + land-use features
        edge_feature_cols=["weight"],          # Edge weight (contiguity/distance)
        keep_geom=False,                       # Don't store geometry in the PyG object
        device="cpu",                          # Keep tensors on CPU for now
    )

    return network_graph, spatial_features, spatial_labels, demographics, distances


def load_pyg_graphs(
    network_data: NetworkData,
    network_nodes: gpd.GeoDataFrame,
    network_edges: gpd.GeoDataFrame,
    cfg: DataConfig,
    project_root: Path | None = None,
    name: str = "Graphs",
):
    """
    Description: Load a list of per-user PyG graphs from a Pickle cache. On the first call,
    builds all per-user graphs in parallel and saves them to disk; subsequent calls load
    the cached file directly.

    Per-user PyG graphs contain node features (census + POI + land-use + user-specific
    indicators), edge features (weight), node labels (is_visited), and positional encodings.

    Input:
      - network_data (NetworkData): Filtered dataset providing user IDs and location visits.
      - network_nodes (gpd.GeoDataFrame): Enriched subsector nodes.
      - network_edges (gpd.GeoDataFrame): Spatial adjacency edges.
      - cfg (DataConfig): Configuration with path to the PyG datasets directory.
      - project_root (Path | None): Project root. Defaults to current directory.
      - name (str): Base name for the cached pickle file (without extension).

    Output:
      - (list[pyg.data.Data]): List of per-user PyG Data objects, one per user.
        Each object has attributes: x (node features), edge_index, edge_attr,
        y (is_visited labels), user_id (string), and positional encoding columns.
    """
    # Path to the pickle cache file
    dataset_path = get_project_root(project_root) / cfg.paths.pyg_datasets / f"{name}.pickle"

    # If the cache exists, load and return without recomputing
    if dataset_path.exists():
        with open(dataset_path, "rb") as f:
            return pickle.load(f)

    # Build all per-user graphs (parallel via joblib)
    graphs = convert_to_pyg_graphs(network_data, network_nodes, network_edges)

    # Save the list of graphs to disk as a pickle file
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dataset_path, "wb") as f:
        pickle.dump(graphs, f)

    return graphs


def convert_to_pyg_graphs(
    network_data: NetworkData,
    network_nodes: gpd.GeoDataFrame,
    network_edges: gpd.GeoDataFrame,
):
    """
    Description: Build one annotated PyG Data object per user, in parallel using joblib.
    Each graph is the shared network topology (same nodes and edges for all users) but with
    user-specific node annotations (is_home, is_work, is_edu, is_visited, purpose).

    Input:
      - network_data (NetworkData): Dataset with user_ids, location_visits, home_locations,
        work_locations, and edu_locations attributes.
      - network_nodes (gpd.GeoDataFrame): Shared enriched subsector nodes.
      - network_edges (gpd.GeoDataFrame): Shared spatial adjacency edges.

    Output:
      - (list[pyg.data.Data]): List of per-user PyG Data objects, one per user,
        in the same order as ``network_data.user_ids``.
    """
    # Extract the shared DataFrames once; passed into each worker to avoid repeated filtering
    visits = network_data.location_visits          # All users' visit records
    home_locations = network_data.home_locations   # All users' home locations
    work_locations = network_data.work_locations   # All users' work locations
    edu_locations = network_data.edu_locations     # All users' education locations

    # Inner function that builds one user's graph; used by joblib's delayed()
    def build_graph(user_id: str):
        """
        Description: Inner helper function that wraps build_user_pyg_graph with the
        shared DataFrames already captured in the enclosing scope (visits,
        home_locations, work_locations, edu_locations, network_nodes, network_edges).
        This closure pattern allows joblib's Parallel/delayed to serialise only the
        user_id argument per call, instead of re-passing the large shared DataFrames
        to every worker process.

        Input:
          - user_id (str): The unique identifier of the user for whom to build a PyG
                graph. Used to filter the shared visit/home/work/edu DataFrames to
                this user's rows inside build_user_pyg_graph.

        Output:
          - (pyg.data.Data): A PyG Data object for this user containing:
                - x: node feature matrix (shared network features + user-specific annotations)
                - edge_index: COO adjacency in [2, n_edges] format
                - edge_attr: edge weights/features in [n_edges, 1] format
                - y: binary is_visited node labels in [n_nodes, 1] format
                - user_id: the user's ID stored as a graph-level attribute
        """
        return build_user_pyg_graph(
            user_id,
            network_nodes,
            network_edges,
            visits,
            home_locations,
            work_locations,
            edu_locations,
        )

    print("Starting graph generation... Total: ", len(network_data.user_ids))

    # Use all available CPU cores (n_jobs=-1); verbose=1 prints progress
    parallel = Parallel(n_jobs=-1, verbose=1)
    # Run build_graph for each user in parallel, collect results in a list
    graphs = parallel(delayed(build_graph)(user_id) for user_id in network_data.user_ids)

    print("Graph generation finished.")

    return graphs


def build_user_pyg_graph(
    user_id: str,
    network_nodes: gpd.GeoDataFrame,
    network_edges: gpd.GeoDataFrame,
    location_visits: pl.DataFrame,
    home_locations: pl.DataFrame,
    work_locations: pl.DataFrame,
    edu_locations: pl.DataFrame,
):
    """
    Description: Build a single user's PyG Data object by combining the shared network
    topology with user-specific node annotations. Also applies two structural positional
    encodings (PE) to enrich node features with graph-structural information:

      - **Random Walk PE** (walk_length=20): Captures local connectivity by running random
        walks of length 20 from each node. Each node gets a 20-dimensional PE vector.
      - **Laplacian Eigenvector PE** (k=8): Captures global graph structure via the
        k smallest eigenvectors of the graph Laplacian. Each node gets an 8-dim PE vector.

    Input:
      - user_id (str): The user to build the graph for.
      - network_nodes (gpd.GeoDataFrame): Shared enriched subsector nodes (indexed by loc_id).
      - network_edges (gpd.GeoDataFrame): Shared spatial adjacency edges.
      - location_visits (pl.DataFrame): All users' visit records (filtered internally to user_id).
      - home_locations (pl.DataFrame): All users' home locations.
      - work_locations (pl.DataFrame): All users' work locations.
      - edu_locations (pl.DataFrame): All users' education locations.

    Output:
      - (pyg.data.Data): A PyG Data object with:
          - x: Node feature matrix [n_nodes, n_features] (census + POI + land-use + is_home + PEs)
          - edge_index: [2, n_edges] COO adjacency
          - edge_attr: [n_edges, 1] edge weights
          - y: [n_nodes, 1] binary is_visited labels
          - user_id: str attribute storing the user's ID
    """
    # Annotate the shared nodes with this user's home/work/edu/visited indicators
    indiv_nodes = add_user_cols(
        user_id,
        network_nodes,
        location_visits,
        home_locations,
        work_locations,
        edu_locations,
    )
    # Determine which columns to include as features (exclude metadata and label columns)
    node_feature_cols = [col for col in indiv_nodes.columns if col not in COLS_EXCLUDED_FROM_FEATURES]

    # Convert the annotated GeoDataFrame to a PyG Data object
    indiv_graph = c2g.gdf_to_pyg(
        indiv_nodes,
        network_edges,
        node_feature_cols=node_feature_cols,   # Feature columns for data.x
        node_label_cols=["is_visited"],         # Label column for data.y
        edge_feature_cols=["weight"],           # Edge feature for data.edge_attr
        keep_geom=False,                        # Don't store Shapely geometry in the graph
        device="cpu",
    )

    # Attach the user ID as a custom attribute for later lookup
    indiv_graph.user_id = user_id

    # Sanity check: the graph must have the same number of nodes as the input GeoDataFrame
    if indiv_graph.num_nodes != len(network_nodes):
        raise ValueError(f"Invalid graph number of nodes {len(indiv_nodes)} for {user_id=}.")

    # Define the sequence of PE transforms to apply
    transforms = T.Compose([
        T.AddRandomWalkPE(walk_length=20, attr_name=None),     # 20-dim random walk PE
        T.AddLaplacianEigenvectorPE(k=8, attr_name=None),      # 8-dim Laplacian eigenvector PE
    ])

    # Apply transforms: PEs are appended to data.x
    return transforms(indiv_graph)


def load_data(cfg: DataConfig, project_root: Path | None = None):
    """
    Description: Dispatch function that loads the correct dataset based on the
    configuration name. Supports "THATS" (Toronto) and "GenevaTPG" (Geneva).
    Returns the filtered data, network nodes, and network edges.

    Input:
      - cfg (DataConfig): Configuration object. The ``cfg.name`` field determines which
        dataset to load. Supported values: ``"THATS"`` and ``"GenevaTPG"``.
      - project_root (Path | None): Root directory of the project. Defaults to current dir.

    Output:
      - (tuple): ``(data, network_nodes, network_edges)`` where ``data`` is a NetworkData
        instance (subsector-filtered), and the two GeoDataFrames are the network graph.
    """
    if cfg.name == "THATS":
        # Cast to the Toronto-specific config type for type-safe attribute access
        cfg = cast(TorontoDataConfig, cfg)
        return _load_toronto_data(cfg, project_root)
    elif cfg.name == "GenevaTPG":
        # Cast to the Geneva-specific config type
        cfg = cast(TorontoDataConfig, cfg)
        return _load_geneva_data(cfg, project_root)

    raise ValueError(f"Unknown Dataset {cfg.name}")


def _load_toronto_data(cfg: TorontoDataConfig, project_root: Path | None = None):
    """
    Description: Load the Toronto THATS dataset and network graph, filtered to subsector level.

    Input:
      - cfg (TorontoDataConfig): Toronto-specific configuration with file paths.
      - project_root (Path | None): Project root directory.

    Output:
      - (tuple): ``(data, network_nodes, network_edges)`` for the Toronto dataset.
    """
    # Load and parse Toronto survey data, then filter to subsector locations only
    data = TorontoData.load(cfg, project_root).with_filter("subsector")
    # Load (or build and cache) the Toronto network graph
    network_nodes, network_edges = load_toronto_network_graph(data, cfg, project_root)

    return data, network_nodes, network_edges


def _load_geneva_data(cfg: GenevaDataConfig, project_root: Path | None = None):
    """
    Description: Load the Geneva TPG dataset and network graph, filtered to subsector level.

    Input:
      - cfg (GenevaDataConfig): Geneva-specific configuration with file paths.
      - project_root (Path | None): Project root directory.

    Output:
      - (tuple): ``(data, network_nodes, network_edges)`` for the Geneva dataset.
    """
    # Load and parse Geneva survey data, then filter to subsector locations only
    data = GenevaData.load(cfg, project_root).with_filter("subsector")
    # Load (or build and cache) the Geneva network graph (with island correction)
    network_nodes, network_edges = load_gva_network_graph(data, cfg, project_root)

    return data, network_nodes, network_edges
