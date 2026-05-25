import pickle
from pathlib import Path
from typing import Callable

import city2graph as c2g
import geopandas as gpd
import pandas as pd
import polars as pl
import torch
import torch_geometric.transforms as T
from joblib import Parallel, delayed

from activitygraphs.base import CRS
from activitygraphs.config import DataConfig, GenevaStatsInputs, StatsInputs, TorontoStatsInputs
from activitygraphs.data.geneva import GenevaData
from activitygraphs.data.overture import Overture
from activitygraphs.data.statistics import (
    add_geneva_population_job_statistics,
    add_toronto_population_job_statistics,
)
from activitygraphs.data.toronto import TorontoData
from activitygraphs.network import NetworkData
from activitygraphs.utils import get_project_root

COLS_EXCLUDED_FROM_FEATURES = [
    "loc_name",
    "type",
    "lon",
    "lat",
    "is_work",
    "is_edu",
    "is_visited",
    "purpose",
    "geometry",
    "original_geometry",
]

# =========================================
# A. Network graph
# =========================================

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
    root = get_project_root(project_root)
    network_path = root / cfg.paths.processed / name
    nodes_path = network_path / "nodes.parquet"
    edges_path = network_path / "edges.parquet"

    if nodes_path.exists() and edges_path.exists():
        network_nodes = gpd.read_parquet(nodes_path)
        network_edges = gpd.read_parquet(edges_path)

        return network_nodes, network_edges

    overture = Overture.load(data.locations_gdf, cfg.inputs.overture, root)

    subsector_data = data.with_filter("subsector")
    network_nodes, network_edges = build_network_graph(subsector_data.locations_gdf, overture, cfg.inputs.statistics)

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
    return load_network_graph(gva_data, cfg, build_gva_network_graph, project_root, name)


def load_toronto_network_graph(
    toronto_data: TorontoData,
    cfg: DataConfig,
    project_root: Path | None = None,
    name: str = "NetworkGraph",
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    return load_network_graph(toronto_data, cfg, build_toronto_network_graph, project_root, name)


def build_gva_network_graph(
    locations: gpd.GeoDataFrame,
    overture: Overture,
    stats_cfg: StatsInputs,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    utm_crs = locations.estimate_utm_crs()
    locations = locations.to_crs(utm_crs)

    stats_cfg: GenevaStatsInputs

    network_locations = add_geneva_population_job_statistics(locations, stats_cfg)
    network_locations = overture.add_poi_counts(network_locations)
    network_locations = overture.add_land_uses(network_locations)
    network_locations = network_locations.set_index("loc_id")

    nodes, edges = c2g.contiguity_graph(network_locations, set_point_nodes=True)

    # Connect disconnected subsectors to main graph
    island_loc_ids = [
        "subsector-174",
        "subsector-141",
        "subsector-10",
        "subsector-325",
    ]
    cross_lake_loc_ids = ["subsector-243", "subsector-261", "subsector-54"]
    island_nodes = nodes[nodes.index.isin(island_loc_ids)]
    mainland_nodes = nodes[~nodes.index.isin(island_loc_ids) & ~nodes.index.isin(cross_lake_loc_ids)]

    _, island_edges = c2g.knn_graph(island_nodes, k=3, target_gdf=mainland_nodes)
    island_edges = island_edges.reset_index()
    island_edges["source"] = island_edges["source"].str[1]
    island_edges["target"] = island_edges["target"].str[1]
    island_edges = island_edges.set_index(["source", "target"])

    nodes = nodes.to_crs(CRS)
    edges: gpd.GeoDataFrame = pd.concat([edges, island_edges])
    edges = edges.to_crs(CRS)

    return nodes, edges


def build_toronto_network_graph(
    locations: gpd.GeoDataFrame,
    overture: Overture,
    stats_cfg: StatsInputs,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    utm_crs = locations.estimate_utm_crs()
    locations = locations.to_crs(utm_crs)

    stats_cfg: TorontoStatsInputs

    network_locations = add_toronto_population_job_statistics(locations, stats_cfg)
    network_locations = overture.add_poi_counts(network_locations)
    network_locations = overture.add_land_uses(network_locations)
    network_locations = network_locations.set_index("loc_id")

    nodes, edges = c2g.contiguity_graph(network_locations, set_point_nodes=True)

    nodes = nodes.to_crs(CRS)
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
    network_nodes = network_nodes.copy().reset_index()

    user_visits = location_visits.filter(user_id=user_id)["loc_id"].to_list()
    home_location = home_locations.filter(user_id=user_id)["loc_id"].to_list()
    work_location = work_locations.filter(user_id=user_id)["loc_id"].to_list()
    edu_location = edu_locations.filter(user_id=user_id)["loc_id"].to_list()

    network_nodes["is_home"] = network_nodes["loc_id"].isin(home_location).astype(int)
    network_nodes["is_work"] = network_nodes["loc_id"].isin(work_location).astype(int)
    network_nodes["is_edu"] = network_nodes["loc_id"].isin(edu_location).astype(int)
    network_nodes["is_visited"] = network_nodes["loc_id"].isin(user_visits).astype(int)

    visit_purposes = (
        location_visits
        .filter(user_id=user_id)
        .select("loc_id", pl.col("purpose"))
        .group_by("loc_id")
        .agg(pl.col("purpose").str.join(", "))
        .to_pandas()
    )
    network_nodes = network_nodes.merge(visit_purposes, on="loc_id", how="left")
    network_nodes = network_nodes.set_index("loc_id").sort_index()

    return network_nodes


def create_spatial_demographics(data: NetworkData) -> tuple[torch.Tensor, torch.Tensor]:
    locations = data.locations_df.select("loc_id").with_row_index("loc_order")
    all_users_and_locs = data.user_ids.to_frame().join(locations, how="cross").lazy()

    n_users = len(data.user_ids)
    n_locs = len(locations)

    # Build spatial features (i.e. will be appended to PyG data.x)
    feature_df = (
        all_users_and_locs
        .pipe(add_indicator_column, data.home_locations, "is_home")
        .pipe(add_indicator_column, data.work_locations, "is_work")
        .pipe(add_indicator_column, data.edu_locations, "is_edu")
        .sort("user_id", "loc_order")
        .drop("loc_order")
        .collect()
    )

    feature_cols = ["is_home"]
    spatial_features = torch.tensor(feature_df.select(feature_cols).to_numpy(), dtype=torch.float32).reshape(
        n_users, n_locs, len(feature_cols)
    )

    # Build spatial demographics labels (i.e. will form PyG data.y)
    label_df = (
        all_users_and_locs
        .pipe(add_indicator_column, data.location_visits, "is_visited")
        .sort("user_id", "loc_order")
        .drop("loc_order")
        .collect()
    )

    label_cols = ["is_visited"]
    spatial_labels = torch.tensor(label_df.select(label_cols).to_numpy(), dtype=torch.float32).reshape(
        n_users, n_locs, len(label_cols)
    )

    return spatial_features, spatial_labels


def add_indicator_column(feature_df: pl.LazyFrame, indicator_df: pl.DataFrame, col_name: str):
    true_indicators = indicator_df.select("user_id", "loc_id", pl.lit(True).alias(col_name)).unique().lazy()

    return feature_df.join(true_indicators, on=["user_id", "loc_id"], how="left").with_columns(
        pl.col(col_name).fill_null(False)
    )


# =========================================
# C. Individual demographics
# =========================================


def create_individual_demographics(data: NetworkData) -> torch.Tensor:
    pass


# =========================================
# PyG / Torch creation
# =========================================


def load_pyg_graphs(
    network_data: NetworkData,
    network_nodes: gpd.GeoDataFrame,
    network_edges: gpd.GeoDataFrame,
    cfg: DataConfig,
    project_root: Path | None = None,
    name: str = "Graphs",
):
    dataset_path = get_project_root(project_root) / cfg.paths.pyg_datasets / f"{name}.pickle"

    if dataset_path.exists():
        with open(dataset_path, "rb") as f:
            return pickle.load(f)

    graphs = convert_to_pyg_graphs(network_data, network_nodes, network_edges)

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dataset_path, "wb") as f:
        pickle.dump(graphs, f)

    return graphs


def convert_to_pyg_graphs(
    network_data: NetworkData,
    network_nodes: gpd.GeoDataFrame,
    network_edges: gpd.GeoDataFrame,
):
    visits = network_data.location_visits
    home_locations = network_data.home_locations
    work_locations = network_data.work_locations
    edu_locations = network_data.edu_locations

    def build_graph(user_id: str):
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

    parallel = Parallel(n_jobs=-1, verbose=1)
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
    indiv_nodes = add_user_cols(
        user_id,
        network_nodes,
        location_visits,
        home_locations,
        work_locations,
        edu_locations,
    )
    node_feature_cols = [col for col in indiv_nodes.columns if col not in COLS_EXCLUDED_FROM_FEATURES]

    indiv_graph = c2g.gdf_to_pyg(
        indiv_nodes,
        network_edges,
        node_feature_cols=node_feature_cols,
        node_label_cols=["is_visited"],
        edge_feature_cols=["weight"],
        keep_geom=False,
        device="cpu",
    )

    indiv_graph.user_id = user_id

    if indiv_graph.num_nodes != len(network_nodes):
        raise ValueError(f"Invalid graph number of nodes {len(indiv_nodes)} for {user_id=}.")

    transforms = T.Compose([
        T.AddRandomWalkPE(walk_length=20, attr_name=None),
        T.AddLaplacianEigenvectorPE(k=8, attr_name=None),
    ])

    return transforms(indiv_graph)
