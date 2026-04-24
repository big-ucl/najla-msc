import pickle
from pathlib import Path

import city2graph as c2g
import geopandas as gpd
import pandas as pd
import polars as pl
import torch_geometric.transforms as T
from joblib import Parallel, delayed

from activitygraphs.base import CRS
from activitygraphs.config import GenevaDataConfig
from activitygraphs.data.geneva import GenevaData
from activitygraphs.data.overture import Overture
from activitygraphs.data.statistics import add_population_job_statistics
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


def load_gva_network_graph(
    gva_data: GenevaData,
    cfg: GenevaDataConfig,
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

    overture = Overture.load(gva_data.locations_gdf, cfg.inputs.overture, root)
    stats_path = root / cfg.inputs.statistics

    subsector_gva_data = gva_data.with_filter("subsector")
    network_nodes, network_edges = build_gva_network_graph(subsector_gva_data.locations_gdf, overture, stats_path)

    network_path.mkdir(parents=True, exist_ok=True)
    network_nodes.to_parquet(nodes_path)
    network_edges.to_parquet(edges_path)

    return network_nodes, network_edges


def build_gva_network_graph(
    locations: gpd.GeoDataFrame,
    overture: Overture,
    stats_path: Path,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    utm_crs = locations.estimate_utm_crs()
    locations = locations.to_crs(utm_crs)

    network_locations = add_population_job_statistics(locations, stats_path)
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


def load_pyg_dataset(
    network_data: NetworkData,
    network_nodes: gpd.GeoDataFrame,
    network_edges: gpd.GeoDataFrame,
    cfg: GenevaDataConfig,
    project_root: Path | None = None,
    name: str = "PyGDataset",
):
    dataset_path = get_project_root(project_root) / cfg.paths.processed / f"{name}.pickle"

    if dataset_path.exists():
        with open(dataset_path, "rb") as f:
            return pickle.load(f)

    graphs = convert_to_pyg_dataset(network_data, network_nodes, network_edges)

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dataset_path, "wb") as f:
        pickle.dump(graphs, f)

    return graphs


def convert_to_pyg_dataset(
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
