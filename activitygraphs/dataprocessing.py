from pathlib import Path

from activitygraphs.base import CRS
from activitygraphs.data.overture import Overture
from activitygraphs.data.statistics import add_statistics

import city2graph as c2g
import pandas as pd
import geopandas as gpd


def build_network_graph(
    locations: gpd.GeoDataFrame,
    overture: Overture,
    stats_path: Path,
):
    utm_crs = locations.estimate_utm_crs()
    locations = locations.to_crs(utm_crs)

    network_locations = add_statistics(locations, stats_path)
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
    edges: gpd.GeoDataFrame = pd.concat([edges, island_edges]).to_crs(CRS)

    return nodes, edges
