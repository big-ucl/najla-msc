import marimo

__generated_with = "0.19.2"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    import geopandas as gpd

    import marimo as mo
    import polars as pl
    import polars.selectors as cs
    from pathlib import Path

    from activitygraphs.config import load_config

    project_root = Path(mo.notebook_dir().parent)
    cfg = load_config(project_root)


@app.cell
def _():
    from activitygraphs.data.geneva import GenevaData

    gva_data = GenevaData.load(cfg.data, project_root)
    gva_data
    return


@app.cell
def _():
    from activitygraphs.network import Network

    network_name = "routes"
    gva_network = Network.load(cfg.data, project_root, network_name)
    gva_network
    return (gva_network,)


@app.cell
def _(gva_network):
    gva_network
    return


@app.cell
def _(gva_network):
    gva_network
    return


@app.cell
def _(gva_network):
    import torch

    from torch_geometric.data import HeteroData
    from activitygraphs.geometric import EnumEncoder, _process_layer_nodes, _process_layer_edges

    layer_name = "subsector"

    data = HeteroData()
    layer = gva_network[layer_name]
    layer_locations_gdf = gva_network.get_layer_locations(layer_name)

    loc_type_encoder = EnumEncoder(gva_network.location_types, 3)
    encoders = {"type": loc_type_encoder}
    return data, layer, layer_locations_gdf, loc_type_encoder


@app.cell
def _(data, layer, layer_locations_gdf, loc_type_encoder):
    node_mapping, node_x = _process_layer_nodes(layer_locations_gdf, {"type": loc_type_encoder})
    edge_list, edge_attrs = _process_layer_edges(layer, node_mapping)

    data[layer.type].x = node_x
    data[layer.type].edge_list = edge_list
    data[layer.type].edge_attrs = edge_attrs
    data[layer.type].num_nodes = len(node_mapping)
    return


@app.cell
def _(data):
    data
    return


if __name__ == "__main__":
    app.run()
