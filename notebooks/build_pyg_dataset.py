import marimo

__generated_with = "0.19.4"
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
    from activitygraphs.geometric import network_to_pyg

    d = network_to_pyg(gva_network)
    d
    return (d,)


@app.cell
def _(d):
    from torch_geometric.utils import to_networkx

    G = to_networkx(d)
    return


if __name__ == "__main__":
    app.run()
