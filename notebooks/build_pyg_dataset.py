import marimo

__generated_with = "0.19.8"
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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Building the PyG graph

    Create the `pyg.HeteroData` graph from `GenevaData` user information and `Network` network information
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 1

    Load the user and network information generated in previous notebooks.
    """)
    return


@app.cell
def _():
    from activitygraphs.data.geneva import GenevaData

    gva_data = GenevaData.load(cfg.data, project_root)
    gva_data
    return (gva_data,)


@app.cell
def _():
    from activitygraphs.network import Network

    network_name = "routes"
    gva_network = Network.load(cfg.data, project_root, network_name)
    gva_network
    return gva_network, network_name


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 2

    Create the base `pyg.HeteroData` object from the network, without user information
    """)
    return


@app.cell
def _(gva_network):
    from activitygraphs.geometric import network_to_pyg

    base_data = network_to_pyg(gva_network)
    base_data
    return (base_data,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 3

    Using `ActivityGraphBuilder`, label the base network with user information to create one `pyg.HeteroData` per user, then save it to a `pyg.Dataset`.
    """)
    return


@app.cell
def _(base_data, gva_data, network_name):
    from activitygraphs.geometric import ActivityGraphBuilder, ActivityDataset

    try:
        dataset = ActivityDataset.from_files(cfg.data, project_root, name=network_name)
    except FileNotFoundError:
        builder = ActivityGraphBuilder(base_data, gva_data.user_journeys_df, separate_na_source_sink=True)
        dataset = ActivityDataset.from_builder(builder, cfg.data, project_root, name=network_name)

    dataset
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Analysing the generated `pyg.HeteroData` graph
    """)
    return


@app.cell
def _(base_data):
    from torch_geometric.utils import to_networkx

    G = to_networkx(base_data, to_multi=True)
    G
    return (G,)


@app.cell
def _(G):
    import networkx as nx

    # Exclude isolated nodes and NA source and sink, all other nodes are strongly connected
    main_subgraph = max(nx.strongly_connected_components(G), key=lambda x: len(x))
    diameter = nx.diameter(G.subgraph(main_subgraph))

    mo.md(f"Main graph diameter: {diameter}")
    return


@app.cell
def _(gva_data):
    num_unique_journeys_per_user = (
        gva_data.user_journeys_df
        .with_columns(pl.col("journey_id").str.replace("_r", ""))
        .group_by("user_id")
        .agg(num_journeys=pl.col("journey_id").unique().len())["num_journeys"]
        .value_counts()
        .sort(by="num_journeys")
    )

    num_unique_journeys_per_user.plot.bar(x="num_journeys:N", y="count").properties(
        title="Freq. of number of unique journeys per user, not incuding return trips"
    )
    return


@app.cell
def _(gva_data):
    gva_data.user_journeys_df["dep_purpose"].value_counts(sort=True).plot.bar(x="dep_purpose:N", y="count")
    return


if __name__ == "__main__":
    app.run()
