import marimo

__generated_with = "0.19.6"
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
    return (gva_data,)


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


@app.cell
def _(gva_network):
    gva_network.location_types
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
def _(user_journeys_df):
    user_journeys_df["dep_purpose"].value_counts(sort=True).plot.bar(x="dep_purpose:N", y="count")
    return


@app.cell
def _(gva_data):
    user_journeys_df = gva_data.user_journeys_df
    return (user_journeys_df,)


@app.cell
def _():
    from activitygraphs.geometric import add_labels_to_pyg
    from tqdm import tqdm
    return add_labels_to_pyg, tqdm


@app.cell
def _(add_labels_to_pyg, d, user_journeys_df):
    next(iter(add_labels_to_pyg(d, user_journeys_df, True)))
    return


@app.cell
def _(add_labels_to_pyg, d, tqdm, user_journeys_df):

    mo.stop(True)

    gs = []
    for g in tqdm(add_labels_to_pyg(d, user_journeys_df, True), total=len(user_journeys_df["user_id"].unique())):
        gs.append(g)
    return


@app.cell
def _(d):
    from torch_geometric.transforms import AddMetaPaths

    metapaths = [
        [
            ("planar", "contains", "public_transport"),
            ("public_transport", "transfer", "public_transport"),
            ("public_transport", "pt", "public_transport"),
            ("public_transport", "transfer", "public_transport"),
            ("public_transport", "is_contained_by", "planar"),
        ]
    ]

    AddMetaPaths(metapaths)(d)
    return


if __name__ == "__main__":
    app.run()
