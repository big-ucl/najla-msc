import marimo

__generated_with = "0.18.1"
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
    ## Project setup and data loading
    """)
    return


@app.cell
def _():
    from activitygraphs.data.geneva import load_files, build_geneva_data

    gva_inputs = load_files(cfg.data, project_root)
    gva_data = build_geneva_data(gva_inputs)
    return (gva_data,)


@app.cell
def _(gva_data):
    gva_data.locations_df
    return


@app.cell
def _(gva_data):
    gva_data.user_journeys_df
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Network building
    """)
    return


@app.cell
def _(gva_data):
    from activitygraphs.data.gtfs import build_pt_network_edges

    pt_edge_df, transfer_edge_df = build_pt_network_edges(gva_data.locations_df, gva_data.gtfs)
    return pt_edge_df, transfer_edge_df


@app.cell
def _(pt_edge_df):
    pt_edge_df
    return


@app.cell
def _(transfer_edge_df):
    transfer_edge_df
    return


@app.cell
def _():
    mo.md(r"""
    ### Questions and To-Dos

    **Question: Multi-edges or duplicate nodes?**

    Do we do multi-edges between `loc_id`s with `route_id` as the edge type, or do we create one node per (`loc_id`, `route_id`) pair. If feasible, multi-nodes would get inter-line transfers for almost free, as well as cleaner interface in PyG.

    Current data:
    - 1736 nodes, 1141 of which are PT nodes
    - 6639 unique (`loc_id`, `route_id`) pairs
    """)
    return


if __name__ == "__main__":
    app.run()
