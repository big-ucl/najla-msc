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

    pt_edge_df = build_pt_network_edges(gva_data.locations_df, gva_data.gtfs)
    pt_edge_df
    return


if __name__ == "__main__":
    app.run()
