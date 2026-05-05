import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo

    import torch
    import torch_geometric as pyg

    from activitygraphs.config import load_config
    from pathlib import Path

    project_root = Path(mo.notebook_dir().parent.parent)
    cfg = load_config(project_root, data="toronto")


@app.cell
def _():
    CRS = "EPSG:4326"
    return


@app.cell
def _():
    import geopandas as gpd
    import polars as pl

    return


@app.cell
def _():
    from activitygraphs.data.toronto import TorontoData

    return (TorontoData,)


@app.cell
def _(TorontoData):
    toronto_data = TorontoData.load(cfg.data, project_root)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
