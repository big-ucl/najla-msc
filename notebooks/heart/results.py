import marimo

__generated_with = "0.21.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import pickle

    import torch
    import torch_geometric as pyg

    import pandas as pd
    import geopandas as gpd
    import polars as pl

    import city2graph as c2g
    import pyproj

    from activitygraphs.config import load_config
    from pathlib import Path

    project_root = Path(mo.notebook_dir().parent.parent)
    cfg = load_config(project_root)


@app.cell
def _():
    reports_path = project_root / cfg.paths.reports / "data"
    figures_path = project_root / cfg.paths.figures
    return figures_path, reports_path


@app.cell
def _(reports_path):
    results = pl.read_parquet(reports_path / "geneva-results.parquet")
    results
    return (results,)


@app.cell
def _():
    import altair as alt

    return (alt,)


@app.cell
def _(alt, figures_path, results):
    _results = results.filter(pl.col("epoch") > 5).unpivot(
        on=["bce_weight"],
        index=["name", "epoch"],
        value_name="BCE",
        variable_name="Dataset",
    )

    fig = (
        alt
        .Chart(_results)
        .mark_line(point=True)
        .encode(
            x="epoch",
            y="BCE",
            color="name",
            facet="Dataset",
            tooltip=["name", "epoch", "BCE"],
        )
    )

    fig.save(figures_path / "results.png")

    fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
