import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    # Activtiy graph metric computation
    ## Setup and configuration
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    import random

    from config import load_config
    return load_config, mo, np, pl, random


@app.cell
def _(load_config, mo, np, random):
    # Set random seeds
    np.random.seed(42)
    random.seed(42)

    # Load project config
    cfg = load_config(mo.notebook_dir().parent)
    return (cfg,)


@app.cell
def _(mo):
    mo.md("""## Data loading""")
    return


@app.cell
def _(cfg):
    from dataprocessing import ActivityDataset
    from graphs import ActivityGraph

    dataset = ActivityDataset.load(cfg.data.paths.act_dataset, cfg.data.name)
    graph = ActivityGraph.from_dataset(dataset)
    return dataset, graph


@app.cell
def _(mo):
    reprocess_button = mo.ui.run_button(label="Reprocess raw data")
    reprocess_button
    return (reprocess_button,)


@app.cell
def _(cfg, graph, pl, reprocess_button):
    from metrics import AllMetrics, WeightColumn
    from pathlib import Path

    metrics_dir = Path(cfg.data.paths.metrics)
    metrics_file = metrics_dir / "metrics.parquet"
    metrics_dir.mkdir(exist_ok=True, parents=True)

    metrics = AllMetrics(weight_col=WeightColumn.DURATION)

    if metrics_file.exists() and not reprocess_button.value:
        results = pl.read_parquet(metrics_file)
    else:
        results = metrics.compute(graph, verbose=True)
        results.write_parquet(metrics_file)
    return Path, metrics, results


@app.cell
def _(mo):
    mo.md("""## Metrics analysis""")
    return


@app.cell
def _(results):
    results.describe()
    return


@app.cell
def _(Path, cfg, metrics, results):
    from plotting import plot_metric_histograms

    chart = plot_metric_histograms(metrics, results)
    chart.save(Path(cfg.paths.figures) / f"{cfg.data.name}_metrics.png")
    chart
    return


@app.cell
def _(metrics, mo):
    selected_metric = mo.ui.dropdown(
        metrics.names(), value=metrics.names()[0], label="Plot mean"
    )

    selected_postcode_split = mo.ui.dropdown(
        ["area", "district", "sector"],
        value="district",
        label="by uk postcode ",
    )

    mo.hstack([selected_metric, selected_postcode_split], justify="start")
    return selected_metric, selected_postcode_split


@app.cell
def _(
    geo_postcode_shapes,
    mean_stats_by_postcode,
    selected_metric,
    selected_postcode_split,
):
    from plotting import geo_plot_mean_stat

    geo_plot_mean_stat(
        mean_stats_by_postcode,
        geo_postcode_shapes,
        postcode_split=selected_postcode_split.value,
        stat=selected_metric.value,
    )
    return


@app.cell
def _(mean_stats_by_postcode):
    mean_stats_by_postcode
    return


@app.cell
def _(Path, selected_postcode_split):
    import geopandas as gpd


    def read_geo_postcode_shapes(postcode_split: str):
        path = Path("data/external/uk-postcodes/")
        gdf = None

        match postcode_split:
            case "area":
                path = path / "Areas.shp"
            case "district":
                path = path / "Districts.shp"
            case "sector":
                path = path / "Sectors.shp"

        return gpd.read_file(path)


    geo_postcode_shapes = read_geo_postcode_shapes(selected_postcode_split.value)
    return (geo_postcode_shapes,)


@app.cell
def _(dataset, pl, results, selected_postcode_split):
    def split_postcode_col(by: str, col_name="loc_home_loc_id"):
        match by:
            case "area":
                return (
                    pl.col(col_name)
                    .str.extract(r"([A-Z]{1,2})\d+", 1)
                    .alias("area")
                )
            case "district":
                return (
                    pl.col(col_name).str.split(" ").list.first().alias("district")
                )
            case "sector":
                return pl.col(col_name).str.head(-2).alias("sector")


    _hh_person_df = dataset.hh_person_df.select(
        "hh_id", split_postcode_col(selected_postcode_split.value)
    )

    mean_stats_by_postcode = (
        results.join(_hh_person_df, on="hh_id")
        .drop("hh_id")
        .group_by(selected_postcode_split.value)
        .agg(pl.len().alias("n_samples"), pl.all().mean())
        .filter(pl.col("n_samples") > 20)
    )

    mean_stats_by_postcode
    return (mean_stats_by_postcode,)


@app.cell
def _(pl):
    postcodes = pl.read_parquet("data/external/ukpostcodeandlad.parquet")
    postcodes
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
