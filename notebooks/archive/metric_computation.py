"""
Module: metric_computation.py

Description:
    A Marimo interactive notebook for computing and visualising structural metrics over
    household activity graphs derived from the LTDS (London Travel Demand Survey) dataset.

    Metrics computed include:
      - Node-level: graph order (number of unique locations).
      - Edge-level: size (number of trips), max/mean/sum duration/distance.
      - Graph-level: radius, diameter, edge density, assortativity, reciprocity,
        degree/betweenness/closeness/Katz centralisations (all weighted by trip duration).

    Results can be saved to a Parquet file and then:
      - Visualised as histograms over all households.
      - Aggregated spatially by UK postcode area/district/sector.
      - Aggregated spatially by municipality (Local Authority District).

Dependencies:
    - activitygraphs library (ActivityDataset, ActivityGraph, AllMetrics, WeightColumn)
    - Marimo, Polars, GeoPandas
"""

import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells

    import numpy as np
    import random

    # Set random seeds
    np.random.seed(42)
    random.seed(42)


@app.cell
def _(mo):
    """
    Description: Display the notebook title and section heading in the Marimo UI.

    Input:
      - mo (module): Marimo.

    Output:
      - (none): renders the heading in the notebook UI.
    """
    mo.md(
        r"""
    # Activtiy graph metric computation
    ## Setup and configuration
    """
    )
    return


@app.cell
def _():
    """
    Description: Import the core libraries for this notebook.

    Input:
      - (none)

    Output:
      - mo (module): Marimo for UI widgets and cell control.
      - pl (module): Polars for DataFrame manipulation.
    """
    import marimo as mo
    import polars as pl

    return mo, pl


@app.cell
def _(mo):
    """
    Description: Load the project configuration from the YAML file in the parent directory.

    Input:
      - mo (module): Marimo (for ``notebook_dir()``).

    Output:
      - cfg: the loaded configuration object with data paths and figure output directory.
    """
    from config import load_config

    cfg = load_config(mo.notebook_dir().parent)
    return (cfg,)


@app.cell
def _(mo):
    """
    Description: Render the data loading section heading in the notebook UI.

    Input:
      - mo (module): Marimo.

    Output:
      - (none): renders a Markdown heading in the notebook UI.
    """
    mo.md("""## Data loading""")
    return


@app.cell
def _(cfg):
    """
    Description: Load the processed LTDS ActivityDataset from disk and build an
    ActivityGraph container from it for metric computation.

    Input:
      - cfg: project configuration (provides dataset name and paths).

    Output:
      - dataset (ActivityDataset): the loaded dataset (also needed for spatial aggregation).
      - graph (ActivityGraph): the activity graph container for the full LTDS dataset.
    """
    from archive.exploration.dataprocessing import ActivityDataset
    from archive.exploration import ActivityGraph

    dataset = ActivityDataset.load(cfg.data.paths.act_dataset, cfg.data.name)
    graph = ActivityGraph.from_dataset(dataset)
    return dataset, graph


@app.cell
def _(mo):
    """
    Description: Create a run-button that forces a full re-computation of all graph
    metrics from scratch when clicked. Without clicking, cached results are loaded
    from the Parquet file.

    Input:
      - mo (module): Marimo.

    Output:
      - reprocess_button (mo.ui.run_button): the button that gates the reprocessing cell.
    """
    reprocess_button = mo.ui.run_button(label="Reprocess raw data")
    reprocess_button
    return (reprocess_button,)


@app.cell
def _(cfg, graph, pl, reprocess_button):
    """
    Description: Compute (or load from cache) all structural graph metrics over the
    household activity graphs.  If a cached Parquet file already exists and the
    reprocess button has not been pressed, results are read from disk to save time.
    Otherwise all metrics are recomputed from scratch using the ``AllMetrics`` bundle.

    Input:
      - cfg: project configuration (provides paths to the metrics directory).
      - graph (ActivityGraph): the activity graph container for the full LTDS dataset.
      - pl (module): Polars for reading/writing Parquet.
      - reprocess_button (mo.ui.run_button): if pressed, forces recomputation.

    Output:
      - Path (type): pathlib.Path, re-exported so downstream cells can build output paths.
      - metrics (AllMetrics): the metrics bundle (provides metric names and computation).
      - results (pl.DataFrame): one row per household with all metric columns.
    """
    from archive.exploration.metrics import AllMetrics, WeightColumn
    from pathlib import Path

    # Directory where metric Parquet files are stored; created if it does not yet exist
    metrics_dir = Path(cfg.data.paths.metrics)
    # Full path to the output Parquet file
    metrics_file = metrics_dir / "metrics.parquet"
    metrics_dir.mkdir(exist_ok=True, parents=True)  # create directory tree if missing

    # AllMetrics bundles all node-, edge- and graph-level metric implementations.
    # WeightColumn.DURATION means edges are weighted by trip duration in minutes.
    metrics = AllMetrics(weight_col=WeightColumn.DURATION)

    if metrics_file.exists() and not reprocess_button.value:
        # Load cached results from disk — avoids expensive O(n) graph metric computation
        results = pl.read_parquet(metrics_file)
    else:
        # Compute all metrics from scratch (verbose=True shows a tqdm progress bar)
        results = metrics.compute(graph, verbose=True)
        # Persist for future notebook runs
        results.write_parquet(metrics_file)
    return Path, metrics, results


@app.cell
def _(mo):
    """
    Description: Render the metrics analysis section heading in the notebook UI.

    Input:
      - mo (module): Marimo.

    Output:
      - (none): renders a Markdown heading in the notebook UI.
    """
    mo.md("""## Metrics analysis""")
    return


@app.cell
def _(results):
    """
    Description: Display descriptive statistics (count, mean, std, min/max, quartiles)
    for all computed graph metrics across the dataset. Provides a quick data overview.

    Input:
      - results (pl.DataFrame): the metrics DataFrame.

    Output:
      - (none): renders the describe table in the notebook UI.
    """
    results.describe()
    return


@app.cell
def _(Path, cfg, metrics, results):
    """
    Description: Plot histograms of the distribution of each metric across all household
    graphs and save the figure as a PNG file in the figures directory. This provides an
    at-a-glance overview of the dataset's structural properties.

    Input:
      - Path (type): pathlib.Path for building the output file path.
      - cfg: project configuration (provides figures output directory and dataset name).
      - metrics (AllMetrics): the metrics object (provides metric names for axis labels).
      - results (pl.DataFrame): the computed metrics DataFrame.

    Output:
      - (none): saves a PNG to ``<cfg.paths.figures>/<dataset_name>_metrics.png`` and
        renders the chart in the notebook UI.
    """
    from plotting import plot_metric_histograms

    chart = plot_metric_histograms(metrics, results)
    chart.save(Path(cfg.paths.figures) / f"{cfg.data.name}_metrics.png")
    chart
    return


@app.cell
def _(metrics, mo):
    """
    Description: Create two dropdown widgets that together control the postcode-spatial
    aggregation map: one selects which metric to plot (mean across households), and the
    other selects the spatial granularity (area / district / sector).

    Input:
      - metrics (AllMetrics): provides metric names for the dropdown options.
      - mo (module): Marimo.

    Output:
      - selected_metric (mo.ui.dropdown): the currently selected metric name string.
      - selected_postcode_split (mo.ui.dropdown): the currently selected postcode level
        (one of ``"area"``, ``"district"``, ``"sector"``).
    """
    selected_metric = mo.ui.dropdown(metrics.names(), value=metrics.names()[0], label="Plot mean")

    selected_postcode_split = mo.ui.dropdown(
        ["area", "district", "sector"],
        value="district",  # default to district level (medium granularity)
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
    """
    Description: Render a choropleth map showing the mean value of the selected metric
    across UK postcode zones at the chosen aggregation level. Updates reactively when
    the dropdowns change.

    Input:
      - geo_postcode_shapes (gpd.GeoDataFrame): postcode polygon boundaries.
      - mean_stats_by_postcode (pl.DataFrame): mean metric values per postcode zone.
      - selected_metric (mo.ui.dropdown): the chosen metric to visualise.
      - selected_postcode_split (mo.ui.dropdown): the chosen postcode granularity.

    Output:
      - geo_plot_mean_stat_by_municipality (callable): exported for the municipality map cell.
    """
    from plotting import (
        geo_plot_mean_stat_by_postcode,
        geo_plot_mean_stat_by_municipality,
    )

    geo_plot_mean_stat_by_postcode(
        mean_stats_by_postcode,
        geo_postcode_shapes,
        postcode_split=selected_postcode_split.value,
        stat=selected_metric.value,
    )
    return (geo_plot_mean_stat_by_municipality,)


@app.cell
def _(mean_stats_by_postcode):
    """
    Description: Display the mean-metric-by-postcode aggregation DataFrame for tabular
    inspection.

    Input:
      - mean_stats_by_postcode (pl.DataFrame): mean metrics per postcode zone.

    Output:
      - (none): renders the DataFrame in the notebook UI.
    """
    mean_stats_by_postcode
    return


@app.cell
def _(Path, selected_postcode_split):
    """
    Description:
        Marimo cell that loads the UK postcode boundary shapefile for the currently
        selected spatial aggregation level (area, district, or sector) and exposes
        both the GeoDataFrame and the geopandas module to downstream cells.

    Input:
      - Path: pathlib.Path class used to construct the shapefile path.
      - selected_postcode_split (mo.ui.dropdown): Marimo dropdown widget whose
            current value determines which spatial granularity to load.

    Output:
      - geo_postcode_shapes (gpd.GeoDataFrame): polygon boundaries for all UK
            postcodes at the chosen aggregation level.
      - gpd: the geopandas module, re-exported so dependent cells can use it.
    """
    import geopandas as gpd

    def read_geo_postcode_shapes(postcode_split: str):
        """
        Description:
            Loads the UK postcode boundary shapefile that corresponds to the chosen
            spatial aggregation level (area / district / sector).

        Input:
          - postcode_split (str): one of "area", "district", or "sector", indicating
                the geographic granularity at which to aggregate metric averages.

        Output:
          - (gpd.GeoDataFrame): a GeoDataFrame with postcode polygon geometries and a
                "name" column containing the postcode prefix (e.g. "SW" for area,
                "SW1A" for district).
        """
        # Base directory containing the three postcode boundary shapefiles
        path = Path("data/external/uk-postcodes/")
        gdf = None  # placeholder; overwritten by the match statement below

        match postcode_split:
            case "area":      # e.g. "SW", "EC" — broadest level
                path = path / "Areas.shp"
            case "district":  # e.g. "SW1A", "EC1V" — medium level
                path = path / "Districts.shp"
            case "sector":    # e.g. "SW1A 1" — finest level
                path = path / "Sectors.shp"

        return gpd.read_file(path)

    # Load the postcode shapes for the currently selected aggregation level
    geo_postcode_shapes = read_geo_postcode_shapes(selected_postcode_split.value)
    return geo_postcode_shapes, gpd


@app.cell
def _(dataset, pl, results, selected_postcode_split):
    """
    Description:
        Marimo cell that joins the metric results table with UK postcode boundaries
        and produces a choropleth GeoDataFrame showing the spatial distribution of
        model performance. Uses the currently selected postcode aggregation level to
        decide at which geographic granularity to aggregate metric averages.

    Input:
      - dataset: the loaded activity dataset providing the home-location column.
      - pl: the Polars library used for fast DataFrame operations.
      - results: a Polars DataFrame containing per-individual metric scores.
      - selected_postcode_split (mo.ui.dropdown): Marimo dropdown indicating the
            chosen spatial aggregation level ("area", "district", or "sector").

    Output:
      - postcode_results (gpd.GeoDataFrame): polygon GeoDataFrame keyed by postcode
            prefix with averaged metric values, ready for choropleth plotting.
    """
    def split_postcode_col(by: str, col_name="loc_home_loc_id"):
        """
        Description:
            Returns a Polars expression that extracts the postcode prefix at the requested
            spatial granularity from the raw home location ID string.

        Input:
          - by (str): "area", "district", or "sector" — the desired aggregation level.
          - col_name (str): the name of the column in the household DataFrame that holds
                the home location ID (a UK postcode string).  Defaults to "loc_home_loc_id".

        Output:
          - (pl.Expr): a Polars expression that extracts and aliases the postcode prefix
                at the appropriate granularity.
        """
        match by:
            case "area":
                # Extract 1-2 capital letters at the start, e.g. "SW" from "SW1A 1AA"
                return pl.col(col_name).str.extract(r"([A-Z]{1,2})\d+", 1).alias("area")
            case "district":
                # Take everything before the first space, e.g. "SW1A" from "SW1A 1AA"
                return pl.col(col_name).str.split(" ").list.first().alias("district")
            case "sector":
                # Drop the last 2 characters (the inward code unit), e.g. "SW1A 1" from "SW1A 1AA"
                return pl.col(col_name).str.head(-2).alias("sector")

    # Extract the postcode prefix for each household and join with the metric results
    _hh_person_df = dataset.hh_person_df.select("hh_id", split_postcode_col(selected_postcode_split.value))

    # Aggregate mean metric values per postcode; only include postcodes with > 20 households
    # to avoid noise from very sparsely populated areas.
    mean_stats_by_postcode = (
        results
        .join(_hh_person_df, on="hh_id")
        .drop("hh_id")
        .group_by(selected_postcode_split.value)
        .agg(pl.len().alias("n_samples"), pl.all().mean())  # mean of all metric columns
        .filter(pl.col("n_samples") > 20)                   # minimum sample-size filter
    )

    mean_stats_by_postcode
    return (mean_stats_by_postcode,)


@app.cell
def _(mo):
    """
    Description: Render the borough-level analysis section heading.

    Input:
      - mo (module): Marimo.

    Output:
      - (none): renders a Markdown heading in the notebook UI.
    """
    mo.md("""## Analysis by Boroughs""")
    return


@app.cell
def _(metrics, mo):
    """
    Description: Create a dropdown widget to choose which metric to visualise on the
    borough-level choropleth map.

    Input:
      - metrics (AllMetrics): provides metric names.
      - mo (module): Marimo.

    Output:
      - selected_muni_metric (mo.ui.dropdown): the currently selected metric.
    """
    selected_muni_metric = mo.ui.dropdown(metrics.names(), value=metrics.names()[0], label="Plot mean")

    mo.hstack([selected_muni_metric, mo.md("by municipality")], justify="start")
    return (selected_muni_metric,)


@app.cell
def _(
    geo_municipality_shapes,
    geo_plot_mean_stat_by_municipality,
    mean_stats_by_municipality,
    selected_muni_metric,
):
    """
    Description: Render a choropleth map showing the mean value of the selected metric
    aggregated by London borough. Updates reactively when the dropdown changes.

    Input:
      - geo_municipality_shapes (gpd.GeoDataFrame): Local Authority District polygon boundaries.
      - geo_plot_mean_stat_by_municipality (callable): the borough-level choropleth function.
      - mean_stats_by_municipality (pl.DataFrame): mean metrics per borough.
      - selected_muni_metric (mo.ui.dropdown): the metric to visualise.

    Output:
      - (none): renders the choropleth map in the notebook UI.
    """
    geo_plot_mean_stat_by_municipality(
        mean_stats_by_municipality,
        geo_municipality_shapes,
        selected_muni_metric.value,
    )
    return


@app.cell
def _(dataset, pl, results):
    """
    Description: Aggregate mean graph-metric values by London borough (Local Authority
    District). Each household is assigned to the borough of its home location via a join
    with the location reference table. Boroughs with 20 or fewer households are excluded
    to prevent unreliable mean estimates from small sample sizes.

    Input:
      - dataset (ActivityDataset): provides ``hh_person_df`` (home postcodes) and
        ``location_df`` (postcode -> municipality mappings).
      - pl (module): Polars for joining and aggregation.
      - results (pl.DataFrame): the graph metrics DataFrame.

    Output:
      - mean_stats_by_municipality (pl.DataFrame): one row per borough with columns for
        ``municipality_id`` (ONS code), ``municipality_name``, ``n_samples``,
        and the mean of every metric column.
    """
    _hh_person_df = dataset.hh_person_df.select("hh_id", "loc_home_loc_id")

    _location_df = dataset.location_df  # contains municipality_id and municipality_name

    mean_stats_by_municipality = (
        results
        .join(_hh_person_df, on="hh_id")
        .join(_location_df, left_on="loc_home_loc_id", right_on="loc_id", how="left")
        .drop("hh_id", "loc_home_loc_id")
        .group_by(["municipality_id", "municipality_name"])
        .agg(pl.len().alias("n_samples"), pl.all().mean())  # mean of all metric columns
        .filter(pl.col("n_samples") > 20)  # remove boroughs with insufficient data
    )

    mean_stats_by_municipality
    return (mean_stats_by_municipality,)


@app.cell
def _(gpd):
    """
    Description: Load the UK Local Authority District (LAD) boundary shapefile for
    the May 2024 release. Only the ONS district code (``LAD24CD``) and geometry columns
    are loaded to keep memory usage low.

    Input:
      - gpd (module): GeoPandas.

    Output:
      - geo_municipality_shapes (gpd.GeoDataFrame): LAD polygon boundaries with
        ``LAD24CD`` (the ONS code that matches ``municipality_id``) and ``geometry``.
    """
    geo_municipality_shapes = gpd.read_file(
        "data/external/uk-local-authorities/LAD_MAY_2024_UK_BFE.shp",
        columns=["LAD24CD", "geometry"],  # only load what is needed
    )
    return (geo_municipality_shapes,)


if __name__ == "__main__":
    app.run()
