"""
Module: notebooks/thats/processing.py

Description:
    A Marimo interactive notebook for loading, processing and analysing the Toronto THATS
    dataset.  It:
      1. Loads TorontoData and builds the city network graph (network_nodes, network_edges).
      2. Converts the data to the PyTorch Geometric ActivityDataset format.
      3. Provides interactive visualisations of the network and dataset properties.
      4. Performs exploratory analysis of trip modes (predicted vs manually-entered),
         activity purposes, and individual observation coverage.
      5. Contains a compare_trip_to_activity function that links trip summaries to hourly
         activity records to measure how well the app's purpose labels agree with the
         activity diary.

Dependencies:
    - activitygraphs library (TorontoData, load_toronto_network_graph, convert_to_torch,
      ActivityDataset, Mode, build_toronto_activities)
    - Marimo, Polars, GeoPandas, Altair
"""

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo

    from activitygraphs.config import load_config
    from pathlib import Path

    project_root = Path(mo.notebook_dir().parent.parent)
    cfg = load_config(project_root, data="toronto")


@app.cell
def _():
    """
    Description:
        Defines the coordinate reference system constant used for all geographic
        operations in this notebook.  Not exported (returned as None) because it is
        used as a module-level constant in several display cells via closure.

    Output:
      - (nothing exported — CRS is used locally within map display cells)
    """
    CRS = "EPSG:4326"  # WGS-84 lat/lon CRS used for map projections and contextily
    return


@app.cell
def _():
    """
    Description:
        Imports the key data-manipulation libraries for this notebook.
        GeoPandas is imported for the GeoDataFrame operations in load_toronto_network_graph;
        Polars is the main tabular data library used throughout the analysis cells.

    Output:
      - pl (module): polars, exported for use in downstream cells.
      - (gpd is also imported but not returned — it is used inside activitygraphs functions)
    """
    import geopandas as gpd
    import polars as pl

    return (pl,)


@app.cell
def _():
    """
    Description:
        Imports the Toronto-specific data loading and processing classes.
          - TorontoData: loads and bundles all four THATS tables (trips, activities,
            demographics, network) into a single object.
          - load_toronto_network_graph: builds the city network GeoDataFrames.
          - convert_to_torch: converts the geographic data to PyTorch tensors.
          - ActivityDataset: the PyG InMemoryDataset wrapper for the Toronto data.

    Output:
      - ActivityDataset (class): PyG dataset wrapper for ML pipeline use.
      - TorontoData (class): THATS data container / loader.
      - convert_to_torch (function): geographic-to-tensor converter.
      - load_toronto_network_graph (function): network GeoDataFrame builder.
    """
    from activitygraphs.data.toronto import TorontoData
    from activitygraphs.dataprocessing import load_toronto_network_graph, convert_to_torch

    from activitygraphs.ml.dataset import ActivityDataset

    return (
        ActivityDataset,
        TorontoData,
        convert_to_torch,
        load_toronto_network_graph,
    )


@app.cell(hide_code=True)
def _():
    """
    Description: Render the data processing section heading.

    Output:
      - (none): renders a Markdown heading in the notebook UI.
    """
    mo.md(r"""
    ## Data processing
    """)
    return


@app.cell
def _(
    ActivityDataset,
    TorontoData,
    convert_to_torch,
    load_toronto_network_graph,
):
    """
    Description: Load the full Toronto dataset, build the geographic network GeoDataFrames,
    convert everything to PyTorch tensors, and instantiate the PyG ActivityDataset (which
    is cached on disk for future runs).

    Input:
      - ActivityDataset (class): PyG InMemoryDataset wrapper for the Toronto data.
      - TorontoData (class): loads and bundles all four THATS tables.
      - convert_to_torch (function): converts geographic GDFs to PyTorch tensors.
      - load_toronto_network_graph (function): builds network_nodes and network_edges GDFs.

    Output:
      - data (TorontoData): the loaded and filtered Toronto dataset.
      - dataset (ActivityDataset): the PyG dataset ready for ML model training.
      - network_edges (gpd.GeoDataFrame): edge geometries for map visualisation.
      - network_nodes (gpd.GeoDataFrame): node geometries for map visualisation.
    """
    # Load TorontoData and filter to include only users with subsector-level home locations
    data = TorontoData.load(cfg.data, project_root).with_filter("subsector")

    # Build the geographic GeoDataFrames for the city network:
    # network_nodes: GDF of all network node locations (subsectors, etc.)
    # network_edges: GDF of all edges as LineStrings (for map visualisation)
    network_nodes, network_edges = load_toronto_network_graph(
        data, cfg.data, project_root
    )

    # Convert the geographic data to PyTorch tensors for the ML pipeline:
    # network_graph    : PyG Data object (base graph topology + edge attributes)
    # spatial_features : per-user per-node home indicator, shape (N_users, N_nodes, 1)
    # spatial_labels   : per-user per-node visit labels, shape (N_users, N_nodes, 1)
    # demographics     : per-user socio-demographic features, shape (N_users, N_features)
    # distances        : per-user per-node distances from home, shape (N_users, N_nodes, 1)
    network_graph, spatial_features, spatial_labels, demographics, distances = convert_to_torch(data, network_nodes, network_edges)

    # Path where the ActivityDataset PyG files will be cached on disk
    dataset_path = project_root / cfg.data.paths.pyg_datasets

    # Construct the ActivityDataset (creates or loads the .pt files)
    dataset = ActivityDataset(dataset_path, network_graph, spatial_features, spatial_labels, demographics, distances)
    return data, dataset, network_edges, network_nodes


@app.cell
def _(data):
    """
    Description:
        Display cell — shows the raw person (individual demographics) table from the
        Toronto dataset to allow a quick inspection of the available columns and values.

    Input:
      - data (TorontoData): the loaded Toronto dataset.

    Output:
      - (pl.DataFrame displayed in notebook; nothing returned)
    """
    data.inputs.raw_person_df
    return


@app.cell
def _(data, pl):
    """
    Description:
        Builds a summary table that links trip-day counts, observed survey days, and
        unique location counts per user.  Used to check how many days each participant's
        GPS data was captured relative to the diary survey and how many distinct locations
        they visited.

    Input:
      - data (TorontoData): the loaded Toronto dataset.
      - pl (module): polars.

    Output:
      - obs (pl.DataFrame): per-person observation day count and total trip count.
      - res (pl.DataFrame): obs joined with trip-day counts and unique location counts.
    """
    # One column per diary survey day (7 days total): 1 = data submitted that day
    cols = [f"THATS D{n}DS? = 1" for n in range(1, 8)]
    # obs: number of days with diary data, total reported trips, experiment span
    obs = data.inputs.raw_person_df.select("person_id", pl.sum_horizontal(cols).alias("num_obs_days"), "THATS Number of Trips", span=pl.col("THATS Experiment start day") - pl.col("THATS Experiment end day"))
    # ttrips: number of unique trip departure days per user from the GPS journey data
    ttrips = data.user_journeys_df.group_by("user_id").agg(n_trip_days=pl.col("dep_day").n_unique().cast(pl.Int32))
    # locs: number of unique locations visited per user
    locs = data.location_visits.group_by("user_id").agg(num_locs=pl.col("loc_id").n_unique())

    # Join all three together: res enables cross-validation of GPS vs diary data
    res = ttrips.join(obs, left_on="user_id", right_on="person_id").with_columns(day_diff=pl.col("n_trip_days") - pl.col("num_obs_days")).join(locs, on="user_id")
    return obs, res


@app.cell
def _(alt, res):
    """
    Description:
        Plots a histogram of the number of GPS trip days per user to show the
        distribution of GPS tracking coverage across participants.

    Input:
      - alt (module): Altair for building the chart.
      - res (pl.DataFrame): summary table with n_trip_days per user.

    Output:
      - (Altair bar chart displayed in notebook; nothing exported)
    """
    alt.Chart(res).mark_bar().encode(x=alt.X("n_trip_days").bin(maxbins=50), y="count()")
    return


@app.cell
def _(alt, obs):
    """
    Description:
        Scatter plot of number of diary observation days (x) vs total reported trips
        (y).  Helps identify whether participants who logged more diary days also
        reported more trips, as a basic data-quality check.

    Input:
      - alt (module): Altair.
      - obs (pl.DataFrame): per-person diary days and trip count.

    Output:
      - (Altair scatter chart displayed in notebook; nothing exported)
    """
    alt.Chart(obs).mark_point().encode(x="num_obs_days", y="THATS Number of Trips")
    return


@app.cell
def _(alt, res):
    """
    Description:
        Bar chart of the number of observation days (treated as a discrete category),
        showing how many users submitted data for each specific number of diary days.

    Input:
      - alt (module): Altair.
      - res (pl.DataFrame): summary table with num_obs_days per user.

    Output:
      - (Altair bar chart displayed in notebook; nothing exported)
    """
    alt.Chart(res).mark_bar().encode(x="num_obs_days:N", y="count()")
    return


@app.cell
def _(pl, res):
    """
    Description:
        Displays users who have at most 7 GPS trip days AND whose GPS day count exactly
        matches their diary observation days (day_diff == 0).  These are the users with
        the most reliable coverage across both data sources.

    Input:
      - pl (module): polars.
      - res (pl.DataFrame): summary table with n_trip_days, num_obs_days, and day_diff.

    Output:
      - (filtered pl.DataFrame displayed in notebook; nothing exported)
    """
    res.filter(pl.col("n_trip_days") <= 7, day_diff=0)
    return


@app.cell
def _(alt, res):
    """
    Description:
        Wide scatter plot comparing the number of GPS trip days (x) with the total
        trips self-reported in the THATS diary (y).  A wide format (width=1000) is
        used to spread out users with many trip days for better readability.

    Input:
      - alt (module): Altair.
      - res (pl.DataFrame): summary table.

    Output:
      - (Altair scatter chart displayed in notebook; nothing exported)
    """
    alt.Chart(res).mark_point().encode(x="n_trip_days", y="THATS Number of Trips").properties(width=1000)
    return


@app.cell
def _(alt, res):
    """
    Description:
        Scatter plot of GPS trip days (x) vs unique locations visited (y).
        Shows whether participants with more GPS days tend to visit more distinct
        locations — a proxy for how well GPS coverage translates to location diversity.

    Input:
      - alt (module): Altair.
      - res (pl.DataFrame): summary table.

    Output:
      - (Altair scatter chart displayed in notebook; nothing exported)
    """
    alt.Chart(res).mark_point().encode(x="n_trip_days", y="num_locs")
    return


@app.cell
def _(alt, res):
    """
    Description:
        Scatter plot of diary observation days (x) vs unique locations visited (y).
        Complements the previous chart by checking whether diary coverage (not just
        GPS days) correlates with location diversity.

    Input:
      - alt (module): Altair.
      - res (pl.DataFrame): summary table.

    Output:
      - (Altair scatter chart displayed in notebook; nothing exported)
    """
    alt.Chart(res).mark_point().encode(x="num_obs_days", y="num_locs")
    return


@app.cell
def _(data):
    """
    Description:
        Display cell — shows the processed users DataFrame (one row per user with
        demographic features) so the analyst can check which features are available
        for the ML models.

    Input:
      - data (TorontoData): the loaded Toronto dataset.

    Output:
      - (pl.DataFrame displayed in notebook; nothing exported)
    """
    data.users_df
    return


@app.cell
def _(data, pl):
    """
    Description:
        Displays the unique departure days per user as a list column.  Useful for
        manually inspecting which calendar dates each user has GPS trip data for.

    Input:
      - data (TorontoData): the loaded Toronto dataset.
      - pl (module): polars.

    Output:
      - (pl.DataFrame displayed in notebook; nothing exported)
    """
    data.user_journeys_df.group_by("user_id", maintain_order=True).agg(n_trip_days=pl.col("dep_day").unique())
    return


@app.cell
def _(dataset):
    """
    Description:
        Imports the PyG DataLoader and displays the number of output classes in the
        dataset (should be 1 for binary node-level classification).  Also makes
        DataLoader available to downstream cells.

    Input:
      - dataset (ActivityDataset): the Toronto PyG dataset.

    Output:
      - (int displayed in notebook; DataLoader imported but not returned here)
    """
    from torch_geometric.loader import DataLoader

    dataset.num_classes  # should be 1 for binary per-node visit prediction
    return


@app.cell(hide_code=True)
def _():
    """
    Description: Render a description of the three components that make up the PyG ActivityDataset.

    Output:
      - (none): renders a Markdown note in the notebook UI.
    """
    mo.md(r"""
    The PyG Dataset is composed of three objects:
    1. network graph (PyG) - simple city2graph call on network_nodes & network_edges
    2. user spatial features (Tensor, dim N_users x N_nodes x 1) - is_home indicator, to be concatenated with NG.x in dataset
    3. user features (Tensor, dim N_users x N_features) - socio-demographics about users, can be concatenated with each row of NG.x
    """)
    return


@app.cell(hide_code=True)
def _():
    """
    Description: Render the visualisation section heading.

    Output:
      - (none): renders a Markdown heading in the notebook UI.
    """
    mo.md(r"""
    ## Visualisation
    """)
    return


@app.cell
def _(network_nodes):
    """
    Description:
        Creates a searchable dropdown widget populated with the network_nodes column
        names.  Allows the analyst to choose which attribute to colour on the map.

    Input:
      - network_nodes (gpd.GeoDataFrame): the Toronto network node GeoDataFrame.

    Output:
      - select_node_col (mo.ui.dropdown): column selector widget.
    """
    select_node_col = mo.ui.dropdown(
        list(network_nodes.columns), searchable=True, label="Column:"
    )
    return (select_node_col,)


@app.cell
def _(network_edges, network_nodes, select_node_col):
    """
    Description:
        Renders an interactive Folium map of the Toronto network with nodes coloured
        by the column selected in select_node_col.  Network edges are drawn in grey.
        The column selector widget appears above the map.

    Input:
      - network_edges (gpd.GeoDataFrame): edge geometries (drawn grey).
      - network_nodes (gpd.GeoDataFrame): node geometries to colour by attribute.
      - select_node_col (mo.ui.dropdown): which column to visualise on the nodes.

    Output:
      - (Folium map displayed in notebook; nothing exported)
    """
    _nodes = network_nodes  # .set_geometry("original_geometry") #if toggle_polygons.value else indiv_nodes
    _m = network_edges.explore(color="gray", tiles="Cartodb Positron")
    _m = _nodes.explore(
        m=_m, column=select_node_col.value, marker_kwds={"radius": 5}
    )

    mo.vstack([
        select_node_col,  # mo.hstack([select_user_id, select_node_col, toggle_polygons], justify="start"),
        _m,
    ])
    return


@app.cell(hide_code=True)
def _():
    """
    Description: Render the Exploratory Data Analysis section heading.

    Output:
      - (none): renders a Markdown heading in the notebook UI.
    """
    mo.md(r"""
    # Exploratory Data Analysis
    """)
    return


@app.cell
def _():
    """
    Description:
        Imports the Altair visualisation library used for the EDA charts (histograms,
        scatter plots, confusion matrices) in the rest of this notebook.

    Output:
      - alt (module): the Altair library, exported for downstream chart cells.
    """
    import altair as alt

    return (alt,)


@app.cell
def _():
    """
    Description:
        Imports the transport Mode enum and the two mode-label lookup dictionaries
        used to map raw mode strings from the GPS app and the manual diary to
        standardised Mode enum values.

        MANUAL_MODE_MAP: maps manual diary mode strings to Mode enum values.
        MODE_MAP:        maps GPS-predicted mode strings to Mode enum values.

    Output:
      - MANUAL_MODE_MAP (dict): manual diary mode -> Mode enum mapping.
      - MODE_MAP (dict): GPS-predicted mode -> Mode enum mapping.
      - Mode (Enum): the standardised transport mode enumeration.
    """
    from activitygraphs.base import Mode
    from activitygraphs.data.toronto import MANUAL_MODE_MAP, MODE_MAP

    return MANUAL_MODE_MAP, MODE_MAP, Mode


@app.cell
def _():
    """
    Description:
        Imports the itertools standard library module, used in plot_enum_confusion_matrix
        to generate all (predicted, actual) category pairs via itertools.product to
        ensure the confusion matrix has a cell for every combination.

    Output:
      - itertools (module): standard library module for combinatorial iteration.
    """
    import itertools

    return (itertools,)


@app.cell(hide_code=True)
def _():
    """
    Description: Render the Trip Modes sub-section heading and a note about the labeling analysis.

    Output:
      - (none): renders a Markdown heading in the notebook UI.
    """
    mo.md(r"""
    ## Trip modes

    First: investigate labeling behaviour
    """)
    return


@app.cell(hide_code=True)
def _():
    """
    Description: Render a label preceding the null-proportion computation cell.

    Output:
      - (none): renders a Markdown note in the notebook UI.
    """
    mo.md(r"""
    Proportion of unlabeled trips
    """)
    return


@app.cell(hide_code=True)
def _(data):
    """
    Description:
        Computes and displays the proportion of null values in the predicted_modes
        and manual_mode columns of the raw journeys table.  A high null proportion
        in manual_mode means most users did not manually label their trips.

    Input:
      - data (TorontoData): the loaded Toronto dataset.

    Output:
      - (proportion table displayed in notebook; nothing exported)
    """
    data.inputs.raw_journeys_df.select(
        "predicted_modes", "manual_mode"
    ).null_count() / len(data.inputs.raw_journeys_df)
    return


@app.cell(hide_code=True)
def _():
    """
    Description: Render a question label preceding the per-user labeling frequency histogram.

    Output:
      - (none): renders a Markdown note in the notebook UI.
    """
    mo.md(r"""
    How often do people label their trips?
    """)
    return


@app.cell
def _(alt, data, pl):
    """
    Description:
        Plots a histogram showing how frequently each user manually labels their trip
        modes.  The x-axis is the fraction of trips labeled (0–1) and the y-axis is
        the number of users.  This reveals how many users label most trips vs none.

    Input:
      - alt (module): Altair.
      - data (TorontoData): the loaded Toronto dataset.
      - pl (module): polars.

    Output:
      - (Altair bar chart displayed in notebook; nothing exported)
    """
    # For each user (app_id), compute the fraction of trips with a manual mode label
    _mode_label_freq = data.inputs.raw_journeys_df.group_by("app_id").agg(
        pct_labeled=pl.col("manual_mode").is_not_null().sum() / pl.len()
    )

    _chart = _mode_label_freq.plot.bar(
        x=alt.X("pct_labeled", bin=alt.Bin(maxbins=20)), y="count()"
    ).properties(
        title="Distribution of proportion of manually-labeled trip modes per user"
    )
    _chart
    return


@app.cell(hide_code=True)
def _(alt, itertools, pl):
    """
    Description:
        Defines the plot_enum_confusion_matrix helper function for visualising the
        agreement between GPS-predicted and manually-entered categorical labels.

    Input:
      - alt (module): Altair.
      - itertools (module): used to generate all category combinations.
      - pl (module): polars.

    Output:
      - plot_enum_confusion_matrix (function): exported for use in the confusion-matrix
            display cells below.
    """
    def plot_enum_confusion_matrix(
        df: pl.DataFrame, pred_col: str, actual_col: str, enum, scale_type="linear"
    ) -> alt.Chart:
        """
        Description:
            Creates an Altair confusion-matrix heatmap for two categorical columns,
            with row and column marginal totals shown as annotation bars alongside the
            main matrix.  Useful for comparing predicted vs manually-entered modes/purposes.

        Input:
          - df (pl.DataFrame): DataFrame containing the two categorical columns to compare.
          - pred_col (str): name of the predicted/GPS-derived categorical column.
          - actual_col (str): name of the manually-entered categorical column.
          - enum: an iterable of all possible category values (used to ensure all cells are
                present even if count = 0).
          - scale_type (str): Altair colour scale type — "linear" or "log".
                "log" makes rare off-diagonal cells more visible.  Defaults to "linear".

        Output:
          - (alt.Chart): a composite Altair chart: main confusion matrix with marginals.
        """
        width, height = 400, 400

        base = pl.DataFrame(
            list(itertools.product(enum, enum)),
            orient="row",
            schema=[pred_col, actual_col],
        )

        modes = df.group_by(pred_col, actual_col).agg(count=pl.len())
        modes = base.join(modes, on=[pred_col, actual_col], how="left").fill_null(
            0
        )
        modes = modes.sort(pred_col, actual_col)

        row_totals = modes.group_by(actual_col, maintain_order=True).agg(
            pl.col("count").sum()
        )
        col_totals = modes.group_by(pred_col, maintain_order=True).agg(
            pl.col("count").sum()
        )

        labels = (
            alt
            .Chart(modes)
            .mark_text()
            .encode(
                x=f"{pred_col}:N",
                y=f"{actual_col}:N",
                text=alt.Text("count:Q"),
            )
        )

        boxes = (
            alt
            .Chart(modes)
            .mark_rect()
            .encode(
                x=f"{pred_col}:N",
                y=f"{actual_col}:N",
                color=alt.Color(
                    "count:Q", scale=alt.Scale(type=scale_type, domainMin=1)
                ),
                tooltip=[pred_col, actual_col, "count"],
            )
        )

        row_boxes = (
            alt
            .Chart(row_totals)
            .mark_rect()
            .encode(
                y=alt.Y(f"{actual_col}:N", axis=None),
                color=alt.Color(
                    "count:Q",
                    scale=alt.Scale(type=scale_type, domainMin=1),
                ),
            )
        )

        row_labels = (
            alt
            .Chart(row_totals)
            .mark_text()
            .encode(
                y=alt.Y(
                    f"{actual_col}:N",
                ),
                text=alt.Text("count:Q"),
            )
        )

        col_boxes = (
            alt
            .Chart(col_totals)
            .mark_rect()
            .encode(
                x=alt.X(f"{pred_col}:N", axis=None),
                color=alt.Color(
                    "count:Q",
                    scale=alt.Scale(type=scale_type, domainMin=1),
                ),
            )
        )

        col_labels = (
            alt
            .Chart(col_totals)
            .mark_text()
            .encode(
                x=alt.X(
                    f"{pred_col}:N",
                ),
                text=alt.Text("count:Q"),
            )
        )

        main = (boxes + labels).properties(width=width, height=height)
        rows = (row_boxes + row_labels).properties(width=30, height=height)
        cols = (col_boxes + col_labels).properties(width=width, height=30)

        bottom = (main | rows).resolve_scale(y="shared")

        return (
            (cols & bottom)
            .resolve_scale(x="shared", color="shared")
            .properties(
                title="Confusion matrix of GPS predicted vs manually-entered modes",
            )
        )

    return (plot_enum_confusion_matrix,)


@app.cell(hide_code=True)
def _():
    """
    Description:
        Creates a dropdown widget to toggle the colour scale of the mode confusion
        matrix between linear and log.  Log scale makes rare off-diagonal cells more
        visible when the diagonal dominates.

    Output:
      - drop_down_log_scale (mo.ui.dropdown): colour scale selector widget.
    """
    drop_down_log_scale = mo.ui.dropdown(
        ["log", "linear"], label="Color scale", value="linear"
    )
    return (drop_down_log_scale,)


@app.cell(hide_code=True)
def _():
    """
    Description: Render a note about the mode confusion matrix shown below it.

    Output:
      - (none): renders a Markdown note in the notebook UI.
    """
    mo.md(r"""
    Comparison between predicted mode and manually-entered mode. In genereal, it's somewhat confused.
    """)
    return


@app.cell(hide_code=True)
def _(
    MANUAL_MODE_MAP,
    MODE_MAP,
    Mode,
    data,
    drop_down_log_scale,
    pl,
    plot_enum_confusion_matrix,
):
    """
    Description:
        Builds the mode confusion matrix comparing GPS-predicted trip modes against
        manually-entered modes, then renders it with the currently selected colour scale.
        The COACH mode is excluded from the enum because it is too rare to be informative.

    Input:
      - MANUAL_MODE_MAP, MODE_MAP (dict): mode label -> Mode enum mappings.
      - Mode (Enum): the standardised transport mode enumeration.
      - data (TorontoData): the loaded Toronto dataset.
      - drop_down_log_scale (mo.ui.dropdown): selects linear or log colour scale.
      - pl (module): polars.
      - plot_enum_confusion_matrix (function): builds the Altair confusion matrix chart.

    Output:
      - (Altair confusion matrix + dropdown displayed in notebook; nothing exported)
    """
    # Exclude COACH as it has too few samples to show in the matrix
    _mode_enum = [m for m in Mode if m not in [Mode.COACH]]
    # Map raw string labels to Mode enum values; unknowns default to Mode.UNKNOWN
    _df = data.inputs.raw_journeys_df.select(
        pl.col("predicted_modes").replace_strict(MODE_MAP, default=Mode.UNKNOWN),
        pl.col("manual_mode").replace_strict(
            MANUAL_MODE_MAP, default=Mode.UNKNOWN
        ),
    )

    _chart = plot_enum_confusion_matrix(
        _df,
        "predicted_modes",
        "manual_mode",
        Mode,
        scale_type=drop_down_log_scale.value,
    )

    mo.vstack([drop_down_log_scale, _chart])
    return


@app.cell(hide_code=True)
def _():
    """
    Description: Render the Purposes EDA sub-section heading.

    Output:
      - (none): renders a Markdown heading in the notebook UI.
    """
    mo.md(r"""
    ## Purposes

    - Link purposes from activities to trips, then compare with accuracy from manually labeled trip purposes
    """)
    return


@app.cell
def _(data):
    """
    Description:
        Builds the collapsed activity-level DataFrame from the raw hourly activity
        diary records and the GPS journey data.  Each row in activs represents one
        contiguous activity (e.g. "work from 09:00 to 17:00") rather than one diary hour.

    Input:
      - data (TorontoData): the loaded Toronto dataset.

    Output:
      - (activities DataFrame printed with head(); nothing exported)
    """
    from activitygraphs.data.toronto import build_toronto_activities

    # Collapse the hourly diary records into contiguous activity spans
    activs = build_toronto_activities(data.inputs, data.user_journeys_df)
    print(activs.head())
    return


@app.cell
def _(data, pl):
    """
    Description:
        Defines compare_trip_to_activity (an exploratory function for the Purposes EDA
        section) and immediately applies it to data.activities_df and data.user_journeys_df.
        The function attempts to match each trip to the activity recorded at the same
        time-of-arrival to check whether the app's automated purpose label agrees with
        the manually-entered diary purpose.

    Input:
      - data (TorontoData): the loaded Toronto dataset.
      - pl (module): polars.

    Output:
      - compared (pl.DataFrame): trip_summaries joined to matched activity purposes,
            with a purpose_match boolean column.
    """
    def compare_trip_to_activity(
        activities: pl.DataFrame,
        trips: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Join trips to the collapsed activity that 'receives' each trip
        (i.e. the activity whose end_trip_id matches the trip_id, or
        failing that, the activity happening at the arrival hour with
        the closest purpose match).

        Returns the trips frame with extra columns:
          - act_purpose        : purpose from the activity log
          - purpose_match      : bool, whether arr_purpose == act_purpose
          - match_method       : how the match was made
        """

        # Combine trip legs into single trips
        trip_summaries = (
            trips
            .sort("user_id", "journey_id", "leg_id")
            .group_by("user_id", "journey_id", maintain_order=True)
            .agg(
                dep_day=pl.col("dep_day").first(),
                dep_time=pl.col("dep_time").first(),
                dep_purpose=pl.col("dep_purpose").first(),
                arr_purpose=pl.col("arr_purpose").last(),
                last_leg_dep=pl.col("dep_time").last(),
                last_leg_duration=pl.col("duration").last(),
                arr_day=pl.col("dep_day").last(),
            )
            .with_columns(
                arr_time=(
                    pl.col("arr_day").dt.combine(pl.col("last_leg_dep"))
                    + pl.col("last_leg_duration")
                ),
            )
            .with_columns(
                arr_day=pl.col("arr_time").dt.date(),
                arr_time=pl.col("arr_time").dt.time(),
            )
            .drop("last_leg_dep", "last_leg_duration")
        )

        # Join trips to activities where the arrival falls within the span.
        # Since act_hour has 1hr resolution, an arrival at e.g. 16:30 should
        # match an activity spanning 16:00–18:00.  We truncate arrival time
        # to the hour for comparison.
        matched = (
            trip_summaries
            .with_columns(arr_hour=pl.col("arr_time").dt.hour())
            .join(
                activities,
                left_on=["user_id", "arr_day"],
                right_on=["person_id", "act_date"],
                how="left",
                suffix="_act",
            )
            .filter(
                (pl.col("arr_hour") >= pl.col("start_time").dt.hour())
                & (pl.col("arr_hour") <= pl.col("end_time").dt.hour())
            )
            .with_columns(
                purpose_match=(pl.col("arr_purpose") == pl.col("act_purpose")),
                start_hour=pl.col("start_time").dt.hour(),
            )
            # Prefer: 1) purpose match, 2) latest start time (most recent activity)
            .sort(
                "user_id",
                "journey_id",
                "purpose_match",
                "start_hour",
                descending=[False, False, True, True],
            )
            .group_by("user_id", "journey_id")
            .first()
        )

        # Re-attach trips that had no matching activity at all
        unmatched = trip_summaries.join(
            matched.select("user_id", "journey_id"),
            on=["user_id", "journey_id"],
            how="anti",
        ).with_columns(
            act_purpose=pl.lit(None, dtype=pl.Categorical),
            purpose_match=pl.lit(None, dtype=pl.Boolean),
            # Add columns that the matched frame has so concat works
            start_time=pl.lit(None, dtype=pl.Time),
            end_time=pl.lit(None, dtype=pl.Time),
            n_hours=pl.lit(None, dtype=pl.UInt32),
            hh_id=pl.lit(None, dtype=pl.Utf8),
            arr_hour=pl.lit(None, dtype=pl.Int8),
        )

        return pl.concat([matched, unmatched], how="diagonal").sort(
            "user_id", "dep_day", "dep_time"
        )


    compared = compare_trip_to_activity(data.activities_df, data.user_journeys_df)
    compared
    return (compared,)


@app.cell(hide_code=True)
def _():
    """
    Description: Render a research question label about unlabeled trip purposes.

    Output:
      - (none): renders a Markdown note in the notebook UI.
    """
    mo.md(r"""
    - What proportion of unlabled trip purposes actually have corresponding data
    """)
    return


@app.cell(hide_code=True)
def _():
    """
    Description: Render a note about a potential future rule-finding ML algorithm.

    Output:
      - (none): renders a Markdown note in the notebook UI.
    """
    mo.md(r"""
    - Rule finding ML algorithm (perhaps)
    """)
    return


@app.cell
def _(data):
    """
    Description:
        Extracts the two main DataFrames used in the purpose-matching diagnostic cells
        below as convenience aliases.

    Input:
      - data (TorontoData): the loaded Toronto dataset.

    Output:
      - activities (pl.DataFrame): collapsed activity records with purpose, time, and person.
      - trips (pl.DataFrame): GPS trip legs with departure/arrival purposes and times.
    """
    trips = data.user_journeys_df      # one row per GPS trip leg
    activities = data.activities_df    # one row per contiguous activity span
    return activities, trips


@app.cell
def _(activities, trips):
    """
    Description:
        Diagnostic cell — counts how many unique users appear in the trips table vs
        the activities table and how many users appear in both.  This checks whether
        most GPS users also completed the activity diary (necessary for purpose matching).

    Input:
      - activities (pl.DataFrame): activity diary records.
      - trips (pl.DataFrame): GPS trip legs.

    Output:
      - (counts printed to console; nothing exported)
    """
    trip_users = trips.select("user_id").unique()     # users with GPS trip data
    act_users = activities.select("person_id").unique()  # users with diary activity data
    print("Trip users:", trip_users.height)
    print("Activity persons:", act_users.height)
    print(
        "Overlap:",
        trip_users.join(
            act_users, left_on="user_id", right_on="person_id", how="inner"
        ).height,
    )
    return


@app.cell
def _(activities, trips):
    """
    Description:
        Diagnostic cell — checks how many distinct calendar dates appear in both the
        trips table and the activities table.  A high date overlap means purpose matching
        has many opportunities to link trips to same-day diary records.

    Input:
      - activities (pl.DataFrame): activity diary records with an act_date column.
      - trips (pl.DataFrame): GPS trip legs with a dep_day column.

    Output:
      - (counts printed to console; nothing exported)
    """
    # Check 2: Do the dates overlap?
    trip_dates = trips.select("dep_day").unique()       # unique departure dates in GPS data
    act_dates = activities.select("act_date").unique()  # unique dates in the activity diary
    print("Trip dates:", trip_dates.height)
    print("Activity dates:", act_dates.height)
    print(
        "Date overlap:",
        trip_dates.join(
            act_dates, left_on="dep_day", right_on="act_date", how="inner"
        ).height,
    )
    return


@app.cell
def _(activities, trips):
    """
    Description:
        Per-user date overlap diagnostic.  For each user, counts how many dates appear
        in both their GPS trip records and their activity diary.  Reports how many users
        have at least one overlapping date and the mean number of overlapping dates.

    Input:
      - activities (pl.DataFrame): activity diary records.
      - trips (pl.DataFrame): GPS trip legs.

    Output:
      - act_user_dates (pl.DataFrame): per-user activity dates (renamed for joining),
            exported for use in the matchable-trips cell below.
    """
    # Per-user date coverage
    trip_user_dates = (
        trips.select("user_id", "dep_day").unique().rename({"dep_day": "date"})
    )
    act_user_dates = (
        activities
        .select("person_id", "act_date")
        .unique()
        .rename({"person_id": "user_id", "act_date": "date"})
    )

    # Inner join: keep only (user, date) pairs present in both sources
    per_user = (
        trip_user_dates
        .join(act_user_dates, on=["user_id", "date"], how="inner")
        .group_by("user_id")
        .len()
    )
    print(f"Users with at least 1 overlapping date: {per_user.height}")
    print(f"Mean overlapping dates per user: {per_user['len'].mean():.1f}")
    print(f"Users with 0 overlapping dates: {588 - per_user.height}")
    return (act_user_dates,)


@app.cell
def _(pl, trips):
    """
    Description:
        Collapses the GPS trip legs into single-trip summaries by grouping on
        (user_id, journey_id).  For each trip, the first leg gives the departure
        info and the last leg's end time (computed from dep_time + duration) gives
        the arrival time, which is needed to match trips to diary activities.

    Input:
      - pl (module): polars.
      - trips (pl.DataFrame): GPS trip legs with leg-level timing columns.

    Output:
      - trip_summaries (pl.DataFrame): one row per trip with dep_day, dep_time,
            dep_purpose, arr_purpose, arr_day, and arr_time columns.
    """
    trip_summaries = (
        trips
        .sort("user_id", "journey_id", "leg_id")
        .group_by("user_id", "journey_id", maintain_order=True)
        .agg(
            dep_day=pl.col("dep_day").first(),
            dep_time=pl.col("dep_time").first(),
            dep_purpose=pl.col("dep_purpose").first(),
            arr_purpose=pl.col("arr_purpose").last(),
            last_leg_dep=pl.col("dep_time").last(),
            last_leg_duration=pl.col("duration").last(),
            arr_day=pl.col("dep_day").last(),
        )
        .with_columns(
            arr_time=(
                pl.col("arr_day").dt.combine(pl.col("last_leg_dep"))
                + pl.col("last_leg_duration")
            ),
        )
        .with_columns(
            arr_day=pl.col("arr_time").dt.date(),
            arr_time=pl.col("arr_time").dt.time(),
        )
        .drop("last_leg_dep", "last_leg_duration")
    )
    return (trip_summaries,)


@app.cell
def _(act_user_dates, compared, pl, trip_summaries):
    """
    Description:
        Evaluates how well the purpose-matching pipeline performed by counting:
          (a) how many trips were "matchable" (user + arrival date both in the diary).
          (b) of those, how many were actually matched to an activity record.
          (c) of those matched, how many had matching purposes.
        Also counts matchable trips where the arrival time was null.

    Input:
      - act_user_dates (pl.DataFrame): per-user activity dates for matchability check.
      - compared (pl.DataFrame): the output of compare_trip_to_activity.
      - pl (module): polars.
      - trip_summaries (pl.DataFrame): collapsed GPS trips with arrival times.

    Output:
      - matchable_compared (pl.DataFrame): the compared rows restricted to matchable trips,
            exported for the mismatch analysis cell below.
    """
    # Trips that are actually matchable (user + date exists in activities)
    matchable_trips = trip_summaries.join(
        act_user_dates,
        left_on=["user_id", "arr_day"],
        right_on=["user_id", "date"],
        how="inner",
    )
    print(f"Matchable trips: {matchable_trips.height}")

    matchable_compared = compared.join(
        matchable_trips.select("user_id", "journey_id"),
        on=["user_id", "journey_id"],
        how="inner",
    )
    total = matchable_compared.height
    matched = matchable_compared.filter(pl.col("act_purpose").is_not_null()).height
    purpose_matched = matchable_compared.filter(
        pl.col("purpose_match") == True
    ).height
    print(f"Matchable trips: {total}")
    print(f"Matched to activity: {matched} ({matched / total:.1%})")
    print(
        f"Purpose agrees: {purpose_matched} ({purpose_matched / max(matched, 1):.1%} of matched)"
    )

    # How many matchable trips have null arr_hour?
    null_arr = matchable_trips.filter(pl.col("arr_time").is_null()).height
    print(
        f"Matchable but null arr_hour: {null_arr} ({null_arr / matchable_trips.height:.1%})"
    )
    return (matchable_compared,)


@app.cell
def _(matchable_compared, pl):
    """
    Description:
        Summarises the most frequent purpose mismatches (GPS-predicted vs diary) and
        counts how many trips have null or "unknown" purposes in each source.
        Used to diagnose whether poor match rates are due to labelling noise or
        genuine temporal misalignment.

    Input:
      - matchable_compared (pl.DataFrame): matched trips restricted to matchable rows.
      - pl (module): polars.

    Output:
      - (mismatch table and null counts printed to console; nothing exported)
    """
    # What are the most common mismatches?
    mismatches = (
        matchable_compared
        .filter(pl.col("purpose_match") == False)
        .group_by("arr_purpose", "act_purpose")
        .len()
        .sort("len", descending=True)
    )
    print("Top 15 purpose mismatches:")
    print(mismatches.head(15))

    # How many trip purposes are already unknown?
    print(
        matchable_compared.select(
            arr_null=pl.col("arr_purpose").is_null().sum(),
            arr_unknown=(pl.col("arr_purpose").cast(pl.Utf8) == "unknown").sum(),
            act_null=pl.col("act_purpose").is_null().sum(),
            act_unknown=(pl.col("act_purpose").cast(pl.Utf8) == "unknown").sum(),
        )
    )
    return


if __name__ == "__main__":
    app.run()
