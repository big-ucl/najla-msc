"""
Module: data_processing.py

Description:
    A Marimo interactive notebook for loading, cleaning and visualising the raw LTDS
    (London Travel Demand Survey) activity dataset.

    The notebook performs the following steps:
      1. Optionally converts LTDS Access .xlsx exports to Parquet format.
      2. Loads or reprocesses the raw data into an ActivityDataset (households, persons,
         trips, locations).
      3. Builds an ActivityGraph from the dataset — a per-household directed multigraph
         where nodes are visited locations and edges are trips.
      4. Visualises sample household graphs both as abstract network diagrams and overlaid
         on a basemap.

Dependencies:
    - activitygraphs library (ActivityDataset, ActivityGraph)
    - Marimo, NumPy, Polars, GeoPandas, Contextily
"""

import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    """
    Description: Entry cell that imports Marimo and displays the notebook title heading.

    Input:
      - (none)

    Output:
      - mo (module): the Marimo module; exported for use by all subsequent cells.
    """
    import marimo as mo

    mo.md("# Data processing")
    return (mo,)


@app.cell
def _():
    """
    Description: Import all third-party libraries needed by the notebook and fix random
    seeds for reproducibility. Seeds are set on both NumPy and Python's built-in ``random``
    module to ensure that any random sampling (e.g. household selection) is deterministic
    across notebook runs.

    Input:
      - (none)

    Output:
      - Path (type): ``pathlib.Path`` class for building file paths.
      - cly (module): ``contextily`` for adding web tile basemaps to Matplotlib plots.
      - gpd (module): ``geopandas`` for geographic data manipulation.
      - load_config (callable): function to load the project configuration YAML.
    """
    # Import modules
    from pathlib import Path
    from config import load_config
    import numpy as np
    import random
    import contextily as cly
    import geopandas as gpd

    # Set random seeds
    np.random.seed(42)
    random.seed(42)
    return Path, cly, gpd, load_config


@app.cell
def _(load_config, mo):
    """
    Description: Resolve the project root directory (one level above the notebook) and
    load the project configuration from the YAML file found there.

    Input:
      - load_config (callable): from the previous cell.
      - mo (module): Marimo module (provides ``notebook_dir()``).

    Output:
      - cfg: the loaded configuration object with data paths and input file names.
    """
    project_root = mo.notebook_dir().parent  # one directory up from the notebooks folder
    cfg = load_config(project_root)
    return (cfg,)


@app.cell
def _(mo):
    """
    Description: Display a Marimo Markdown section heading for the "Excel to Parquet
    conversion" section, including a note that this step only needs to be run when
    converting raw LTDS Access `.xlsx` exports for the first time. Pure presentation.

    Input:
      - mo (module): Marimo (for ``mo.md``).

    Output:
      - (none): renders the section heading in the notebook UI.
    """
    mo.md(
        r"""
    # Excel to parquet conversion

    Does not need to be run unless converting from LTDS Access `.xslx` exports
    """
    )
    return


@app.cell
def _(mo):
    """
    Description: Create and display a warning-style run button that the user must click
    to trigger the Excel-to-Parquet conversion. The button is styled as a warning
    (``kind="warn"``) to make it clear this is a potentially time-consuming, destructive
    step that should not be run accidentally.

    Input:
      - mo (module): Marimo (for ``mo.ui.run_button``).

    Output:
      - run_button (mo.ui.run_button): the button widget; its ``.value`` attribute becomes
        ``True`` when clicked, which unlocks execution of the conversion cell below.
    """
    run_button = mo.ui.run_button(kind="warn", label="Run excel to parquet conversion")
    run_button
    return (run_button,)


@app.cell
def _(Path, cfg, mo, run_button):
    """
    Description: Convert raw LTDS Access `.xlsx` exports to Parquet format. This cell only
    runs when the user clicks the "Run excel to parquet conversion" button (guarded by
    ``mo.stop``). The converted Parquet files are written to the raw data directory defined
    in the configuration and are a prerequisite for all subsequent data loading.

    Input:
      - Path (type): pathlib.Path for constructing file paths.
      - cfg: project configuration (provides raw data paths and input file names).
      - mo (module): Marimo (provides ``mo.stop`` and ``mo.md``).
      - run_button (mo.ui.run_button): the button that gates this cell's execution.

    Output:
      - (none): the cell writes Parquet files to disk as a side effect.
    """
    from utils import convert_excel_to_parquet

    mo.stop(not run_button.value, mo.md("Click button above to run conversion"))

    _files = (Path(s) for s in cfg.inputs.values())
    convert_excel_to_parquet(cfg.data.paths.raw, *_files)
    return


@app.cell
def _(mo):
    """
    Description: Display a Marimo Markdown section heading for the "Data loading and
    processing" part of the notebook. Subsequent cells load or reprocess the LTDS
    ActivityDataset and display sample records. Pure presentation; no computation.

    Input:
      - mo (module): Marimo (for ``mo.md``).

    Output:
      - (none): renders the section heading in the notebook UI.
    """
    mo.md(r"""# Data loading and processing""")
    return


@app.cell
def _(mo):
    """
    Description: Create and display a run button that forces a full reprocessing of the
    raw LTDS data. If the button has not been clicked the notebook will instead load the
    previously saved ActivityDataset from disk (fast path).

    Input:
      - mo (module): Marimo (for ``mo.ui.run_button``).

    Output:
      - reprocess_button (mo.ui.run_button): the button widget; its ``.value`` is ``True``
        when clicked, causing the data-loading cell to re-run the full processing pipeline.
    """
    reprocess_button = mo.ui.run_button(label="Reprocess raw data")
    reprocess_button
    return (reprocess_button,)


@app.cell
def _(cfg, mo, reprocess_button):
    """
    Description: Load the LTDS ``ActivityDataset`` from disk if it already exists, or
    re-process the raw data from scratch if the "Reprocess raw data" button has been clicked.
    Saving to disk avoids the slow processing step on subsequent notebook runs.

    Input:
      - cfg: project configuration (provides dataset name and file paths).
      - mo (module): Marimo (provides spinner status widget and ``mo.status``).
      - reprocess_button (mo.ui.run_button): if clicked, forces a full reprocessing run.

    Output:
      - dataset (ActivityDataset): the loaded or freshly processed dataset object containing
        household, person, trip, and location DataFrames.
    """
    from archive.exploration.dataprocessing import ActivityDataset
    from data.ltds import read_and_process_ltds

    _dataset_name = cfg.data.name
    _dataset_path = cfg.data.paths.act_dataset

    if ActivityDataset.exists_on_disk(_dataset_path, _dataset_name) and not reprocess_button.value:
        dataset = ActivityDataset.load(_dataset_path, _dataset_name)
        print(f"Loaded `{dataset.name}` dataset from disk.")
    else:
        with mo.status.spinner(title="Processing data...") as _spinner:
            dataset = read_and_process_ltds(cfg.data)
            _spinner.update("Saving to file...")
            dataset.save(_dataset_path)
            _spinner.update("Done")
        print(f"Read, processed and saved `{dataset.name}` dataset from raw data")

    dataset.name
    return (dataset,)


@app.cell
def _(dataset):
    """
    Description: Display 10 randomly sampled rows from the household-person DataFrame
    as a quick data sanity check.

    Input:
      - dataset (ActivityDataset): the loaded dataset from the previous cell.

    Output:
      - (none): renders the sample table in the notebook UI.
    """
    dataset.hh_person_df.sample(10)
    return


@app.cell
def _(dataset):
    """
    Description: Display the full trip DataFrame for inspection.

    Input:
      - dataset (ActivityDataset): the loaded dataset.

    Output:
      - (none): renders the trip DataFrame in the notebook UI.
    """
    dataset.trip_df
    return


@app.cell
def _(dataset):
    """
    Description: Display the first few rows of the location reference DataFrame as a
    quick check that postcodes and municipality codes have been decoded correctly.

    Input:
      - dataset (ActivityDataset): the loaded dataset.

    Output:
      - (none): renders the first 5 rows in the notebook UI.
    """
    dataset.location_df.head()
    return


@app.cell
def _(mo):
    """
    Description: Display a Marimo Markdown section heading for the "Graph Generation"
    part of the notebook. Subsequent cells build the ActivityGraph from the dataset and
    expose interactive visualisations. Pure presentation; no computation.

    Input:
      - mo (module): Marimo (for ``mo.md``).

    Output:
      - (none): renders the section heading in the notebook UI.
    """
    mo.md("""# Graph Generation""")
    return


@app.cell
def _(dataset):
    """
    Description: Build an ``ActivityGraph`` from the cleaned dataset. An ActivityGraph
    groups trips by household and stores them as per-household directed multigraphs where
    nodes are visited locations and edges are individual trips. Prints the total number
    of household subgraphs for a quick sanity check.

    Input:
      - dataset (ActivityDataset): the loaded dataset.

    Output:
      - graph (ActivityGraph): the activity graph container with one subgraph per household.
    """
    from archive.exploration import ActivityGraph

    graph = ActivityGraph.from_dataset(dataset)

    print(f"Loaded {graph} with {graph.n_subgraphs} subgraphs")
    return (graph,)


@app.cell
def _(graph, new_graph_button):
    """
    Description: Prepare a NetworkX representation of the currently selected household's
    activity graph, along with visual styling dictionaries (edge line styles, node colours,
    node labels) for Matplotlib rendering.

    Input:
      - graph (ActivityGraph): the full activity graph container from the previous cell.
      - new_graph_button (mo.ui.button): provides the currently selected ``hh_id`` via
        its ``.value`` attribute (updated each time the user clicks "Sample new graph").

    Output:
      - G (nx.MultiDiGraph): the NetworkX directed multigraph for the selected household.
      - draw_hh_graph (callable): the plotting helper function (exported for the render cell).
      - hh_graph (ActivityGraph subobject): the raw ActivityGraph for the selected household,
        containing ``node_df`` and ``edge_df`` (used for the map display cell).
      - node_colours (dict): node colour mapping keyed by purpose (for Matplotlib).
      - node_labels (dict): short node label mapping keyed by purpose (for Matplotlib).
    """
    from plotting import (
        draw_hh_graph,
        line_styles_by_key,
        node_colours_by_purpose,
        node_short_labels_by_purpose,
    )

    _hh_id = new_graph_button.value  # currently selected household ID

    G = graph.to_nx(_hh_id)                               # convert to NetworkX graph
    hh_graph = graph.hh_graph(_hh_id)                     # raw ActivityGraph sub-object
    line_styles = line_styles_by_key(G, key="person_id")  # different line style per person
    node_colours = node_colours_by_purpose(G)              # colour nodes by activity purpose
    node_labels = node_short_labels_by_purpose(G)          # short text labels on nodes
    return G, draw_hh_graph, hh_graph, node_colours, node_labels


@app.cell
def _(mo):
    """
    Description: Define the pre-selected "interesting" household ID used as the initial
    graph to display, and create a toggle switch that enables the "Sample new graph" button.

    Input:
      - mo (module): Marimo.

    Output:
      - interesting_hh_id (str): the hardcoded hh_id of a household with a visually
        interesting multi-person, multi-location activity graph.
      - new_graph_switch (mo.ui.switch): when toggled on, enables the random-sampling
        button in the next cell.
    """
    interesting_hh_id = "12109151"  # pre-selected household with interesting trip structure
    new_graph_switch = mo.ui.switch(label="Use sampled graph")
    return interesting_hh_id, new_graph_switch


@app.cell
def _(graph):
    """
    Description: Display the total number of household subgraphs in the activity graph
    as a quick summary statistic.

    Input:
      - graph (ActivityGraph): the full activity graph container.

    Output:
      - (none): renders the count in the notebook UI.
    """
    graph.n_subgraphs
    return


@app.cell
def _(mo):
    """
    Description: Display a Marimo Markdown section heading for the "Visualisation" part
    of the notebook, which contains the interactive household-graph drawing and map overlay
    cells. Pure presentation; no computation.

    Input:
      - mo (module): Marimo (for ``mo.md``).

    Output:
      - (none): renders the section heading in the notebook UI.
    """
    mo.md("""# Visualisation""")
    return


@app.cell
def _(dataset, interesting_hh_id, mo, new_graph_switch):
    """
    Description: Create the interactive "Sample new graph" button. On each click the
    button picks a random household with at least one trip record and updates its ``.value``
    to the new household ID, which reactively triggers the graph-drawing cells below.
    The button is disabled when the toggle switch is off to prevent accidental sampling.

    Input:
      - dataset (ActivityDataset): the loaded dataset; used inside the click callback to
        sample a household ID from ``hh_person_df`` and verify it against ``trip_df``.
      - interesting_hh_id (str): the pre-selected "interesting" household shown initially.
      - mo (module): Marimo (for ``mo.ui.button`` and ``mo.hstack``).
      - new_graph_switch (mo.ui.switch): when ``True``, enables the sampling button.

    Output:
      - new_graph_button (mo.ui.button): the button widget whose ``.value`` always holds
        the currently active household ID string.
    """
    def _draw_new_random_hh_id(value: str) -> str:
        """
        Description:
            Callback that picks a random household ID that has at least one trip entry.
            Called each time the "Sample new graph" button is clicked.

        Input:
          - value (str): the current button value (previous hh_id, not used directly).

        Output:
          - (str): a household ID string that exists in both hh_person_df and trip_df.
        """
        while True:
            # Sample one random hh_id from the household-person table
            hh_id = dataset.hh_person_df.select("hh_id").unique().sample(1)[0, "hh_id"]

            # Only return the ID if this household also has trip records; otherwise
            # re-sample (some households may appear only in demographics, not in trips).
            if hh_id in dataset.trip_df["hh_id"]:
                return hh_id

    # Button that controls which household graph is displayed.
    # Disabled unless the toggle switch is on.  Starts with a pre-selected interesting example.
    new_graph_button = mo.ui.button(
        label="Sample new graph from dataset ",
        disabled=not new_graph_switch.value,  # only active when switch is on
        value=interesting_hh_id,              # initial household ID to display
        on_click=_draw_new_random_hh_id,      # fired on each button click
    )

    mo.hstack([new_graph_switch, new_graph_button])
    return (new_graph_button,)


@app.cell
def _(G, draw_hh_graph, new_graph_button, node_colours, node_labels):
    """
    Description: Render the currently selected household's activity graph as a Matplotlib
    figure using geographic coordinates (``use_coords=True``), coloured by activity purpose.
    The figure is also used as the base axes for the optional map overlay in the next cell.

    Input:
      - G (nx.MultiDiGraph): the NetworkX graph for the selected household.
      - draw_hh_graph (callable): the plotting helper.
      - new_graph_button (mo.ui.button): provides the current ``hh_id`` as ``.value``.
      - node_colours (dict): purpose-to-colour mapping from the preparation cell.
      - node_labels (dict): purpose-to-short-label mapping.

    Output:
      - ax (matplotlib.axes.Axes): the axes of the rendered figure; exported so the
        basemap cell can overlay a web tile background on it.
    """
    fig, ax = draw_hh_graph(
        G,
        new_graph_button.value,
        node_colours=node_colours,
        node_labels=node_labels,
        use_coords=True,
    )

    fig
    return (ax,)


@app.cell
def _(mo, new_graph_button):
    """
    Description: Create a run-button that triggers the expensive basemap rendering for the
    currently selected household. The label shows the current household ID so the user knows
    which household will be mapped.

    Input:
      - mo (module): Marimo.
      - new_graph_button (mo.ui.button): provides the current ``hh_id`` as ``.value``
        (used for the button label).

    Output:
      - generate_map_toggle (mo.ui.run_button): the button that gates the map-rendering cell.
    """
    generate_map_toggle = mo.ui.run_button(label=f"Show household {new_graph_button.value} on map")
    generate_map_toggle
    return (generate_map_toggle,)


@app.cell
def _(ax, cly, generate_map_toggle, gpd, hh_graph, mo):
    """
    Description: Optionally render the currently selected household's visited locations
    on a geographic basemap. This cell only executes when the user clicks the
    "Show household on map" button (guarded by ``mo.stop``). It overlays GeoDataFrame
    points for each visited location onto the Matplotlib axes produced by the graph-drawing
    cell, then adds a web tile background using contextily.

    Input:
      - ax (matplotlib.axes.Axes): the axes from the household graph drawing cell; the
        basemap is added to this same axes object.
      - cly (module): contextily, for fetching and adding web tile basemaps.
      - generate_map_toggle (mo.ui.run_button): when ``.value`` is True the map is drawn;
        otherwise this cell exits early via ``mo.stop``.
      - gpd (module): GeoPandas, for constructing the GeoDataFrame from node lat/lon.
      - hh_graph (ActivityGraph subobject): provides ``node_df`` with lat/lon columns for
        each visited location in the currently selected household.
      - mo (module): Marimo (for ``mo.stop``).

    Output:
      - (none): adds the basemap and location points to the existing matplotlib axes.
    """
    def _draw_on_map(ax, nodelist_df):
        """
        Description:
            Plots the household graph's visited locations as points on a geographic basemap.
            Converts the node DataFrame (which contains lat/lon columns) to a GeoDataFrame
            and overlays it on a web tile basemap using contextily.

        Input:
          - ax (matplotlib.axes.Axes): the matplotlib axes on which to draw the activity-graph
                diagram (the geographic points are overlaid on the same axes).
          - nodelist_df (pl.DataFrame): a Polars DataFrame with at least "lon" and "lat"
                columns giving the geographic coordinates of each visited location.

        Output:
          - ax (matplotlib.axes.Axes): the same axes object after adding the basemap layer.
        """
        # Convert the Polars nodelist to a GeoDataFrame by creating Point geometries from
        # the lon/lat columns.  CRS EPSG:4326 is standard WGS-84 (lat/lon degrees).
        geo = gpd.GeoDataFrame(
            nodelist_df,
            geometry=gpd.points_from_xy(nodelist_df["lon"], nodelist_df["lat"]),
            crs="EPSG:4326",
        )
        # Plot the visited locations as dots on top of the existing axes
        ax = geo.plot(ax=ax)
        # Add a web basemap (e.g. OpenStreetMap tiles) aligned to the data CRS
        cly.add_basemap(ax, crs=geo.crs.to_string(), attribution=False)
        return ax

    # Guard: only render the expensive map if the user has clicked the toggle button
    mo.stop(not generate_map_toggle.value)

    _draw_on_map(ax, hh_graph.node_df)
    return


@app.cell
def _(G):
    """
    Description: Print all node attribute dictionaries for the currently selected
    household's activity graph. Useful for debugging what metadata is stored per node
    (e.g. location ID, activity purposes, coordinates).

    Input:
      - G (nx.MultiDiGraph): the NetworkX graph for the selected household.

    Output:
      - (none): prints each ``(node_id, attribute_dict)`` pair to the notebook output.
    """
    for _n in G.nodes(data=True):
        print(_n)
    return


@app.cell
def _(G):
    """
    Description: Print all edge attribute dictionaries for the currently selected
    household's activity graph. Useful for debugging trip metadata stored per edge
    (e.g. mode, duration, distance, person ID).

    Input:
      - G (nx.MultiDiGraph): the NetworkX graph for the selected household.

    Output:
      - (none): prints each ``(origin, destination, attribute_dict)`` triple to the output.
    """
    for _e in G.edges(data=True):
        print(_e)
    return


if __name__ == "__main__":
    app.run()
