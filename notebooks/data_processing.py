import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells

    # Import modules
    import networkx as nx
    import marimo as mo
    import polars as pl
    from pathlib import Path
    from config import load_config
    import numpy as np
    import random
    import contextily as cly
    import geopandas as gpd


    # Load configuration from files
    project_root = mo.notebook_dir().parent

    cfg = load_config(project_root)
    data_path = project_root / cfg.paths.data_raw_ltds

    print(f"Configuration loaded: {cfg}")

    # Set random seeds
    np.random.seed(42)
    random.seed(42)


@app.cell
def _():
    mo.md(
        r"""
    # Excel to parquet conversion

    Does not need to be run unless converting from LTDS Access `.xslx` exports
    """
    )
    return


@app.cell
def _():
    run_button = mo.ui.run_button(
        kind="warn", label="Run excel to parquet conversion"
    )
    run_button
    return (run_button,)


@app.cell
def _(run_button):
    from dataprocessing import convert_excel_to_parquet

    mo.stop(not run_button.value, mo.md("Click button above to run conversion"))

    _files = (Path(s) for s in cfg.files.values())
    convert_excel_to_parquet(data_path, *_files)
    return


@app.cell
def _():
    mo.md(r"""# Data loading and processing""")
    return


@app.cell
def _():
    reprocess_button = mo.ui.run_button(label="Reprocess raw data")
    reprocess_button
    return (reprocess_button,)


@app.cell
def _(reprocess_button):
    import dataprocessing as dp

    from dataprocessing import ActivityDataset
    from data.ltds import read_and_parse_ltds

    _ltds_name = "LTDS"
    _dataset_path = Path(cfg.paths.data_processed)

    if (
        ActivityDataset.exists_on_disk(_dataset_path, _ltds_name)
        and not reprocess_button.value
    ):
        dataset = ActivityDataset.load(_dataset_path, _ltds_name)
        print(f"Loaded `{dataset.name}` dataset from disk.")
    else:
        dataset = read_and_parse_ltds(cfg, name=_ltds_name)
        dataset.save(_dataset_path)
        print(f"Read, processed and saved `{dataset.name}` dataset from raw data")

    dataset.name
    return dataset, dp


@app.cell
def _(dataset):
    dataset.hh_person_df.head()
    return


@app.cell
def _(dataset):
    dataset.trip_df.head()
    return


@app.cell
def _():
    mo.md("""# Graph Generation""")
    return


@app.cell
def _():
    from graphs import generate_hh_node_and_edgelist, generate_hh_graph, generate_node_attribute_df
    return (
        generate_hh_graph,
        generate_hh_node_and_edgelist,
        generate_node_attribute_df,
    )


@app.cell
def _(dataset, generate_node_attribute_df):
    ndf = generate_node_attribute_df(dataset.hh_person_df, dataset.trip_df)
    ndf.head()
    return


@app.cell
def _(dp):
    dp.Purpose(2097280)
    return


@app.cell
def _():
    interesting_hh_id = "12109151"
    new_graph_switch = mo.ui.switch(label="Use sampled graph")
    return interesting_hh_id, new_graph_switch


@app.cell
def _(
    dataset,
    generate_hh_graph,
    generate_hh_node_and_edgelist,
    new_graph_button,
):
    from plotting import (
        draw_hh_graph,
        line_styles_by_key,
        node_colours_by_purpose,
        node_short_labels_by_purpose,
    )

    _hh_id = new_graph_button.value

    nodelist_df, edgelist_df = generate_hh_node_and_edgelist(
        _hh_id, dataset.hh_person_df, dataset.trip_df
    )

    G = generate_hh_graph(nodelist_df, edgelist_df)
    line_styles = line_styles_by_key(G, key="person_id")
    node_colours = node_colours_by_purpose(G)
    node_labels = node_short_labels_by_purpose(G)
    return (
        G,
        draw_hh_graph,
        line_styles,
        node_colours,
        node_labels,
        nodelist_df,
    )


@app.cell
def _():
    mo.md("""# Visualisation""")
    return


@app.cell
def _(dataset, interesting_hh_id, new_graph_switch):
    def _draw_new_random_hh_id(value: str) -> str:
        return dataset.hh_person_df.select("hh_id").unique().sample(1)[0, "hh_id"]


    new_graph_button = mo.ui.button(
        label="Sample new graph from dataset ",
        disabled=not new_graph_switch.value,
        value=interesting_hh_id,
        on_click=_draw_new_random_hh_id,
    )

    mo.hstack([new_graph_switch, new_graph_button])
    return (new_graph_button,)


@app.cell
def _(
    G,
    draw_hh_graph,
    line_styles,
    new_graph_button,
    node_colours,
    node_labels,
):
    fig, ax = draw_hh_graph(
        G,
        new_graph_button.value,
        line_styles=line_styles,
        node_colours=node_colours,
        node_labels=node_labels,
        use_coords=True,
    )

    fig
    return (ax,)


@app.cell
def _(new_graph_button):
    generate_map_toggle = mo.ui.run_button(
        label=f"Show household {new_graph_button.value} on map"
    )
    generate_map_toggle
    return (generate_map_toggle,)


@app.cell
def _(ax, generate_map_toggle, nodelist_df):
    def _draw_on_map(ax, nodelist_df):
        geo = gpd.GeoDataFrame(
            nodelist_df,
            geometry=gpd.points_from_xy(nodelist_df["lon"], nodelist_df["lat"]),
            crs="EPSG:4326",
        )
        ax = geo.plot(ax=ax)
        cly.add_basemap(ax, crs=geo.crs.to_string(), attribution=False)
        return ax


    mo.stop(not generate_map_toggle.value)

    _draw_on_map(ax, nodelist_df)
    return


@app.cell
def _(G):
    for _n in G.nodes(data=True):
        print(_n)
    return


@app.cell
def _(G):
    for _e in G.edges(data=True):
        print(_e)
    return


@app.cell
def _():
    mo.md(
        r"""
    TODO Notes:

     - Draw graph in a more interesting manner
     - Handle case with unknown location (negative weights)
     - Add physical location information to graph
    """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
