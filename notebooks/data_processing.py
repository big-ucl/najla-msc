import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md("# Data processing")
    return (mo,)


@app.cell
def _():
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
    project_root = mo.notebook_dir().parent
    cfg = load_config(project_root)
    return (cfg,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # Excel to parquet conversion

    Does not need to be run unless converting from LTDS Access `.xslx` exports
    """
    )
    return


@app.cell
def _(mo):
    run_button = mo.ui.run_button(kind="warn", label="Run excel to parquet conversion")
    run_button
    return (run_button,)


@app.cell
def _(Path, cfg, mo, run_button):
    from dataprocessing import convert_excel_to_parquet

    mo.stop(not run_button.value, mo.md("Click button above to run conversion"))

    _files = (Path(s) for s in cfg.files.values())
    convert_excel_to_parquet(cfg.data.paths.raw, *_files)
    return


@app.cell
def _(mo):
    mo.md(r"""# Data loading and processing""")
    return


@app.cell
def _(mo):
    reprocess_button = mo.ui.run_button(label="Reprocess raw data")
    reprocess_button
    return (reprocess_button,)


@app.cell
def _(cfg, mo, reprocess_button):
    import dataprocessing as dp

    from dataprocessing import ActivityDataset
    from data.ltds import read_and_process_ltds

    _dataset_name = cfg.data.name
    _dataset_path = cfg.data.paths.act_dataset

    if (ActivityDataset.exists_on_disk(_dataset_path, _dataset_name) and not reprocess_button.value):
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
    dataset.hh_person_df.head()
    return


@app.cell
def _(dataset):
    dataset.trip_df.head()
    return


@app.cell
def _(dataset):
    dataset.location_df.head()
    return


@app.cell
def _(mo):
    mo.md("""# Graph Generation""")
    return


@app.cell
def _(dataset):
    from graphs import ActivityGraph

    graph = ActivityGraph.from_dataset(dataset)

    print(f"Loaded {graph} with {graph.n_subgraphs} subgraphs")
    return (graph,)


@app.cell
def _(graph, new_graph_button):
    from plotting import (
        draw_hh_graph,
        line_styles_by_key,
        node_colours_by_purpose,
        node_short_labels_by_purpose,
    )

    _hh_id = new_graph_button.value

    G = graph.to_nx(_hh_id)
    hh_graph = graph.hh_graph(_hh_id)
    line_styles = line_styles_by_key(G, key="person_id")
    node_colours = node_colours_by_purpose(G)
    node_labels = node_short_labels_by_purpose(G)
    return G, draw_hh_graph, hh_graph, line_styles, node_colours, node_labels


@app.cell
def _(mo):
    interesting_hh_id = "12109151"
    new_graph_switch = mo.ui.switch(label="Use sampled graph")
    return interesting_hh_id, new_graph_switch


@app.cell
def _(mo):
    mo.md("""# Visualisation""")
    return


@app.cell
def _(dataset, interesting_hh_id, mo, new_graph_switch):
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
def _(mo, new_graph_button):
    generate_map_toggle = mo.ui.run_button(
        label=f"Show household {new_graph_button.value} on map"
    )
    generate_map_toggle
    return (generate_map_toggle,)


@app.cell
def _(ax, cly, generate_map_toggle, gpd, hh_graph, mo):
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

    _draw_on_map(ax, hh_graph.node_df)
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


if __name__ == "__main__":
    app.run()
