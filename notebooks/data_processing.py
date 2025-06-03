import marimo

__generated_with = "0.13.11"
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
    from data.ltds import LTDS_PURPOSES, LTDS_LAND_USES

    purposes_mapping = LTDS_PURPOSES
    land_uses_mapping = LTDS_LAND_USES
    return


@app.cell
def _():
    mo.md(r"""# Data loading and processing""")
    return


@app.cell
def _():
    from data.ltds import read_raw_data

    raw_household_df, raw_person_df, raw_trip_df = read_raw_data(cfg)
    raw_household_df.head()
    return raw_household_df, raw_person_df, raw_trip_df


@app.cell
def _(raw_person_df):
    from dataprocessing import bng_to_lat_long

    df = raw_person_df.head(10)
    lat, lon = bng_to_lat_long(df, "pwsose", "pwsosn")
    lat, lon
    return


@app.cell
def _(raw_household_df, raw_person_df):
    from data.ltds import create_hh_person_df

    hh_person_df = create_hh_person_df(raw_person_df, raw_household_df)
    hh_person_df.head()
    return (hh_person_df,)


@app.cell
def _(raw_trip_df):
    from data.ltds import create_trip_df

    trip_df = create_trip_df(raw_trip_df)
    trip_df.head()
    return (trip_df,)


@app.cell
def _():
    mo.md("""# Graph Generation""")
    return


@app.cell
def _():
    from dataprocessing import Purpose, LandUse

    def generate_node_attribute_df(
        single_hh_person_df: pl.DataFrame, single_trip_df: pl.DataFrame
    ):
        schema_df = pl.DataFrame(
            schema={
                "pid": pl.String,
                "is_home": pl.Boolean,
                "is_work": pl.Boolean,
                "land_use": LandUse.polars_enum(),
                "purpose": Purpose.polars_enum(),
                "lat": pl.Float64(),
                "lon": pl.Float64(),
            }
        )

        origin_trip_nodes = single_trip_df.select(
            pid="loc_origin_pid",
            is_home=False,
            is_work=False,
            land_use="land_use",
            purpose="purpose",
            lat="loc_origin_lat",
            lon="loc_origin_lon",
        ).unique()

        dest_trip_nodes = single_trip_df.select(
            pid="loc_dest_pid",
            is_home=False,
            is_work=False,
            land_use=None,
            purpose="purpose_dest",
            lat="loc_destination_lat",
            lon="loc_destination_lon",
        ).unique()

        home_nodes = single_hh_person_df.select(
            pid="loc_home_pid",
            is_home=True,
            is_work=False,
            land_use=None,
            purpose=pl.lit(Purpose.HOME).cast(Purpose.polars_enum()),
            lat="loc_home_lat",
            lon="loc_home_lon",
        ).unique()

        work_nodes = (
            single_hh_person_df.filter(pl.col("loc_work_pid") != "-1")
            .select(
                pid="loc_work_pid",
                is_home=False,
                is_work=True,
                land_use=None,
                purpose=pl.lit(Purpose.WORK).cast(Purpose.polars_enum()),
                lat="loc_work_lat",
                lon="loc_work_lon",
            )
            .unique()
        )

        nodes = pl.concat(
            [schema_df, origin_trip_nodes, dest_trip_nodes, home_nodes, work_nodes]
        )

        return nodes.group_by("pid").agg(
            pl.col("is_home").any(),
            pl.col("is_work").any(),
            pl.col("land_use").unique().drop_nulls().alias("land_uses"),
            pl.col("purpose").unique().drop_nulls().alias("purposes"),
            pl.col("lat").drop_nulls().first().alias("lat"),
            pl.col("lon").drop_nulls().first().alias("lon"),
        )
    return (generate_node_attribute_df,)


@app.cell
def _(generate_node_attribute_df):
    def generate_hh_node_and_edgelist(
        hh_id: str, hh_person_df: pl.DataFrame, trip_df: pl.DataFrame
    ):
        single_hh_person_df = hh_person_df.filter(pl.col("hh_id") == hh_id)
        single_trip_df = trip_df.filter(pl.col("hh_id") == hh_id)

        node_attribute_df = generate_node_attribute_df(
            single_hh_person_df, single_trip_df
        )

        return node_attribute_df, single_trip_df


    def generate_hh_graph(nodelist_df: pl.DataFrame, edgelist_df: pl.DataFrame):
        G = nx.from_pandas_edgelist(
            edgelist_df,
            source="loc_origin_pid",
            target="loc_dest_pid",
            edge_key="trip_id",
            create_using=nx.MultiDiGraph,
            edge_attr=[
                "person_id",
                "mode",
                "duration",
                "distance",
                "start_time",
                "end_time",
            ],
        )

        node_attribute_dict = nodelist_df.rows_by_key(
            key="pid", named=True, unique=True
        )
        nx.set_node_attributes(G, node_attribute_dict)

        return G
    return generate_hh_graph, generate_hh_node_and_edgelist


@app.cell
def _():
    interesting_hh_id = "12109151"
    new_graph_switch = mo.ui.switch(label="Use sampled graph")
    return interesting_hh_id, new_graph_switch


@app.cell
def _(
    generate_hh_graph,
    generate_hh_node_and_edgelist,
    hh_person_df,
    new_graph_button,
    trip_df,
):
    from plotting import (
        draw_hh_graph,
        line_styles_by_key,
        node_colours_by_purpose,
        node_short_labels_by_purpose,
    )

    _hh_id = new_graph_button.value

    nodelist_df, edgelist_df = generate_hh_node_and_edgelist(
        _hh_id, hh_person_df, trip_df
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
def _(hh_person_df, interesting_hh_id, new_graph_switch):
    def _draw_new_random_hh_id(value: str) -> str:
        return hh_person_df.select("hh_id").unique().sample(1)[0, "hh_id"]


    new_graph_button = mo.ui.button(
        label="Click to sample new graph from data set ",
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
    generate_map_toggle = mo.ui.run_button(label=f"Show household {new_graph_button.value} on map")
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
