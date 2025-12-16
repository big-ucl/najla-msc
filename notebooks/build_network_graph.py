import marimo

__generated_with = "0.18.4"
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
    return gva_data, gva_inputs


@app.cell
def _(gva_inputs):
    gva_inputs.raw_journeys_df
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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
 
    """)
    return


@app.cell
def _(gva_data):
    from activitygraphs.data.gtfs import build_pt_network_edges

    pt_edge_df, transfer_edge_df = build_pt_network_edges(gva_data.locations_df, gva_data.gtfs, drop_null_headways=True)
    return pt_edge_df, transfer_edge_df


@app.cell
def _(pt_edge_df):
    pt_edge_df
    return


@app.cell
def _(transfer_edge_df):
    transfer_edge_df
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    PT edges (bus / tram / train / boat) between stops, as well as (official) walking transfers between stops. Geneva subsectors and french municipalities in the background.
    """)
    return


@app.cell
def _(gva_data, pt_edge_df, transfer_edge_df):
    from activitygraphs.mapping import (
        explore_pt_edges_by_mode,
        explore_transfer_edges,
        explore_locations_by_type,
        add_legend_pane_to_map,
        ROUTE_MODE_COLOUR_MAP,
        LOCATION_TYPE_COLOR_MAP,
    )

    legends = {
        "Transport modes": ("line", ROUTE_MODE_COLOUR_MAP),
        "Location types": ("circle", LOCATION_TYPE_COLOR_MAP),
    }

    _m = explore_locations_by_type(
        gva_data.locations_gdf, types=["subsector", "municipality_swiss", "municipality_french"], as_points=False
    )
    _m = explore_pt_edges_by_mode(pt_edge_df, gva_data.locations_gdf, m=_m)
    _m = explore_transfer_edges(transfer_edge_df, gva_data.locations_gdf, m=_m)
    _m = explore_locations_by_type(gva_data.locations_gdf, types=["public_transport", "na"], m=_m)
    add_legend_pane_to_map(_m, legends)

    _m
    return (explore_locations_by_type,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Walking layer

    Add walking links between adjacent Geneva subsectors
    """)
    return


@app.cell
def _(gva_data):
    from activitygraphs.routing import TravelTimeCalculator, OSRMRouter
    from activitygraphs.base import Mode

    _walk_router = OSRMRouter("http://127.0.0.1:5000", Mode.WALK).with_cache()
    walk_travel_time_f = TravelTimeCalculator(gva_data.locations_gdf, _walk_router)
    return (walk_travel_time_f,)


@app.cell
def _(gva_data):
    municipality_types = ["municipality_french", "municipality_swiss", "municipality_geneva"]

    pt_stops = gva_data.locations_gdf[gva_data.locations_gdf["type"] == "public_transport"]
    subsectors = gva_data.locations_gdf[gva_data.locations_gdf["type"] == "subsector"]
    municipalities = gva_data.locations_gdf[gva_data.locations_gdf["type"].isin(municipality_types)]
    geneva_municipalities = municipalities[municipalities["type"] == "municipality_geneva"]
    non_geneva_municipalities = municipalities[municipalities["type"] != "municipality_geneva"]
    return (
        geneva_municipalities,
        non_geneva_municipalities,
        pt_stops,
        subsectors,
    )


@app.cell
def _(subsectors, walk_travel_time_f):
    from activitygraphs.network import build_planar_edges

    subsector_walk_edge_df = build_planar_edges(subsectors, travel_time_f=walk_travel_time_f)
    subsector_walk_edge_df
    return (subsector_walk_edge_df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Walking edges between subsectors
    """)
    return


@app.cell
def _(explore_locations_by_type, gva_data, subsector_walk_edge_df):
    from activitygraphs.mapping import explore_walk_edges

    _m = explore_locations_by_type(gva_data.locations_gdf, types=["subsector"], as_points=False)
    _m = explore_walk_edges(subsector_walk_edge_df, gva_data.locations_gdf, m=_m)
    _m = explore_locations_by_type(gva_data.locations_gdf, types=["subsector"], m=_m)
    _m
    return (explore_walk_edges,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Inter-layer links

    Create three kinds of edges between different layers:
      - Create edges between subsectors of canton Geneva and PT stops within them, linking the PT layer with the subsector layer.
      - Create edges between municipalities of canton Geneva and subsectors within them, linking the subsectors with the municipalities layer
      - Create edges between municipalities outside of Geneva (France and rest of Switzerland), and PT stops, linking the PT layer with (some of) the municipality layer
    """)
    return


@app.cell
def _(non_geneva_municipalities, pt_stops, subsectors):
    import pandas as pd
    from activitygraphs.network import build_layer_link_edges


    pt_subsector_link_edge_df = build_layer_link_edges(subsectors, pt_stops, 0.0)
    pt_municipality_link_edge_df = build_layer_link_edges(non_geneva_municipalities, pt_stops, 0.0)
    pt_link_edge_df = pl.concat([pt_subsector_link_edge_df, pt_municipality_link_edge_df])

    pt_link_edge_df
    return (
        build_layer_link_edges,
        pt_municipality_link_edge_df,
        pt_subsector_link_edge_df,
    )


@app.cell
def _(build_layer_link_edges, geneva_municipalities, subsectors):
    municipality_subsector_link_edge_df = build_layer_link_edges(
        geneva_municipalities, subsectors, 0.0, mode="centroid_nearest"
    )
    municipality_subsector_link_edge_df
    return (municipality_subsector_link_edge_df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Links between subsectors and PT stops:
    """)
    return


@app.cell
def _(
    explore_locations_by_type,
    explore_walk_edges,
    gva_data,
    pt_subsector_link_edge_df,
):
    _m = explore_locations_by_type(gva_data.locations_gdf, types=["subsector"], as_points=False)
    _m = explore_walk_edges(pt_subsector_link_edge_df, gva_data.locations_gdf, m=_m)
    _m = explore_locations_by_type(gva_data.locations_gdf, types=["subsector", "public_transport"], m=_m)
    _m
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Links between municipalities, both French and Swiss (but not from Geneva), and PT stops
    """)
    return


@app.cell
def _(
    explore_locations_by_type,
    explore_walk_edges,
    gva_data,
    pt_municipality_link_edge_df,
):
    _m = explore_locations_by_type(
        gva_data.locations_gdf, types=["municipality_swiss", "municipality_french"], as_points=False
    )
    _m = explore_walk_edges(pt_municipality_link_edge_df, gva_data.locations_gdf, m=_m)
    _m = explore_locations_by_type(
        gva_data.locations_gdf, types=["public_transport", "municipality_swiss", "municipality_french"], m=_m
    )
    _m
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Links between municipalities and subsectors
    """)
    return


@app.cell
def _(
    explore_locations_by_type,
    explore_walk_edges,
    gva_data,
    municipality_subsector_link_edge_df,
):
    _m = explore_locations_by_type(gva_data.locations_gdf, types=["municipality_geneva", "subsector"], as_points=False)
    _m = explore_walk_edges(municipality_subsector_link_edge_df, gva_data.locations_gdf, m=_m)
    _m = explore_locations_by_type(gva_data.locations_gdf, types=["subsector", "municipality_geneva"], m=_m)
    _m
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
