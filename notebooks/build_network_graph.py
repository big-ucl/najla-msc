import marimo

__generated_with = "0.18.3"
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
def _(gva_inputs):
    _gdf = gpd.read_file(
        "data/external/boundaries/swissboundaries3d_2025-04_2056_5728.shp/swissBOUNDARIES3D_1_5_TLM_HOHEITSGEBIET.shp"
    )
    _gdf = _gdf[_gdf["KANTONSNUM"] == 25].geometry.to_crs("EPSG:4326").union_all()
    gva_inputs.localities_gdf.to_crs("EPSG:4326").within(_gdf)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - Avusy
    - Bardonnex
    - Carouge (GE)
    - Collex-Bossy
    - Corsier (GE)
    - Lancy
    - Perly-Certoux
    - Pregny-Chambésy
    """)
    return


@app.cell
def _():
    from activitygraphs.data.geneva import load_files, build_geneva_data

    gva_inputs = load_files(cfg.data, project_root)
    gva_data = build_geneva_data(gva_inputs)
    return gva_data, gva_inputs


@app.cell
def _():
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
    from activitygraphs.network import (
        explore_pt_edges_by_mode,
        explore_transfer_edges,
        explore_locations_by_type,
        add_legend_pane_to_map,
    )
    from activitygraphs.network import ROUTE_MODE_COLOUR_MAP, LOCATION_TYPE_COLOR_MAP

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
def _(subsectors):
    from activitygraphs.network import build_planar_edges

    subsector_walk_edge_df = build_planar_edges(subsectors, travel_time_f=0.0)
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
    from activitygraphs.network import explore_walk_edges

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
    mo.md(r"""
    TODO:
    - add a travel time calculation method
    - start creating a PyG graph: heterogeneous graph?
    """)
    return


@app.cell
def _(subsector_walk_edge_df):
    import networkx as nx

    G = nx.from_pandas_edgelist(subsector_walk_edge_df.to_pandas(), source="orig_loc_id", target="dest_loc_id")
    set(nx.greedy_color(G, "saturation_largest_first").values())
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    return


@app.cell
def _(gva_data, subsector_walk_edge_df):
    walk_loc_ids = pl.concat([
        subsector_walk_edge_df.select(loc_id="orig_loc_id"),
        subsector_walk_edge_df.select(loc_id="dest_loc_id"),
    ]).unique().sort("loc_id").with_row_index()
    walk_locations_df = (
        walk_loc_ids.join(gva_data.locations_df, on="loc_id")
        .sort("loc_id")
        .with_columns(coords=pl.format("{},{}", pl.col("lon"), pl.col("lat")))
    )
    walk_locations_df
    return (walk_locations_df,)


@app.cell
def _(walk_locations_df):
    import requests
    from pypolyline.cutil import encode_coordinates

    mode = "foot"
    coordinates = walk_locations_df["coords"][:10].str.join("\n").item()
    polyline = encode_coordinates(list(walk_locations_df.select("lon", "lat").iter_rows()), 5).decode()
    osrm = f"http://127.0.0.1:5000/table/v1/driving/polyline({polyline})"

    response = requests.get(osrm)
    return (response,)


@app.cell
def _(response):
    import numpy as np

    if response.ok and (json := response.json())["code"] == "Ok":
        durations = np.array(json["durations"]) / 60
        print(durations)
    return durations, json


@app.cell
def _(json):
    json.keys()
    return


@app.cell
def _(durations, subsector_walk_edge_df, walk_locations_df):
    edge_indices = subsector_walk_edge_df.join(walk_locations_df.select(orig_index="index", orig_loc_id="loc_id"), on="orig_loc_id").join(walk_locations_df.select(dest_index="index", dest_loc_id="loc_id"), on="dest_loc_id")
    edge_indices_np = edge_indices.select("orig_index", "dest_index").to_numpy()

    travel_times = durations[edge_indices_np[:, 0], edge_indices_np[:, 1]]
    subsector_walk_edge_df.with_columns(travel_time_min=travel_times)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
