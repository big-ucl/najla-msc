import marimo

__generated_with = "0.19.2"
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


@app.cell(hide_code=True)
def _():
    from activitygraphs.data.geneva import GenevaData

    gva_data = GenevaData.load(cfg.data, project_root)

    mo.accordion({
        "Table: Raw journeys": gva_data.inputs.raw_journeys_df,
        "Table: User journeys (cleaned)": gva_data.user_journeys_df,
    })
    return (gva_data,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Network building
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The Network is a heterogeneous graph organised into distinct layers. Each layer has its own set of nodes and edges between them. We have the follwing layers:

    | Layer | Type | Edge travel time | Edge key | Node key | Extra edge features | Description |
    | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
    | `public_transport`| PT | Avg. route travel time | (`orig_loc_id`, `route_id`, `dest_loc_id`) | (`loc_id`, `route_id`) | Avg. headway, First & last departure time | See note below
    | `subsectors`| Planar | Walking time | (`orig_loc_id`, `dest_loc_id`) | `loc_id` | - | Walking time is computed between (rook) adjacent subsectors, from their centroids
    | `municipality_geneva` | Planar | - | - | `loc_id` | - | Municipalities in the canton of Geneva. No walking between municipalities, must go through subsectors
    | `municipality_swiss` | Planar | - | - | `loc_id` | - | Swiss municipalities outside the canton of Geneva. No walking between municipalities
    | `municipality_french` | Planar | - | - | `loc_id` | - | French municipalities. No walking between municipalities
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The PT transport layer is organised as follows:
    - Each PT stop `loc_id` has one transfer node `(loc_id, `transfer_route`)` and one node `(loc_id, route_id)` per route `route_id` stopping at the stop.
    - Route edges (`layer.pt_edge_df`): Nodes connect to each other if a route connects the two locations, with a `(orig_loc_id, route_id, dest_loc_id)` edge
    - Internal transfer edges (`layer.transfer_edge_df`): Nodes at the same PT stop all connect to the central transfer node. Travel time is either the default 2 mins or the one specified in the GTFS file.
    - External transfer edges (`layer.transfer_edge_df`): Walking transfers between two PT stops are connected to each other's transfer nodes. Travel time is specified in the GTFS.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Additionally the layers are connected with each other through link edges.

    | Lower layer | Upper layer | Edge travel time | Description |
    | ----- | ----- | ----- | ----- |
    | `subsector` | `public_transport` | 0 | Connects PT stops contained within the subsector boundaries
    | `municipality_swiss` | `public_transport` | 0 | Connects PT stops contained within the municipality boundaries
    | `municipality_french` | `public_transport` | 0 | Connects PT stops contained within the municipality boundaries
    | `municipality_geneva` | `subsector` | 0 | Connects subsectors whose boundary intersects with the municipality boundaries
    """)
    return


@app.cell
def _(gva_data):
    from activitygraphs.base import Mode
    from activitygraphs.routing import TravelTimeCalculator, OSRMRouter

    walk_router = OSRMRouter("http://127.0.0.1:5000", Mode.WALK).with_cache()
    walk_travel_time_f = TravelTimeCalculator(gva_data.locations_gdf, walk_router)
    return (walk_travel_time_f,)


@app.cell
def _(gva_data, walk_travel_time_f):
    from activitygraphs.network import Network
    from activitygraphs.data.gtfs import build_pt_layer_edges
    from activitygraphs.base import PTNodeType


    def build_gva_pt_layer(locations_gdf: gpd.GeoDataFrame, pt_node_type: PTNodeType):
        return build_pt_layer_edges(locations_gdf, gva_data.gtfs, pt_node_type, drop_null_headways=True)


    def build_gva_network(name: str, pt_node_type: PTNodeType = PTNodeType.ONE_PER_ROUTE):
        network = (
            Network
            .empty_network(gva_data)
            .add_pt_layer(
                "public_transport",
                loc_ids="public_transport",
                pt_layer_builder=build_gva_pt_layer,
                pt_node_type=pt_node_type,
            )
            .add_planar_layer("subsector", loc_ids="subsector", travel_time_f=walk_travel_time_f)
            .add_planar_layer("municipality_geneva", loc_ids="municipality_geneva")
            .add_planar_layer("municipality_swiss", loc_ids="municipality_swiss")
            .add_planar_layer("municipality_french", loc_ids="municipality_french")
            .add_na_layer(separate_in_out_nodes=True)
            .connect_layers("subsector", "public_transport", 0.0)
            .connect_layers("municipality_swiss", "public_transport", 0.0)
            .connect_layers("municipality_french", "public_transport", 0.0)
            .connect_layers("municipality_geneva", "subsector", 0.0, mode="centroid_nearest")
            .connect_na_layer(
                ["public_transport", "subsector", "municipality_geneva", "municipality_swiss", "municipality_french"],
                10000.0,
            )
        )

        network.save(cfg.data, project_root, name=name)
        return network

    def load_gva_network(name: str, pt_node_type: PTNodeType):
        if Network.exists_on_disk(cfg.data, project_root, name):
            return Network.load(cfg.data, project_root, name)

        return build_gva_network(name, pt_node_type)


    network_expanded_routes = load_gva_network("routes", PTNodeType.ONE_PER_ROUTE)
    network_collapsed_routes = load_gva_network("stops", PTNodeType.ONE_PER_STOP)

    network = network_collapsed_routes
    network
    return (network,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Public transport layer
    """)
    return


@app.cell(hide_code=True)
def _(network):
    mo.accordion({
        "Table: PT Route edges": network["public_transport"].pt_edge_df,
        "Table: PT Transfer edges": network["public_transport"].transfer_edge_df,
    })
    return


@app.cell(hide_code=True)
def _(gva_data, network):
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
    _m = explore_pt_edges_by_mode(network["public_transport"].pt_edge_df, gva_data.locations_gdf, m=_m)
    _m = explore_transfer_edges(network["public_transport"].transfer_edge_df, gva_data.locations_gdf, m=_m)
    _m = explore_locations_by_type(gva_data.locations_gdf, types=["public_transport", "na"], m=_m)
    add_legend_pane_to_map(_m, legends)

    mo.vstack([
        mo.md(
            "PT edges (bus / tram / train / boat) between stops, as well as (official) walking transfers between stops. Geneva subsectors and french municipalities in the background."
        ),
        _m,
    ])
    return (explore_locations_by_type,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Walking layer

    Add walking links between adjacent Geneva subsectors
    """)
    return


@app.cell(hide_code=True)
def _(network):
    mo.accordion({"Subsector walking links": network["subsector"].edge_list})
    return


@app.cell(hide_code=True)
def _(explore_locations_by_type, gva_data, network):
    from activitygraphs.mapping import explore_walk_edges

    _m = explore_locations_by_type(gva_data.locations_gdf, types=["subsector"], as_points=False)
    _m = explore_walk_edges(network["subsector"].edge_list, gva_data.locations_gdf, m=_m)
    _m = explore_locations_by_type(gva_data.locations_gdf, types=["subsector"], m=_m)

    mo.vstack([mo.md("Walking edges between subsectors"), _m])
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


@app.cell(hide_code=True)
def _(network):
    mo.accordion({
        "Table: Links between subsectors and PT stops": network.get_links("subsector", "public_transport"),
        "Table: Links between Swiss municipalities and PT stops": network.get_links(
            "municipality_swiss", "public_transport"
        ),
        "Table: Links between French municipalities and PT stops": network.get_links(
            "municipality_french", "public_transport"
        ),
        "Table: Links between Geneva municipalities and subsectors": network.get_links("municipality_geneva", "subsector"),
    })
    return


@app.cell(hide_code=True)
def _(explore_locations_by_type, explore_walk_edges, gva_data, network):
    _m = explore_locations_by_type(gva_data.locations_gdf, types=["subsector"], as_points=False)
    _m = explore_walk_edges(network.get_links("subsector", "public_transport"), gva_data.locations_gdf, m=_m)
    _m = explore_locations_by_type(gva_data.locations_gdf, types=["subsector", "public_transport"], m=_m)

    mo.vstack([mo.md("Links between subsectors and PT stops:"), _m])
    return


@app.cell(hide_code=True)
def _(explore_locations_by_type, explore_walk_edges, gva_data, network):
    _muni_links = pl.concat([
        network.get_links("municipality_swiss", "public_transport"),
        network.get_links("municipality_french", "public_transport"),
    ])

    _m = explore_locations_by_type(
        gva_data.locations_gdf, types=["municipality_swiss", "municipality_french"], as_points=False
    )
    _m = explore_walk_edges(_muni_links, gva_data.locations_gdf, m=_m)
    _m = explore_locations_by_type(
        gva_data.locations_gdf, types=["public_transport", "municipality_swiss", "municipality_french"], m=_m
    )
    mo.vstack([mo.md("Links between municipalities, both French and Swiss (but not from Geneva), and PT stops"), _m])
    return


@app.cell(hide_code=True)
def _(explore_locations_by_type, explore_walk_edges, gva_data, network):
    _m = explore_locations_by_type(gva_data.locations_gdf, types=["municipality_geneva", "subsector"], as_points=False)
    _m = explore_walk_edges(network.get_links("municipality_geneva", "subsector"), gva_data.locations_gdf, m=_m)
    _m = explore_locations_by_type(gva_data.locations_gdf, types=["subsector", "municipality_geneva"], m=_m)

    mo.vstack([mo.md("Links between municipalities and subsectors"), _m])
    return


@app.cell(hide_code=True)
def _():
    na_dropdown = mo.ui.dropdown(
        ["public_transport", "subsector", "municipality_geneva", "municipality_swiss", "municipality_french"],
        value="subsector",
    )

    mo.hstack([mo.md("Links between NA and "), na_dropdown], justify="start")
    return (na_dropdown,)


@app.cell
def _(explore_locations_by_type, explore_walk_edges, na_dropdown, network):
    _m = explore_walk_edges(
        network.get_links("na", na_dropdown.value),
        network.locations_gdf,
    )
    _m = explore_locations_by_type(network.locations_gdf, types=["na", na_dropdown.value], as_points=True, m=_m)

    _m
    return


if __name__ == "__main__":
    app.run()
