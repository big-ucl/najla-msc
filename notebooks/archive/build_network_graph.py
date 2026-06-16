"""
Module: build_network_graph.py

Description:
    A Marimo interactive notebook that constructs a multi-layer heterogeneous transport network
    graph for the Geneva (GVA) region.  The network is used downstream as the base graph on which
    individual activity schedules are overlaid.

    The graph contains the following layers:
      - public_transport : PT stops/routes from GTFS data (bus, tram, train, boat)
      - subsector        : Geneva canton subsector polygons connected by walking time
      - municipality_geneva : Geneva canton municipality polygons
      - municipality_swiss  : Swiss municipalities outside Geneva
      - municipality_french : French municipalities near Geneva

    Inter-layer link edges connect PT stops to the surrounding planar layers so that a routing
    algorithm can move from one layer to another.

    The resulting Network object is saved to disk and later loaded by build_pyg_dataset.py.

Dependencies:
    - activitygraphs library (GenevaData, Network, OSRMRouter, TravelTimeCalculator, etc.)
    - Marimo (reactive notebook framework)
    - geopandas, polars
"""

import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    import geopandas as gpd

    import polars as pl
    from pathlib import Path

    from activitygraphs.config import load_config

    project_root = Path(mo.notebook_dir().parent)
    cfg = load_config(project_root)


@app.cell(hide_code=True)
def _():
    """
    Description: Display a Marimo Markdown heading cell for the "Project setup and
    data loading" section of the notebook. Pure presentation; no computation.

    Input:
      - (none)

    Output:
      - (none): returns nothing; the mo.md call renders the heading in the notebook UI.
    """
    mo.md(r"""
    ## Project setup and data loading
    """)
    return


@app.cell(hide_code=True)
def _():
    """
    Description: Load the GenevaData object from disk (or build it if not yet cached) and
    display two data tables in an accordion widget: the raw GPS journey records and the
    cleaned user-journey records.

    Input:
      - (none): depends on ``cfg`` and ``project_root`` from the setup block.

    Output:
      - gva_data (GenevaData): the loaded Geneva dataset object; exported for use by later
        cells (GTFS data, location GeoDataFrame, user journeys, etc.).
    """
    from activitygraphs.data.geneva import GenevaData

    gva_data = GenevaData.load(cfg.data, project_root)

    mo.accordion({
        "Table: Raw journeys": gva_data.inputs.raw_journeys_df,
        "Table: User journeys (cleaned)": gva_data.user_journeys_df,
    })
    return (gva_data,)


@app.cell(hide_code=True)
def _():
    """
    Description: Display a Marimo Markdown heading cell for the "Network building" section.
    Pure presentation cell with no computation.

    Input:
      - (none)

    Output:
      - (none)
    """
    mo.md(r"""
    ## Network building
    """)
    return


@app.cell(hide_code=True)
def _():
    """
    Description: Display a Marimo Markdown table describing each network layer: its type
    (PT or Planar), edge travel-time semantics, edge/node key columns, any extra edge
    features, and a brief description. Pure presentation; no computation.

    Input:
      - (none)

    Output:
      - (none): renders the layer-description table in the notebook UI.
    """
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
    """
    Description: Display a Marimo Markdown explanation of the internal structure of the
    public-transport layer: how stops become nodes, how route edges and internal/external
    transfer edges are formed, and the travel-time conventions used for each edge type.
    Pure presentation; no computation.

    Input:
      - (none)

    Output:
      - (none): renders the PT-layer description text in the notebook UI.
    """
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
    """
    Description: Display a Marimo Markdown table describing the four types of inter-layer
    link edges that connect different network layers to one another, including which layer
    pairs are linked, the assumed travel time (zero for topological links), and a brief
    description of the spatial rule used to create each link. Pure presentation.

    Input:
      - (none)

    Output:
      - (none): renders the inter-layer link table in the notebook UI.
    """
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
    """
    Description: Initialise a walking travel-time calculator backed by a locally running
    OSRM routing server. The calculator is cached on disk so that repeated queries for
    the same origin-destination pair do not hit the server again.

    Input:
      - gva_data (GenevaData): provides ``locations_gdf`` (the lat/lon coordinates needed
        to geocode loc_id strings for OSRM routing requests).

    Output:
      - walk_travel_time_f (TravelTimeCalculator): callable that accepts two loc_id strings
        and returns the walking travel time in minutes.
    """
    from activitygraphs.base import Mode
    from activitygraphs.routing import TravelTimeCalculator, OSRMRouter

    # Connect to a locally-running OSRM routing server on port 5000.
    # The .with_cache() call wraps the router so that repeated requests for the same
    # origin-destination pair are served from an on-disk cache rather than hitting the
    # server again.
    walk_router = OSRMRouter("http://127.0.0.1:5000", Mode.WALK).with_cache()

    # TravelTimeCalculator wraps the router and provides a convenient callable that,
    # given two loc_id strings, returns the walking time in minutes.
    # gva_data.locations_gdf supplies the lat/lon coordinates needed to geocode the loc_ids.
    walk_travel_time_f = TravelTimeCalculator(gva_data.locations_gdf, walk_router)
    return (walk_travel_time_f,)


@app.cell
def _(gva_data, walk_travel_time_f):
    """
    Description: Define helper functions for building and loading the Geneva multi-layer
    network, then build (or load from disk) both the "routes" and "stops" network variants.
    The cell also sets ``network`` to the collapsed (stops) variant for downstream display.

    This cell defines three local helper functions (``build_gva_pt_layer``,
    ``build_gva_network``, ``load_gva_network``) before executing them. See each function's
    own docstring for details.

    Input:
      - gva_data (GenevaData): provides GTFS data and the locations GeoDataFrame.
      - walk_travel_time_f (TravelTimeCalculator): walking travel-time callable from the
        previous cell, used to populate walking edges in the subsector planar layer.

    Output:
      - network (Network): the collapsed (one-node-per-stop) version of the Geneva network,
        selected for downstream visualisation in this notebook.
    """
    from archive.network import Network
    from activitygraphs.data.gtfs import build_pt_layer_edges
    from activitygraphs.base import PTNodeType

    def build_gva_pt_layer(locations_gdf: gpd.GeoDataFrame, pt_node_type: PTNodeType):
        """
        Description:
            Builds the PT (public transport) layer edge tables from the GTFS feed for
            Geneva.  This is a thin wrapper around the generic build_pt_layer_edges
            function that binds the GVA-specific GTFS data and drops stops with unknown
            headways.

        Input:
          - locations_gdf (gpd.GeoDataFrame): GeoDataFrame of all locations in the network;
                used to match GTFS stop_ids to internal loc_ids.
          - pt_node_type (PTNodeType): controls whether each stop gets one node total
                (ONE_PER_STOP) or one node per route passing through the stop
                (ONE_PER_ROUTE).

        Output:
          - (tuple): a pair (pt_edge_df, transfer_edge_df) of Polars DataFrames
                describing route edges and transfer edges respectively.
        """
        return build_pt_layer_edges(locations_gdf, gva_data.gtfs, pt_node_type, drop_null_headways=True)

    def build_gva_network(name: str, pt_node_type: PTNodeType = PTNodeType.ONE_PER_ROUTE):
        """
        Description:
            Assembles and saves a full Geneva multi-layer network from scratch.  It adds
            every spatial layer (PT, subsectors, three municipality types, NA source/sink),
            then wires all the inter-layer link edges, and persists the result so it can be
            re-loaded without recomputation.

        Input:
          - name (str): a short label (e.g. "routes" or "stops") that becomes part of the
                on-disk directory name so different variants can coexist.
          - pt_node_type (PTNodeType): the node-type strategy for the PT layer.
                Defaults to ONE_PER_ROUTE.

        Output:
          - (Network): the fully assembled Network object (also saved to disk).
        """
        network = (
            Network
            .empty_network(gva_data)
            # ---- Add the public-transport layer ----------------------------------
            .add_pt_layer(
                "public_transport",
                loc_ids="public_transport",          # filter to PT locations only
                pt_layer_builder=build_gva_pt_layer, # function that builds edges
                pt_node_type=pt_node_type,
            )
            # ---- Add planar (polygon) layers with walking travel times -----------
            .add_planar_layer("subsector", loc_ids="subsector", travel_time_f=walk_travel_time_f)
            # Municipality layers have no walking edges between them
            .add_planar_layer("municipality_geneva", loc_ids="municipality_geneva")
            .add_planar_layer("municipality_swiss", loc_ids="municipality_swiss")
            .add_planar_layer("municipality_french", loc_ids="municipality_french")
            # ---- Add special NA (not-assigned) source and sink nodes -------------
            .add_na_layer(separate_in_out_nodes=True)
            # ---- Connect layers with inter-layer link edges ----------------------
            # PT stops inside a subsector are linked to that subsector
            .connect_layers("subsector", "public_transport", 0.0)
            # PT stops inside Swiss/French municipalities are linked to that municipality
            .connect_layers("municipality_swiss", "public_transport", 0.0)
            .connect_layers("municipality_french", "public_transport", 0.0)
            # Geneva municipalities are linked to nearest subsector centroid
            .connect_layers("municipality_geneva", "subsector", 0.0, mode="centroid_nearest")
            # NA source/sink linked to all layers within 10 km
            .connect_na_layer(
                ["public_transport", "subsector", "municipality_geneva", "municipality_swiss", "municipality_french"],
                10000.0,  # maximum link distance in metres
            )
        )

        # Persist to disk under cfg.data paths so future runs skip re-building
        network.save(cfg.data, project_root, name=name)
        return network

    def load_gva_network(name: str, pt_node_type: PTNodeType):
        """
        Description:
            Loads a previously built GVA network from disk if it exists, otherwise
            builds it from scratch and saves it.

        Input:
          - name (str): the label identifying which network variant to load/build.
          - pt_node_type (PTNodeType): PT node strategy used when building from scratch.

        Output:
          - (Network): the loaded or freshly-built Network object.
        """
        if Network.exists_on_disk(cfg.data, project_root, name):
            return Network.load(cfg.data, project_root, name)

        return build_gva_network(name, pt_node_type)

    # "routes" variant: one PyG node per (stop, route) pair — more detailed, larger graph
    network_expanded_routes = load_gva_network("routes", PTNodeType.ONE_PER_ROUTE)
    # "stops" variant: one PyG node per stop — simpler, smaller graph
    network_collapsed_routes = load_gva_network("stops", PTNodeType.ONE_PER_STOP)

    # Use the collapsed (stop-level) network for downstream visualisation in this notebook
    network = network_collapsed_routes
    network
    return (network,)


@app.cell(hide_code=True)
def _():
    """
    Description: Display a Marimo Markdown sub-heading for the "Public transport layer"
    section. Pure presentation; no computation.

    Input:
      - (none)

    Output:
      - (none): renders the section heading in the notebook UI.
    """
    mo.md(r"""
    ### Public transport layer
    """)
    return


@app.cell(hide_code=True)
def _(network):
    """
    Description: Display the public transport layer's route edges and transfer edges in
    accordion tables for inspection.

    Input:
      - network (Network): the loaded Geneva network object.

    Output:
      - (none): displays two tables in the notebook UI.
    """
    mo.accordion({
        "Table: PT Route edges": network["public_transport"].pt_edge_df,
        "Table: PT Transfer edges": network["public_transport"].transfer_edge_df,
    })
    return


@app.cell(hide_code=True)
def _(gva_data, network):
    """
    Description: Create an interactive Folium map showing:
      - Geneva subsectors and nearby municipalities as filled polygons.
      - PT route edges colour-coded by transport mode (bus, tram, train, boat).
      - Walking transfer edges between PT stops.
      - PT stop and NA node locations as points.
    Displayed with a mode/type legend panel for readability.

    Input:
      - gva_data (GenevaData): provides ``locations_gdf`` for geocoding.
      - network (Network): provides the PT edge tables and link tables.

    Output:
      - (none): renders the map and legend in the notebook UI.
    """
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
    """
    Description: Display a Marimo Markdown sub-heading for the "Walking layer" section,
    along with a brief note that this section covers adding walking links between adjacent
    Geneva subsectors. Pure presentation; no computation.

    Input:
      - (none)

    Output:
      - (none): renders the section heading and note in the notebook UI.
    """
    mo.md(r"""
    ### Walking layer

    Add walking links between adjacent Geneva subsectors
    """)
    return


@app.cell(hide_code=True)
def _(network):
    """
    Description: Display the subsector walking-link edge list in an accordion widget.

    Input:
      - network (Network): the loaded Geneva network.

    Output:
      - (none): renders the accordion table in the notebook UI.
    """
    mo.accordion({"Subsector walking links": network["subsector"].edge_list})
    return


@app.cell(hide_code=True)
def _(explore_locations_by_type, gva_data, network):
    """
    Description: Create an interactive map showing walking edges between adjacent Geneva
    subsectors, overlaid on the subsector polygons.

    Input:
      - explore_locations_by_type (callable): helper function imported in a previous cell.
      - gva_data (GenevaData): provides ``locations_gdf``.
      - network (Network): provides the subsector edge list.

    Output:
      - explore_walk_edges (callable): the imported walk-edge mapping helper; exported for
        use by later cells that draw other edge types.
    """
    from activitygraphs.mapping import explore_walk_edges

    _m = explore_locations_by_type(gva_data.locations_gdf, types=["subsector"], as_points=False)
    _m = explore_walk_edges(network["subsector"].edge_list, gva_data.locations_gdf, m=_m)
    _m = explore_locations_by_type(gva_data.locations_gdf, types=["subsector"], m=_m)

    mo.vstack([mo.md("Walking edges between subsectors"), _m])
    return (explore_walk_edges,)


@app.cell(hide_code=True)
def _():
    """
    Description: Display a Marimo Markdown sub-heading for the "Inter-layer links" section
    together with a bullet-list describing the three categories of inter-layer edges that
    are created: subsector-to-PT, municipality_geneva-to-subsector, and
    Swiss/French-municipality-to-PT. Pure presentation; no computation.

    Input:
      - (none)

    Output:
      - (none): renders the section heading and description in the notebook UI.
    """
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
    """
    Description: Display all four inter-layer link edge tables in accordion widgets:
      - subsector <-> PT stops
      - Swiss municipalities <-> PT stops
      - French municipalities <-> PT stops
      - Geneva municipalities <-> subsectors

    Input:
      - network (Network): the loaded Geneva network.

    Output:
      - (none): renders accordion tables in the notebook UI.
    """
    mo.accordion({
        "Table: Links between subsectors and PT stops": network.get_links("subsector", "public_transport"),
        "Table: Links between Swiss municipalities and PT stops": network.get_links(
            "municipality_swiss", "public_transport"
        ),
        "Table: Links between French municipalities and PT stops": network.get_links(
            "municipality_french", "public_transport"
        ),
        "Table: Links between Geneva municipalities and subsectors": network.get_links(
            "municipality_geneva", "subsector"
        ),
    })
    return


@app.cell(hide_code=True)
def _(explore_locations_by_type, explore_walk_edges, gva_data, network):
    """
    Description: Visualise the inter-layer links between Geneva subsectors and PT stops
    on an interactive map.

    Input:
      - explore_locations_by_type (callable): location polygon/point mapping helper.
      - explore_walk_edges (callable): edge line mapping helper.
      - gva_data (GenevaData): provides ``locations_gdf``.
      - network (Network): provides ``get_links("subsector", "public_transport")``.

    Output:
      - (none): renders the map in the notebook UI.
    """
    _m = explore_locations_by_type(gva_data.locations_gdf, types=["subsector"], as_points=False)
    _m = explore_walk_edges(network.get_links("subsector", "public_transport"), gva_data.locations_gdf, m=_m)
    _m = explore_locations_by_type(gva_data.locations_gdf, types=["subsector", "public_transport"], m=_m)

    mo.vstack([mo.md("Links between subsectors and PT stops:"), _m])
    return


@app.cell(hide_code=True)
def _(explore_locations_by_type, explore_walk_edges, gva_data, network):
    """
    Description: Combine Swiss and French municipality -> PT stop links and display them
    on an interactive map, overlaid on the corresponding municipality polygons.

    Input:
      - explore_locations_by_type (callable): location polygon/point mapping helper.
      - explore_walk_edges (callable): edge line mapping helper.
      - gva_data (GenevaData): provides ``locations_gdf``.
      - network (Network): provides municipality-to-PT link edge tables.

    Output:
      - (none): renders the map in the notebook UI.
    """
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
    """
    Description: Visualise the inter-layer links between Geneva canton municipalities and
    subsectors on an interactive map.

    Input:
      - explore_locations_by_type (callable): location polygon/point mapping helper.
      - explore_walk_edges (callable): edge line mapping helper.
      - gva_data (GenevaData): provides ``locations_gdf``.
      - network (Network): provides ``get_links("municipality_geneva", "subsector")``.

    Output:
      - (none): renders the map in the notebook UI.
    """
    _m = explore_locations_by_type(gva_data.locations_gdf, types=["municipality_geneva", "subsector"], as_points=False)
    _m = explore_walk_edges(network.get_links("municipality_geneva", "subsector"), gva_data.locations_gdf, m=_m)
    _m = explore_locations_by_type(gva_data.locations_gdf, types=["subsector", "municipality_geneva"], m=_m)

    mo.vstack([mo.md("Links between municipalities and subsectors"), _m])
    return


@app.cell(hide_code=True)
def _():
    """
    Description: Create a Marimo dropdown widget that lets the user choose which network
    layer to inspect for NA (not-assigned) source/sink node links. The selected value
    drives the map in the following cell reactively.

    Input:
      - (none)

    Output:
      - na_dropdown (mo.ui.dropdown): reactive dropdown with options for each layer name;
        the ``.value`` attribute is read by the next cell to filter displayed links.
    """
    # Interactive dropdown widget: the user picks which layer to inspect links between the NA
    # node and.  The selected value is used by the map cell below to filter links.
    na_dropdown = mo.ui.dropdown(
        ["public_transport", "subsector", "municipality_geneva", "municipality_swiss", "municipality_french"],
        value="subsector",   # default layer shown on first render
    )

    mo.hstack([mo.md("Links between NA and "), na_dropdown], justify="start")
    return (na_dropdown,)


@app.cell
def _(explore_locations_by_type, explore_walk_edges, na_dropdown, network):
    """
    Description: Reactively display links between the NA (not-assigned) source/sink node
    and whichever layer is currently selected in ``na_dropdown``. Updates automatically
    when the dropdown value changes.

    Input:
      - explore_locations_by_type (callable): location polygon/point mapping helper.
      - explore_walk_edges (callable): edge line mapping helper.
      - na_dropdown (mo.ui.dropdown): the layer selection dropdown from the previous cell.
      - network (Network): provides link edge tables and ``locations_gdf``.

    Output:
      - (none): renders the interactive map in the notebook UI.
    """
    _m = explore_walk_edges(
        network.get_links("na", na_dropdown.value),
        network.locations_gdf,
    )
    _m = explore_locations_by_type(network.locations_gdf, types=["na", na_dropdown.value], as_points=True, m=_m)

    _m
    return


if __name__ == "__main__":
    app.run()
