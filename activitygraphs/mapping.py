"""
Interactive map visualisation helpers using Folium and GeoPandas.

This module provides functions that render transport network data on interactive
Leaflet maps (via the Folium library).  Each function takes a DataFrame of
network elements (locations, edges, etc.) and adds a styled layer to an
optional existing Folium map object, returning the map so calls can be chained.

Available visualisation functions:
  - explore_locations_by_type          : colour-code location markers by type
  - explore_locations_by_affluence     : size/colour markers by visit frequency
  - explore_pt_edges_by_mode           : draw transit edges coloured by route mode
  - explore_walk_edges                 : draw walking edges as dashed lines
  - explore_link_edges                 : draw synthetic link edges as dashed lines
  - explore_transfer_edges             : draw transit transfer edges (inter-stop walks)
  - add_legend_pane_to_map             : attach an HTML legend box to any Folium map

Internal helpers:
  - _add_line_geometry_to_edge_df      : join location geometries to an edge list
                                         to produce a GeoDataFrame of LineStrings.

Module-level constants:
  - LOCATION_TYPE_COLOR_MAP  : hex colour per location type string.
  - ROUTE_MODE_COLOUR_MAP    : hex colour per transit route mode (with default).
  - TILES                    : default Folium tile provider name.
"""

from collections import defaultdict

import folium
import geopandas as gpd
import polars as pl
from shapely import LineString

from activitygraphs.base import (
    CRS,
    LOCATIONS_SCHEMA,
    PT_EDGE_LIST_SCHEMA,
    TRANSFER_EDGE_LIST_SCHEMA,
    USER_JOURNEY_SCHEMA,
    WALK_EDGE_LIST_SCHEMA,
    Mode,
)
from activitygraphs.utils import check_schema, convert_locations_to_point_geometry

# ---------------------------------------------------------------------------
# Colour constants for map visualisation
# ---------------------------------------------------------------------------

# Maps each location *type* string (from the LOCATIONS_SCHEMA "type" column)
# to a hex colour code used when plotting location markers.
# "na"               → very dark red (unknown / unclassified locations)
# "subsector"        → deep pink (Geneva administrative subsectors)
# "municipality_*"   → purple (any Swiss or French municipality zone)
# "public_transport" → bright blue (transit stops / stations)
LOCATION_TYPE_COLOR_MAP = {
    "na": "#570408",                     # Dark red — unclassified / missing type
    "subsector": "#9f1853",              # Deep pink — Geneva administrative subsectors
    "municipality_geneva": "#8a3ffc",   # Purple — Geneva municipality zones
    "municipality_swiss": "#8a3ffc",    # Purple — Swiss municipality zones
    "municipality_french": "#8a3ffc",   # Purple — French municipality zones (cross-border)
    "public_transport": "#1192e8",      # Bright blue — public-transport stops
}


# Maps each transit route *mode* (Mode enum string) to a hex colour code used
# when drawing network edges on the map.
ROUTE_MODE_COLOUR_MAP = {
    Mode.BUS: "#82cfff",     # Light blue — bus routes
    Mode.TRAMWAY: "#6929c4", # Purple — tram / light rail routes
    Mode.TRAIN: "#0072c3",   # Dark blue — heavy rail / suburban train routes
    Mode.BOAT: "#005d5d",    # Teal — ferry / boat routes
    Mode.WALK: "#8a3800",    # Brown — walking edges
    Mode.OTHER: "#808080",   # Grey — any unclassified route mode
}
# Convert to a defaultdict so that any mode not listed above falls back to the
# bright blue default colour (#1192e8), avoiding KeyErrors on unknown modes.
ROUTE_MODE_COLOUR_MAP = defaultdict(lambda: "#1192e8", **ROUTE_MODE_COLOUR_MAP)

# Name of the default Folium/Leaflet tile layer used for all maps.
# "Cartodb Positron" is a clean, light-grey basemap well-suited for data overlays.
TILES = "Cartodb Positron"


def explore_locations_by_type(
    locations_gdf: gpd.GeoDataFrame,
    types: str | list[str] | None = None,
    as_points: bool = True,
    m: folium.Map | None = None,
    tiles: str = TILES,
) -> folium.Map:
    """
    Description: Plot locations on an interactive Folium map, colouring each
    marker according to its location type (using LOCATION_TYPE_COLOR_MAP).
    Can optionally filter to show only certain location types, and can render
    locations as circle markers (points) or as their original polygon shapes.

    Input:
      - locations_gdf (gpd.GeoDataFrame): locations table matching LOCATIONS_SCHEMA
                                          (loc_id, loc_name, type, lon, lat, geometry).
      - types (str | list[str] | None): if given, only show locations whose "type"
                                        column value matches.  Pass a single string
                                        or a list of strings.  None shows all types.
      - as_points (bool): if True (default), replace polygon geometries with point
                          markers derived from lon/lat; if False, render polygons.
      - m (folium.Map | None): existing Folium map to add the layer to.  If None,
                               a new map is created automatically.
      - tiles (str): name of the Folium tile layer to use. Defaults to TILES.

    Output:
      - (folium.Map): Folium map with the locations layer added.
    """
    # Validate that the input GeoDataFrame matches the locations schema.
    check_schema(locations_gdf, LOCATIONS_SCHEMA)

    # Optionally filter to only show the requested type(s).
    if types is not None:
        # Normalise to a list so we can always use .isin().
        types = [types] if isinstance(types, str) else types
        locations_gdf = locations_gdf[locations_gdf["type"].isin(types)]

    # Optionally convert polygon geometries to point markers.
    if as_points:
        locations_gdf = convert_locations_to_point_geometry(locations_gdf)

    # Map each location's type string to the corresponding hex colour.
    colors = locations_gdf["type"].map(LOCATION_TYPE_COLOR_MAP)
    # For point markers, set a radius; for polygons, use default (no radius).
    marker_kwds = {"radius": 4} if as_points else {}
    # Make polygons semi-transparent so the basemap is visible underneath.
    style_kwds = {} if as_points else {"opacity": 0.2, "fillOpacity": 0.05}
    # When hovering over a polygon, increase its fill opacity slightly.
    highlight_kwds = {} if as_points else {"fillOpacity": 0.2}

    return locations_gdf.explore(
        m=m,
        color=colors,                              # Per-row colour from type → colour map
        tiles=tiles,                               # Basemap tile provider
        tooltip=["loc_name", "loc_id", "type"],    # Columns shown on hover
        marker_kwds=marker_kwds,                   # Circle marker options (radius, etc.)
        style_kwds=style_kwds,                     # GeoJSON style for polygons
        highlight_kwds=highlight_kwds,             # Style applied when mouse hovers
    )


def explore_locations_by_affluence(
    user_journeys_df: pl.DataFrame,
    locations_gdf: gpd.GeoDataFrame,
    loc_id_column: str,
    m: folium.Map | None = None,
    tiles: str = TILES,
) -> folium.Map:
    """
    Description: Plot locations as circle markers whose colour encodes how many
    times each location was visited in the journey survey data.  More-visited
    locations appear in lighter/brighter colours (viridis_r colour scheme),
    providing a quick "hotspot" visualisation.

    The colour scale is discretised into 10 classes using Jenks Natural Breaks
    classification, which groups values where natural gaps exist in the data.

    Input:
      - user_journeys_df (pl.DataFrame): journey data matching USER_JOURNEY_SCHEMA.
                                         The column specified by ``loc_id_column``
                                         is used to count visits per location.
      - locations_gdf (gpd.GeoDataFrame): locations table matching LOCATIONS_SCHEMA.
      - loc_id_column (str): name of the column in ``user_journeys_df`` that holds
                             location IDs to count.  Typically "dep_loc_id" or
                             "arr_loc_id".
      - m (folium.Map | None): existing Folium map to add the layer to, or None
                               to create a new map.
      - tiles (str): Folium tile layer name. Defaults to TILES.

    Output:
      - (folium.Map): Folium map with the visit-frequency location layer added.
    """
    # Validate input schemas before processing.
    check_schema(user_journeys_df, USER_JOURNEY_SCHEMA)
    check_schema(locations_gdf, LOCATIONS_SCHEMA)

    # Count how many times each location ID appears in the specified column.
    # value_counts returns a Polars DataFrame with columns [loc_id_column, "num_visits"].
    # Merge right so only locations that appear in the journey data are shown.
    stops = locations_gdf.merge(
        user_journeys_df[loc_id_column].value_counts(name="num_visits").to_pandas(),
        left_on="loc_id",
        right_on=loc_id_column,
        how="right",
    )
    # Convert polygon geometries to point markers using lon/lat columns.
    stops = convert_locations_to_point_geometry(stops)

    return stops.explore(
        m=m,
        column="num_visits",               # Colour markers by visit count
        cmap="viridis_r",                  # Reversed viridis: dark = rare, bright = popular
        tiles=tiles,
        scheme="NaturalBreaks",            # Jenks Natural Breaks classification for colour bins
        k=10,                              # Number of colour classes
        tooltip=["loc_name", "num_visits"],  # Columns shown on hover
        marker_kwds={"radius": 5},         # Circle marker radius in pixels
    )


def explore_pt_edges_by_mode(
    pt_edge_df: pl.DataFrame, locations_gdf: gpd.GeoDataFrame, m: folium.Map | None = None, tiles: str = TILES
):
    """
    Description: Plot public-transport network edges (connections between transit
    stops) as coloured lines on a Folium map.  Each edge is coloured according to
    the route mode (bus = light blue, tram = purple, train = dark blue, etc.).

    Multiple routes operating between the same pair of stops are aggregated into
    a single line; hovering over the line shows a tooltip with travel time and a
    formatted list of all routes that use that stop-pair.

    Input:
      - pt_edge_df (pl.DataFrame): public-transport edge list matching
                                   PT_EDGE_LIST_SCHEMA (one row per route-segment).
      - locations_gdf (gpd.GeoDataFrame): locations table matching LOCATIONS_SCHEMA,
                                          used to look up stop coordinates.
      - m (folium.Map | None): existing Folium map to add the layer to.  None
                               creates a new map.
      - tiles (str): Folium tile layer name. Defaults to TILES.

    Output:
      - (folium.Map): Folium map with the PT edges layer added.
    """
    # Validate input schemas.
    pt_edge_df = check_schema(pt_edge_df, PT_EDGE_LIST_SCHEMA)
    locations_gdf = check_schema(locations_gdf, LOCATIONS_SCHEMA)

    # Polars expression that formats a route struct into a human-readable string:
    # "Line 15 (T=8.2 min / N=42 / H=7.5 min) - route_id_xyz"
    formatted_routes_expr = pl.format(
        "{} (T={} min / N={} / H={} min) - {}",
        pl.element().struct.field("route_name"),           # Route name (e.g. "Line 15")
        pl.element().struct.field("travel_time_min").round(1),  # Travel time rounded to 1 decimal
        pl.element().struct.field("daily_trip_count"),     # Number of trips per day
        pl.element().struct.field("avg_headway_min").round(1),  # Average headway (minutes)
        pl.element().struct.field("route_id"),             # GTFS route_id for reference
    )

    # Aggregate multiple routes between the same stop-pair into a single row.
    # For each unique (orig, dest, mode) combination:
    #   - compute mean travel time across all routes
    #   - collect all route attributes as a list of structs
    _map_edges = (
        pt_edge_df
        .group_by("orig_loc_id", "dest_loc_id", "route_mode")
        .agg(
            travel_time_min=pl.col("travel_time_min").mean(),   # Average travel time for this stop-pair
            route_attrs=pl.struct([                              # List of per-route detail structs
                "route_id",
                "route_name",
                "travel_time_min",
                "daily_trip_count",
                "avg_headway_min",
            ]).unique(),
        )
        # Convert the list of route structs to a single HTML-formatted string for the tooltip.
        .with_columns(pl.col("route_attrs").list.eval(formatted_routes_expr).list.join("<br />"))
    )

    # Join stop coordinates and build LineString geometries for each edge.
    map_edges_gdf = _add_line_geometry_to_edge_df(_map_edges, locations_gdf)
    # Map the route_mode column to colours using the ROUTE_MODE_COLOUR_MAP.
    route_mode_colors = map_edges_gdf["route_mode"].map(ROUTE_MODE_COLOUR_MAP).astype(str)

    return map_edges_gdf.explore(
        m=m,
        tiles=tiles,
        color=route_mode_colors,                                       # Per-edge colour by mode
        tooltip=["route_mode", "travel_time_min", "route_attrs"],      # Hover tooltip columns
        style_kwds={"weight": 3, "opacity": 0.5},                      # Line width and transparency
    )


def explore_walk_edges(
    walk_edge_df: pl.DataFrame,
    locations_gdf: gpd.GeoDataFrame,
    m: folium.Map | None = None,
    tiles: str = TILES,
    walk_color: str = ROUTE_MODE_COLOUR_MAP[Mode.WALK],
) -> folium.Map:
    """
    Description: Plot walking network edges (pedestrian connections between nearby
    locations) as dashed lines on a Folium map.  All edges are drawn in the same
    colour (default: the walking brown from ROUTE_MODE_COLOUR_MAP).

    Input:
      - walk_edge_df (pl.DataFrame): walking edge list matching WALK_EDGE_LIST_SCHEMA
                                     (orig_loc_id, dest_loc_id, travel_time_min).
      - locations_gdf (gpd.GeoDataFrame): locations table matching LOCATIONS_SCHEMA,
                                          used to look up stop coordinates.
      - m (folium.Map | None): existing Folium map to add the layer to.  None
                               creates a new map.
      - tiles (str): Folium tile layer name. Defaults to TILES.
      - walk_color (str): hex colour string for the walking edges.  Defaults to
                          the brown colour defined in ROUTE_MODE_COLOUR_MAP.

    Output:
      - (folium.Map): Folium map with the walking edges layer added.
    """
    # Validate input schemas.
    check_schema(walk_edge_df, WALK_EDGE_LIST_SCHEMA)
    check_schema(locations_gdf, LOCATIONS_SCHEMA)

    # Build a GeoDataFrame of LineString geometries connecting each origin/destination pair.
    walk_edge_gdf = _add_line_geometry_to_edge_df(walk_edge_df, locations_gdf)
    return walk_edge_gdf.explore(
        m=m,
        tiles=tiles,
        # Render as dashed lines (dashArray="10") to visually distinguish from transit edges.
        style_kwds={"weight": 3, "opacity": 0.5, "color": walk_color, "dashArray": "10"},
    )


def explore_link_edges(
    walk_edge_df: pl.DataFrame,
    locations_gdf: gpd.GeoDataFrame,
    m: folium.Map | None = None,
    tiles: str = TILES,
    link_color: str = ROUTE_MODE_COLOUR_MAP[Mode.OTHER],
) -> folium.Map:
    """
    Description: Plot synthetic link edges (connections used to join different
    network layers, e.g. a zone centroid to a nearby transit stop) as dashed
    lines on a Folium map.  This is a thin wrapper around ``explore_walk_edges``
    that uses a grey default colour to distinguish link edges from walking edges.

    Input:
      - walk_edge_df (pl.DataFrame): link edge list conforming to LINK_EDGE_LIST_SCHEMA
                                     (same columns as WALK_EDGE_LIST_SCHEMA).
      - locations_gdf (gpd.GeoDataFrame): locations table matching LOCATIONS_SCHEMA.
      - m (folium.Map | None): existing Folium map to add the layer to.  None
                               creates a new map.
      - tiles (str): Folium tile layer name. Defaults to TILES.
      - link_color (str): hex colour string for link edges.  Defaults to the grey
                          colour for Mode.OTHER in ROUTE_MODE_COLOUR_MAP.

    Output:
      - (folium.Map): Folium map with the link edges layer added.
    """
    # Reuse explore_walk_edges with the link colour (grey by default).
    return explore_walk_edges(walk_edge_df, locations_gdf, m=m, tiles=tiles, walk_color=link_color)


def explore_transfer_edges(
    transfer_edge_df: pl.DataFrame,
    locations_gdf: gpd.GeoDataFrame,
    show_locations: bool = False,
    m: folium.Map | None = None,
    tiles: str = TILES,
) -> folium.Map:
    """
    Description: Plot transit transfer edges on a Folium map.  A transfer edge
    represents the connection a passenger must make when switching from one transit
    route to another.

    Two types of transfers exist:
      - External transfers: the passenger walks between two *different* stops
        (orig_loc_id != dest_loc_id).  These are drawn as dashed walking lines.
      - Internal transfers: the passenger changes routes at the *same* stop
        (orig_loc_id == dest_loc_id).  When ``show_locations=True``, the stops
        that have internal transfers are plotted as separate markers.

    Input:
      - transfer_edge_df (pl.DataFrame): transfer edge list matching
                                         TRANSFER_EDGE_LIST_SCHEMA.
      - locations_gdf (gpd.GeoDataFrame): locations table matching LOCATIONS_SCHEMA.
      - show_locations (bool): if True, also plot markers at stops that have
                               internal (same-stop) transfers.  Defaults to False.
      - m (folium.Map | None): existing Folium map to add the layer to.  None
                               creates a new map.
      - tiles (str): Folium tile layer name. Defaults to TILES.

    Output:
      - (folium.Map): Folium map with the transfer edges (and optionally transfer
                      stop markers) layers added.
    """
    # Validate input schemas.
    check_schema(transfer_edge_df, TRANSFER_EDGE_LIST_SCHEMA)
    check_schema(locations_gdf, LOCATIONS_SCHEMA)

    # Extract only the external transfers (different origin and destination stops).
    # Select only the three columns needed by explore_walk_edges.
    external_transfers_df = transfer_edge_df.filter(pl.col("orig_loc_id") != pl.col("dest_loc_id")).select(
        "orig_loc_id", "dest_loc_id", "travel_time_min"
    )

    # Use the walking colour for transfer edges (they represent walking between stops).
    walk_color = ROUTE_MODE_COLOUR_MAP[Mode.WALK]
    # Draw the external transfer walking lines on the map.
    m = explore_walk_edges(external_transfers_df, locations_gdf, m=m, tiles=tiles, walk_color=walk_color)

    # If the user does not want internal transfer stop markers, return here.
    if not show_locations:
        return m

    # Filter to internal transfers only (same stop, dest_route_id = "transfer_route").
    internal_transfers = transfer_edge_df.filter(
        pl.col("orig_loc_id") == pl.col("dest_loc_id"), dest_route_id="transfer_route"
    )

    # Aggregate per stop: count how many internal transfers occur and get travel time.
    route_transfer_locs_df = (
        internal_transfers
        .rename({"orig_loc_id": "loc_id"})  # Rename for the merge
        .group_by("loc_id")
        .agg(pl.col("travel_time_min").first(), num_transfers=pl.len())  # Count transfers per stop
    )
    # Merge transfer stats with location geometries for mapping.
    route_transfer_locs_df = locations_gdf.merge(route_transfer_locs_df.to_pandas(), on="loc_id")

    # Add the internal transfer stop markers to the map.
    m = route_transfer_locs_df.explore(m=m, tiles=tiles, color=walk_color)

    return m


def add_legend_pane_to_map(m: folium.Map, legends: dict[str, tuple[str, dict[str, str]]]):
    """
    Description: Attach a fixed-position HTML legend box to a Folium map.  The
    legend is rendered in the bottom-right corner and supports multiple legend
    groups, each with an icon shape (circle, square, or line) and a colour map.

    Example:
        add_legend_pane_to_map(m, {
            "Location type": ("circle", {"Stop": "#1192e8", "Zone": "#9f1853"}),
            "Route mode":    ("line",   {"Bus": "#82cfff", "Tram": "#6929c4"}),
        })

    Input:
      - m (folium.Map): the Folium map object to attach the legend to.
      - legends (dict[str, tuple[str, dict[str, str]]]): a dict where each key is
          a legend group heading (e.g. "Route mode") and each value is a 2-tuple:
            [0] icon_shape (str): "circle", "square", or "line".
            [1] colour_map (dict[str, str]): maps label string to hex colour string.

    Output:
      - (None): modifies ``m`` in place by adding the legend HTML element.
    """
    # Start building the legend HTML as a fixed-position floating div.
    legend_html = """
    <div style="
        position: fixed;
        bottom: 40px;
        right: 40px;
        z-index:9999;
        background-color:white;
        padding: 10px;
        border:2px solid grey;
        border-radius:5px;
        font-size:14px;
    ">
    """

    def create_icon(shape: str, col: str):
        """
        Description: Generate an HTML <i> tag that renders a small coloured icon
        to use as a legend symbol.  Supports circle, square, and line shapes.

        Input:
          - shape (str): one of "circle", "square", or "line".
          - col (str): hex colour string for the icon background (e.g. "#1192e8").

        Output:
          - (str): an HTML string containing the <i> element, or "" if shape is
                   unrecognised.
        """
        # Append "C0" to the colour hex to create a 75%-opacity translucent variant.
        translucent = col + "C0"

        if shape == "circle":
            # Circular icon: uses border-radius to round the div into a circle.
            return f'<i style="border: 1px {col}; background: {translucent}; width: 10px; height: 10px; margin-right: 5px; border-radius: 10px"></i>'

        if shape == "square":
            # Square icon: plain coloured rectangle.
            return f'<i style="border: 1px {col}; background: {translucent}; width: 10px; height: 10px; margin-right: 5px;"></i>'

        if shape == "line":
            # Line icon: a wide, short coloured bar to simulate a map line.
            return f'<i style="background: {col}; width: 10px; height: 3px; margin-right: 5px;"></i>'

        # Unknown shape — return an empty string (no icon).
        return ""

    # Iterate over each legend group and append its heading and coloured entries.
    for legend_name, (icon_shape, colour_map) in legends.items():
        # Add a bold heading for this legend group.
        legend_html += f"<b>{legend_name}</b><br>"

        for element, colour in colour_map.items():
            # Generate the icon HTML for this element.
            icon = create_icon(icon_shape, colour)

            # Append a flex row with the icon and label text.
            legend_html += f"""
                <div style="display: flex; align-items: center">
                    {icon} {element}
                </div>
            """

    # Close the outer legend div.
    legend_html += "</div>"

    # Inject the legend HTML into the map's root element so it renders on the page.
    m.get_root().html.add_child(folium.Element(legend_html))


def _add_line_geometry_to_edge_df(edge_df: pl.DataFrame, locations_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Description: Join location point geometries to an edge list and produce a
    GeoDataFrame where each edge is represented as a Shapely LineString connecting
    its origin and destination stop geometries.  This is an internal helper used
    by all the ``explore_*_edges`` functions before calling GeoPandas .explore().

    The function:
      1. Validates that ``edge_df`` has orig_loc_id and dest_loc_id columns.
      2. Converts location polygons to point geometries (centroid / lon-lat).
      3. Merges origin and destination point geometries onto the edge table.
      4. Builds a LineString from each origin–destination point pair.
      5. Returns a full GeoDataFrame (all original edge columns + geometry).

    Input:
      - edge_df (pl.DataFrame): any edge-list DataFrame with at least
                                ``orig_loc_id`` and ``dest_loc_id`` columns.
                                Extra columns are kept and included in the output.
      - locations_gdf (gpd.GeoDataFrame): locations table matching LOCATIONS_SCHEMA,
                                          used to look up coordinates for each stop.

    Output:
      - (gpd.GeoDataFrame): the edge DataFrame converted to a GeoDataFrame where
                            the "geometry" column holds a WGS-84 LineString for
                            each edge (origin point → destination point).
    """
    # Validate that the required ID columns are present (extra columns are fine).
    check_schema(edge_df, pl.Schema({"orig_loc_id": pl.String, "dest_loc_id": pl.String}), ignore_extra_cols=True)
    check_schema(locations_gdf, LOCATIONS_SCHEMA)

    # Get the unique set of origin–destination pairs to avoid redundant geometry merges.
    edge_locs = edge_df.select("orig_loc_id", "dest_loc_id").unique().to_pandas()

    # Convert location polygons to point geometries (uses lon/lat columns).
    locs = convert_locations_to_point_geometry(locations_gdf)
    # Keep only the identifier and geometry columns needed for the merge.
    locs = locs[["loc_id", "geometry"]]

    # Merge origin and destination point geometries onto the edge table using prefixed columns.
    # add_prefix("orig_") renames loc_id → orig_loc_id and geometry → orig_geometry.
    # add_prefix("dest_") renames loc_id → dest_loc_id and geometry → dest_geometry.
    points = edge_locs.merge(locs.add_prefix("orig_"), on="orig_loc_id").merge(
        locs.add_prefix("dest_"), on="dest_loc_id"
    )

    # Create a LineString from each (origin_point, destination_point) pair.
    # The lambda receives a Series [orig_geometry, dest_geometry] and builds the line.
    points["geometry"] = points[["orig_geometry", "dest_geometry"]].apply(lambda x: LineString(x), axis=1)
    # Drop the helper point columns and set the new LineString geometry.
    edges = points.drop(columns=["orig_geometry", "dest_geometry"]).set_geometry("geometry", crs=CRS)

    # Merge all original edge columns back with the geometry, then set the CRS.
    return edge_df.to_pandas().merge(edges, on=["orig_loc_id", "dest_loc_id"]).set_geometry("geometry", crs=CRS)
