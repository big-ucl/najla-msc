from abc import ABC
from collections import defaultdict
from enum import StrEnum

import folium
import geopandas as gpd
import polars as pl
from shapely.geometry.linestring import LineString

from activitygraphs.utils import check_schema


class NetworkData(ABC):
    user_journeys: pl.DataFrame
    locations_gdf: gpd.GeoDataFrame


CRS = "EPSG:4326"
TILES = "Cartodb Positron"
NA_LON, NA_LAT = 6.1709475192397605, 46.24348817355701

LOCATIONS_SCHEMA = {
    "loc_id": "object",
    "loc_name": "object",
    "type": "object",
    "lon": "float64",
    "lat": "float64",
    "geometry": "geometry",
}
LOCATIONS_COLUMNS = LOCATIONS_SCHEMA.keys()

USER_JOURNEY_SCHEMA = pl.Schema({
    "user_id": pl.String,
    "journey_id": pl.String,
    "leg_id": pl.Int8,
    "leg_mode": pl.Categorical(),
    "leg_line": pl.String,
    "dep_day": pl.Date,
    "dep_time": pl.Time,
    "dep_purpose": pl.Categorical(),
    "dep_loc_id": pl.String,
    "arr_loc_id": pl.String,
    "arr_purpose": pl.Categorical(),
})

PT_EDGE_LIST_SCHEMA = pl.Schema({
    "orig_loc_id": pl.String,
    "route_id": pl.String,
    "dest_loc_id": pl.String,
    "first_departure_time": pl.Time,
    "last_departure_time": pl.Time,
    "avg_dwell_time_min": pl.Float64,
    "avg_travel_time_min": pl.Float64,
    "avg_headway_min": pl.Float64,
    "route_mode": pl.Categorical(),
    "route_name": pl.String,
})

TRANSFER_EDGE_LIST_SCHEMA = pl.Schema({
    "orig_loc_id": pl.String,
    "orig_route_id": pl.String,
    "dest_loc_id": pl.String,
    "dest_route_id": pl.String,
    "transfer_time_min": pl.Float64,
})


class Mode(StrEnum):
    OTHER = "mode_other"
    UNKNOWN = "mode_unknown"
    BOAT = "mode_boat"
    BUS = "mode_bus"
    COACH = "mode_coach"
    WALK = "mode_walk"
    CYCLE = "mode_cycle"
    MOTORCYCLE = "mode_motorcycle"
    TAXI = "mode_taxi"
    TRAIN = "mode_train"
    TRAMWAY = "mode_tramway"
    VEH_PASS = "mode_vehicle_passenger"
    CAR = "mode_car"


ROUTE_MODE_COLOUR_MAP = {
    Mode.BUS: "#82cfff",
    Mode.TRAMWAY: "#6929c4",
    Mode.TRAIN: "#0072c3",
    Mode.BOAT: "#005d5d",
    Mode.WALK: "#8a3800",
}
ROUTE_MODE_COLOUR_MAP = defaultdict(lambda: "#1192e8", **ROUTE_MODE_COLOUR_MAP)

LOCATION_TYPE_COLOR_MAP = {
    "na": "#570408",
    "subsector": "#9f1853",
    "municipality_swiss": "#8a3ffc",
    "municipality_french": "#8a3ffc",
    "public_transport": "#1192e8",
}


def add_line_geometry_to_edge_df(edge_df: pl.DataFrame, locations_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    check_schema(edge_df, pl.Schema({"orig_loc_id": pl.String, "dest_loc_id": pl.String}), ignore_extra_cols=True)
    check_schema(locations_gdf, LOCATIONS_SCHEMA)

    edge_locs = edge_df.select("orig_loc_id", "dest_loc_id").unique().to_pandas()
    locs = locations_gdf[["loc_id", "geometry"]]

    points = edge_locs.merge(locs.add_prefix("orig_"), on="orig_loc_id").merge(
        locs.add_prefix("dest_"), on="dest_loc_id"
    )

    points["geometry"] = points[["orig_geometry", "dest_geometry"]].apply(lambda x: LineString(x), axis=1)
    edges = points.drop(columns=["orig_geometry", "dest_geometry"]).set_geometry("geometry", crs=CRS)

    return edge_df.to_pandas().merge(edges, on=["orig_loc_id", "dest_loc_id"]).set_geometry("geometry", crs=CRS)


def _convert_to_point_geometry(locations_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    locations_gdf = locations_gdf.copy()
    locations_gdf.set_geometry(gpd.points_from_xy(locations_gdf["lon"], locations_gdf["lat"], crs=CRS), inplace=True)

    return locations_gdf


def explore_locations_by_type(
    locations_gdf: gpd.GeoDataFrame,
    types: str | list[str] | None = None,
    as_points: bool = True,
    m: folium.Map | None = None,
    tiles: str = TILES,
) -> folium.Map:
    locations_gdf = check_schema(locations_gdf, LOCATIONS_SCHEMA)

    if types is not None:
        types = [types] if isinstance(types, str) else types
        locations_gdf = locations_gdf[locations_gdf["type"].isin(types)]

    if as_points:
        locations_gdf = _convert_to_point_geometry(locations_gdf)

    colors = locations_gdf["type"].map(LOCATION_TYPE_COLOR_MAP)
    marker_kwds = {"radius": 4} if as_points else {}
    style_kwds = {} if as_points else {"opacity": 0.2, "fillOpacity": 0.05}
    highlight_kwds = {} if as_points else {"fillOpacity": 0.2}

    return locations_gdf.explore(
        m=m,
        color=colors,
        tiles=tiles,
        tooltip=["loc_name", "loc_id", "type"],
        marker_kwds=marker_kwds,
        style_kwds=style_kwds,
        highlight_kwds=highlight_kwds,
    )


def explore_locations_by_affluence(
    user_journeys_df: pl.DataFrame,
    locations_gdf: gpd.GeoDataFrame,
    loc_id_column: str,
    m: folium.Map | None = None,
    tiles: str = TILES,
) -> folium.Map:
    check_schema(user_journeys_df, USER_JOURNEY_SCHEMA)
    check_schema(locations_gdf, LOCATIONS_SCHEMA)

    stops = locations_gdf.merge(
        user_journeys_df[loc_id_column].value_counts(name="num_visits").to_pandas(),
        left_on="loc_id",
        right_on=loc_id_column,
        how="right",
    )
    stops = _convert_to_point_geometry(stops)

    return stops.explore(
        m=m,
        column="num_visits",
        cmap="viridis_r",
        tiles=tiles,
        scheme="NaturalBreaks",
        k=10,
        tooltip=["loc_name", "num_visits"],
        marker_kwds={"radius": 5},
    )


def explore_pt_edges_by_mode(
    pt_edge_df: pl.DataFrame, locations_gdf: gpd.GeoDataFrame, m: folium.Map | None = None, tiles: str = TILES
):
    pt_edge_df = check_schema(pt_edge_df, PT_EDGE_LIST_SCHEMA)
    locations_gdf = check_schema(locations_gdf, LOCATIONS_SCHEMA)

    formatted_routes_expr = pl.format(
        "{} (T={} min / H={} min) - {}",
        pl.element().struct.field("route_name"),
        pl.element().struct.field("avg_travel_time_min").round(1),
        pl.element().struct.field("avg_headway_min").round(1),
        pl.element().struct.field("route_id"),
    )

    _map_edges = (
        pt_edge_df.group_by("orig_loc_id", "dest_loc_id", "route_mode")
        .agg(
            travel_time_min=pl.col("avg_travel_time_min").mean(),
            route_attrs=pl.struct(["route_id", "route_name", "avg_travel_time_min", "avg_headway_min"]).unique(),
        )
        .with_columns(pl.col("route_attrs").list.eval(formatted_routes_expr).list.join("<br />"))
    )

    map_edges_gdf = add_line_geometry_to_edge_df(_map_edges, locations_gdf)
    route_mode_colors = map_edges_gdf["route_mode"].map(ROUTE_MODE_COLOUR_MAP).astype(str)

    return map_edges_gdf.explore(
        m=m,
        tiles=tiles,
        # column="route_mode",
        color=route_mode_colors,
        tooltip=["route_mode", "travel_time_min", "route_attrs"],
        style_kwds={"weight": 3, "opacity": 0.5},
    )


def explore_transfer_edges(
    transfer_edge_df: pl.DataFrame,
    locations_gdf: gpd.GeoDataFrame,
    show_locations: bool = False,
    m: folium.Map | None = None,
    tiles: str = TILES,
) -> folium.Map:
    check_schema(transfer_edge_df, TRANSFER_EDGE_LIST_SCHEMA)
    check_schema(locations_gdf, LOCATIONS_SCHEMA)

    walk_color = ROUTE_MODE_COLOUR_MAP[Mode.WALK]

    external_transfers = transfer_edge_df.filter(pl.col("orig_loc_id") != pl.col("dest_loc_id"))
    external_transfers_gdf = add_line_geometry_to_edge_df(external_transfers, locations_gdf)
    m = external_transfers_gdf.explore(
        m=m,
        tiles=tiles,
        style_kwds={"weight": 3, "opacity": 0.5, "color": walk_color, "dashArray": "10"},
    )

    if not show_locations:
        return m

    internal_transfers = transfer_edge_df.filter(
        pl.col("orig_loc_id") == pl.col("dest_loc_id"), dest_route_id="transfer_route"
    )

    route_transfer_locs_df = (
        internal_transfers.rename({"orig_loc_id": "loc_id"})
        .group_by("loc_id")
        .agg(pl.col("transfer_time_min").first(), num_transfers=pl.len())
    )
    route_transfer_locs_df = locations_gdf.merge(route_transfer_locs_df.to_pandas(), on="loc_id")

    m = route_transfer_locs_df.explore(m=m, tiles=tiles, color=walk_color)

    return m


def add_legend_pane_to_map(m: folium.Map, legends: dict[str, tuple[str, dict[str, str]]]):
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
        translucent = col + "C0"

        if shape == "circle":
            return f'<i style="border: 1px {col}; background: {translucent}; width: 10px; height: 10px; margin-right: 5px; border-radius: 10px"></i>'

        if shape == "square":
            return f'<i style="border: 1px {col}; background: {translucent}; width: 10px; height: 10px; margin-right: 5px;"></i>'

        if shape == "line":
            return f'<i style="background: {col}; width: 10px; height: 3px; margin-right: 5px;"></i>'

        return ""

    for legend_name, (icon_shape, colour_map) in legends.items():
        legend_html += f"<b>{legend_name}</b><br>"

        for element, colour in colour_map.items():
            icon = create_icon(icon_shape, colour)

            legend_html += f"""
                <div style="display: flex; align-items: center">
                    {icon} {element}
                </div>
            """

    legend_html += "</div>"

    m.get_root().html.add_child(folium.Element(legend_html))
