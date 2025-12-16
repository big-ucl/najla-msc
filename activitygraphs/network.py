from abc import ABC
from collections.abc import Callable, Iterable
from enum import Enum, auto
from typing import Literal, Self

import geopandas as gpd
import polars as pl
from geopandas.sindex import SpatialIndex
from shapely.geometry.polygon import Polygon

from activitygraphs import utils
from activitygraphs.base import (
    EDGE_LIST_SCHEMA,
    LINK_EDGE_LIST_SCHEMA,
    LOCATIONS_SCHEMA,
    USER_JOURNEY_SCHEMA,
    WALK_EDGE_LIST_SCHEMA,
)
from activitygraphs.routing import TravelTimeCalculator
from activitygraphs.utils import check_geometry_shapes, check_schema, convert_locations_to_point_geometry

NA_LON, NA_LAT = 6.1709475192397605, 46.24348817355701

TravelTimeFactory = float | pl.Expr | pl.DataFrame | TravelTimeCalculator | Callable[[str, str], float]


class NetworkData(ABC):
    user_journeys: pl.DataFrame
    locations_gdf: gpd.GeoDataFrame


class LayerType(Enum):
    PUBLIC_TRANSPORT = auto()
    POINT = auto()
    PLANAR = auto()


class Layer:
    def __init__(self, name: str, layer_type: LayerType, loc_ids: Iterable[str], edge_list: pl.DataFrame | None):
        edge_list = edge_list if edge_list is not None else pl.DataFrame(schema=EDGE_LIST_SCHEMA)
        check_schema(edge_list, EDGE_LIST_SCHEMA, ignore_extra_cols=True)

        self.name = name
        self.type = layer_type
        self.loc_ids = list(loc_ids)

        mismatches = edge_list.filter(
            ~pl.col("orig_loc_id").is_in(self.loc_ids) or ~pl.col("dest_loc_id").is_in(self.loc_ids)
        )

        if mismatches.count() > 0:
            mismatches_fmt = edge_list.select(pl.format("({}, {})", "orig_loc_id", "dest_loc_id"))
            mismatches_str = ",".join(mismatches_fmt.to_series())
            raise ValueError(f"Found edges that connect outside of layer:\n*******{mismatches_str}\n*********")

        self.edge_list = edge_list


class Network:
    def __init__(self, network_data: NetworkData):
        self._locations_gdf = check_schema(network_data.locations_gdf, LOCATIONS_SCHEMA).copy()
        self._user_journeys = check_schema(network_data.user_journeys, USER_JOURNEY_SCHEMA)
        self._layers: dict[str, Layer] = {}
        self._links: dict[tuple[str, str], pl.DataFrame] = {}

    @property
    def locations_gdf(self) -> gpd.GeoDataFrame:
        return self._locations_gdf.copy()

    @property
    def locations_df(self) -> pl.DataFrame:
        return utils.gdf_to_polars(self._locations_gdf)

    def get_layer_locations(self, name: str) -> gpd.GeoDataFrame:
        loc_ids = self._layers[name].loc_ids
        return self._locations_gdf[self.locations_gdf["loc_id"].isin(loc_ids)]

    def add_layer(
        self,
        name: str,
        layer_type: LayerType,
        loc_ids: gpd.GeoDataFrame | Iterable[str],
        edge_list: pl.DataFrame | None,
    ) -> Self:
        loc_ids = self._check_layer_loc_ids(name, loc_ids)
        self._layers[name] = Layer(name, layer_type, loc_ids, edge_list)
        return self

    def add_pt_layer(self, name: str, pt_edge_df: pl.DataFrame, transfer_edge_df: pl.DataFrame) -> Self:
        pass

    def add_planar_layer(
        self, name: str, loc_ids: gpd.GeoDataFrame | Iterable[str], travel_time_f: TravelTimeFactory
    ) -> Self:
        if name in self._layers:
            raise ValueError(f"Layer `{name}` is already in the network.")

        loc_ids = self._check_layer_loc_ids(name, loc_ids)
        planar_locations_gdf = self._locations_gdf[self.locations_gdf["loc_id"].isin(loc_ids)]
        planar_edges = build_planar_edges(planar_locations_gdf, travel_time_f)

        return self.add_layer(name, LayerType.PLANAR, loc_ids, planar_edges)

    def _check_layer_loc_ids(self, name: str, loc_ids: gpd.GeoDataFrame | Iterable[str]):
        if name in self._layers:
            raise ValueError(f"Layer `{name}` is already in the network.")

        if isinstance(loc_ids, gpd.GeoDataFrame):
            check_schema(loc_ids, LOCATIONS_SCHEMA)
            loc_ids = loc_ids["loc_id"].tolist()
        else:
            loc_ids = list(loc_ids)

        unknown_locations = set(loc_ids).difference(self._locations_gdf["loc_id"])
        if unknown_locations:
            raise ValueError(f"Locations in `loc_ids` are not in network locations:\n\t{', '.join(unknown_locations)}")

        return loc_ids

    def connect_layers(
        self,
        lower: str,
        upper: str,
        travel_time_f: TravelTimeFactory,
        mode: Literal["strict", "centroid_strict", "nearest", "centroid_nearest"] = "strict",
        direction: Literal["both", "ascending", "descending"] = "both",
    ) -> Self:
        lower_locations_gdf = self.get_layer_locations(lower)
        upper_locations_gdf = self.get_layer_locations(upper)

        link_edges = build_layer_link_edges(lower_locations_gdf, upper_locations_gdf, travel_time_f, mode, direction)
        self._links[(lower, upper)] = link_edges

        return self


def build_planar_edges(planar_locations_gdf: gpd.GeoDataFrame, travel_time_f: TravelTimeFactory) -> pl.DataFrame:
    check_schema(planar_locations_gdf, LOCATIONS_SCHEMA)
    check_geometry_shapes(planar_locations_gdf.geometry, "Polygon", "MultiPolygon")

    neighbours = _compute_neighbour_loc_ids(planar_locations_gdf)
    planar_edge_df = neighbours.select(orig_loc_id="loc_id", dest_loc_id="neighbour")
    planar_edge_df = _add_travel_time_column(planar_edge_df, travel_time_f)

    return check_schema(planar_edge_df, WALK_EDGE_LIST_SCHEMA)


def build_layer_link_edges(
    lower_locations_gdf: gpd.GeoDataFrame,
    upper_locations_gdf: gpd.GeoDataFrame,
    travel_time_f: TravelTimeFactory,
    mode: Literal["strict", "centroid_strict", "nearest", "centroid_nearest"] = "strict",
    direction: Literal["both", "ascending", "descending"] = "both",
) -> pl.DataFrame:
    check_schema(lower_locations_gdf, LOCATIONS_SCHEMA)
    check_schema(upper_locations_gdf, LOCATIONS_SCHEMA)

    if mode == "centroid_strict" or mode == "centroid_nearest":
        upper_locations_gdf = convert_locations_to_point_geometry(upper_locations_gdf)

    projected_crs = upper_locations_gdf.estimate_utm_crs()
    upper = upper_locations_gdf[["loc_id", "geometry"]].to_crs(projected_crs)
    lower = lower_locations_gdf[["loc_id", "geometry"]].to_crs(projected_crs)

    if mode == "strict" or mode == "centroid_strict":
        intersection = upper.sjoin(lower, predicate="intersects")
    elif mode == "nearest" or mode == "centroid_nearest":
        intersection = upper.sjoin_nearest(lower)
    else:
        raise ValueError(f"Invalid mode {mode}, must be 'strict' or 'nearest'")

    edges = pl.DataFrame(intersection[["loc_id_left", "loc_id_right"]]).select(
        orig_loc_id="loc_id_left", dest_loc_id="loc_id_right"
    )
    edged_reversed = edges.select(orig_loc_id="dest_loc_id", dest_loc_id="orig_loc_id")

    if direction == "both":
        edges = pl.concat([edges, edged_reversed])
    elif direction == "ascending":
        edges = edged_reversed
    elif direction != "descending":
        raise ValueError(f"Invalid direction {direction}, must be 'both', 'ascending' or 'descending'")

    edges = _add_travel_time_column(edges, travel_time_f)

    return check_schema(edges, LINK_EDGE_LIST_SCHEMA)


def _add_travel_time_column(edge_df: pl.DataFrame, travel_time_f: TravelTimeFactory) -> pl.DataFrame:
    check_schema(edge_df, pl.Schema({"orig_loc_id": pl.String, "dest_loc_id": pl.String}), ignore_extra_cols=True)

    def travel_time_f_helper(struct):
        return travel_time_f(struct["orig_loc_id"], struct["dest_loc_id"])

    if isinstance(travel_time_f, float):
        return edge_df.with_columns(travel_time_min=travel_time_f)
    elif isinstance(travel_time_f, pl.Expr):
        return edge_df.with_columns(travel_time_min=travel_time_f)
    elif isinstance(travel_time_f, pl.DataFrame):
        check_schema(travel_time_f, WALK_EDGE_LIST_SCHEMA)
        return edge_df.join(travel_time_f, on=["orig_loc_id", "dest_loc_id"], how="left")
    elif isinstance(travel_time_f, TravelTimeCalculator):
        return travel_time_f.add_travel_times(edge_df)
    elif callable(travel_time_f):
        return edge_df.with_columns(input=travel_time_f).with_columns(
            travel_time_min=pl.col("input").map_elements(travel_time_f_helper, return_dtype=pl.Float64)
        )

    raise ValueError(f"Invalid travel_time_f of type {type(travel_time_f)}")


def _find_neighbours(row_geometry: Polygon, sindex: SpatialIndex, geometries: gpd.GeoSeries):
    candidates = geometries.iloc[sindex.intersection(row_geometry.bounds)]
    true_neighbours = candidates[candidates.intersects(row_geometry)].index

    return true_neighbours


def _compute_neighbour_loc_ids(planar_locations_gdf: gpd.GeoDataFrame, exclude_self: bool = True) -> pl.DataFrame:
    check_schema(planar_locations_gdf, {"loc_id": "object", "geometry": "geometry"}, ignore_extra_cols=True)
    check_geometry_shapes(planar_locations_gdf.geometry, "Polygon", "MultiPolygon")

    sindex = planar_locations_gdf.sindex
    geometries = planar_locations_gdf.set_index("loc_id").geometry

    neighbours = (
        geometries.apply(_find_neighbours, sindex=sindex, geometries=geometries)
        .rename("neighbour")
        .explode()
        .reset_index()
    )

    if exclude_self:
        neighbours = neighbours[neighbours["loc_id"] != neighbours["neighbour"]]

    return pl.DataFrame(neighbours)
