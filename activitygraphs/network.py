from abc import ABC
from collections.abc import Callable, Iterable
from enum import Enum, auto
from typing import Literal, Self

import geopandas as gpd
import pandas as pd
import polars as pl
from geopandas.sindex import SpatialIndex
from shapely.geometry.polygon import Polygon

from activitygraphs import utils
from activitygraphs.base import (
    CRS,
    EDGE_LIST_SCHEMA,
    LINK_EDGE_LIST_SCHEMA,
    LOCATIONS_SCHEMA,
    PT_EDGE_LIST_SCHEMA,
    TRANSFER_EDGE_LIST_SCHEMA,
    USER_JOURNEY_SCHEMA,
    WALK_EDGE_LIST_SCHEMA,
)
from activitygraphs.routing import TravelTimeCalculator
from activitygraphs.utils import (
    check_geometry_shapes,
    check_schema,
    convert_locations_to_point_geometry,
    extract_unique_loc_ids,
)

NA_LON, NA_LAT = 6.1709475192397605, 46.24348817355701
NA, NA_SOURCE, NA_SINK = "NA", "NA_SOURCE", "NA_SINK"

PTNetworkBuilder = Callable[[pl.DataFrame], tuple[pl.DataFrame, pl.DataFrame]]
TravelTimeFactory = float | pl.Expr | pl.DataFrame | TravelTimeCalculator | Callable[[str, str], float]


class NetworkData(ABC):
    user_journeys_df: pl.DataFrame
    locations_gdf: gpd.GeoDataFrame


class LayerType(Enum):
    PUBLIC_TRANSPORT = auto()
    POINT = auto()
    PLANAR = auto()
    NA = auto()


class Layer:
    def __init__(self, name: str, layer_type: LayerType, loc_ids: Iterable[str], edge_list: pl.DataFrame | None):
        edge_list = edge_list if edge_list is not None else pl.DataFrame(schema=EDGE_LIST_SCHEMA)
        check_schema(edge_list, EDGE_LIST_SCHEMA, ignore_extra_cols=True)

        self.name = name
        self.type = layer_type
        self.loc_ids = list(loc_ids)

        mismatches = edge_list.filter(
            ~pl.col("orig_loc_id").is_in(self.loc_ids) | ~pl.col("dest_loc_id").is_in(self.loc_ids)
        )

        if len(mismatches) > 0:
            mismatches_fmt = edge_list.select(pl.format("({}, {})", "orig_loc_id", "dest_loc_id"))
            mismatches_str = ",".join(mismatches_fmt.to_series())
            raise ValueError(f"Found edges that connect outside of layer:\n*******{mismatches_str}\n*********")

        self.edge_list = edge_list

    def __repr__(self):
        return f"Layer({self.name}, type={self.type.name})"


class PTLayer(Layer):
    def __init__(self, name: str, loc_ids: Iterable[str], pt_edge_df: pl.DataFrame, transfer_edge_df: pl.DataFrame):
        edge_cols = EDGE_LIST_SCHEMA.keys()
        edge_df = pl.concat([pt_edge_df.select(*edge_cols), transfer_edge_df.select(*edge_cols)])

        super().__init__(name, LayerType.PUBLIC_TRANSPORT, loc_ids, edge_df)

        self.pt_edge_df = check_schema(pt_edge_df, PT_EDGE_LIST_SCHEMA)
        self.transfer_edge_df = check_schema(transfer_edge_df, TRANSFER_EDGE_LIST_SCHEMA)


class Network:
    def __init__(self, network_data: NetworkData):
        self._locations_gdf = check_schema(network_data.locations_gdf, LOCATIONS_SCHEMA)
        self._user_journeys = check_schema(network_data.user_journeys_df, USER_JOURNEY_SCHEMA)
        self._layers: dict[str, Layer] = {}
        self._links: dict[tuple[str, str], pl.DataFrame] = {}

    @property
    def locations_gdf(self) -> gpd.GeoDataFrame:
        return self._locations_gdf.copy()

    @property
    def locations_df(self) -> pl.DataFrame:
        return utils.gdf_to_polars(self._locations_gdf)

    @property
    def layers(self):
        return {name: layer.type for name, layer in self._layers.items()}

    @property
    def links(self):
        return [(name1, name2) for (name1, name2), _ in self._links.items()]

    @property
    def has_separate_na_source_sink(self):
        num_na_locations = len(self._locations_gdf.query("type == 'na'"))

        if num_na_locations not in [1, 2]:
            raise ValueError(f"Invalid number of NA locaitons {num_na_locations}")

        return num_na_locations == 2

    def __getitem__(self, name: str) -> Layer:
        if name not in self._layers:
            raise ValueError(f"Cannot find layer `{name}`")

        return self._layers[name]

    def get_layer_locations(self, layer_name: str) -> gpd.GeoDataFrame:
        loc_ids = self[layer_name].loc_ids
        return self._locations_gdf[self._locations_gdf["loc_id"].isin(loc_ids)]

    def get_pt_layer(self, name: str) -> PTLayer:
        layer = self[name]

        if not isinstance(layer, PTLayer):
            raise ValueError(f"Layer `{name}` is not a PT layer")

        return layer

    def get_links(self, lower_layer: str, upper_layer: str) -> pl.DataFrame:
        return self._links[(lower_layer, upper_layer)]

    def add_layer(
        self,
        name: str,
        layer_type: LayerType,
        loc_ids: str | gpd.GeoDataFrame | Iterable[str],
        edge_list: pl.DataFrame | None,
    ) -> Self:
        loc_ids = self._check_layer_loc_ids(name, loc_ids)
        self._layers[name] = Layer(name, layer_type, loc_ids, edge_list)
        return self

    def add_pt_layer(
        self,
        name: str,
        loc_ids: str | gpd.GeoDataFrame | Iterable[str] | None = None,
        pt_network_builder: PTNetworkBuilder | None = None,
        pt_edge_df: pl.DataFrame | None = None,
        transfer_edge_df: pl.DataFrame | None = None,
    ) -> Self:
        if pt_edge_df is None and transfer_edge_df is None and pt_network_builder is not None:
            pt_edge_df, transfer_edge_df = pt_network_builder(self.locations_df)
        elif pt_network_builder is None:
            raise ValueError("No PTNetworkBuilder provided, cannot build edges.")
        elif pt_edge_df is None or transfer_edge_df is None:
            raise ValueError("Arguments `pt_edge_df` and `transfer_edge_df` must be both None or both DataFrames")

        if loc_ids is None:
            loc_ids = extract_unique_loc_ids(pt_edge_df, transfer_edge_df)

        loc_ids = self._check_layer_loc_ids(name, loc_ids)

        self._layers[name] = PTLayer(name, loc_ids, pt_edge_df, transfer_edge_df)
        return self

    def add_planar_layer(
        self, name: str, loc_ids: str | gpd.GeoDataFrame | Iterable[str], travel_time_f: TravelTimeFactory | None = None
    ) -> Self:
        loc_ids = self._check_layer_loc_ids(name, loc_ids)
        planar_locations_gdf = self._locations_gdf[self.locations_gdf["loc_id"].isin(loc_ids)]
        planar_edges = build_planar_edges(planar_locations_gdf, travel_time_f) if travel_time_f is not None else None

        return self.add_layer(name, LayerType.PLANAR, loc_ids, planar_edges)

    def add_na_layer(self, separate_in_out_nodes: bool = False, na_coords: tuple[float, float] | None = None) -> Self:
        if "na" in self._layers:
            raise ValueError("NA Layer is already in the network. Connect it using `connect_na_layer`")

        na_coords = (NA_LON, NA_LAT) if na_coords is None else na_coords

        n_duplicates = 2 if separate_in_out_nodes else 1
        loc_ids = [NA_SOURCE, NA_SINK] if separate_in_out_nodes else NA
        empty_edge_list = pl.DataFrame(schema=EDGE_LIST_SCHEMA)

        na_locations = gpd.GeoDataFrame(
            {
                "loc_id": loc_ids,
                "loc_name": loc_ids,
                "type": ["na"] * n_duplicates,
                "lon": [na_coords[0]] * n_duplicates,
                "lat": [na_coords[1]] * n_duplicates,
            },
            crs=CRS,
            geometry=gpd.points_from_xy([na_coords[0]] * n_duplicates, [na_coords[1]] * n_duplicates, crs=CRS),
        )

        self._locations_gdf = pd.concat([
            na_locations,
            self._locations_gdf.query("type != 'na'"),
        ])

        return self.add_layer("na", LayerType.NA, na_locations, empty_edge_list)

    def _check_layer_loc_ids(self, name: str, loc_ids: str | gpd.GeoDataFrame | Iterable[str]) -> list[str]:
        if name in self._layers:
            raise ValueError(f"Layer `{name}` is already in the network.")

        if isinstance(loc_ids, str):
            loc_ids = self._locations_gdf[self._locations_gdf["type"] == loc_ids]["loc_id"].tolist()
        elif isinstance(loc_ids, gpd.GeoDataFrame):
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

    def connect_na_layer(self, others: str | list[str], travel_time_f: TravelTimeFactory):
        others = [others] if isinstance(others, str) else others

        for layer in others:
            na_source = NA_SOURCE if self.has_separate_na_source_sink else NA
            na_sink = NA_SINK if self.has_separate_na_source_sink else NA

            layer_locations = self.get_layer_locations(layer)
            na_source_link_edges = build_na_link_edges(na_source, layer_locations, travel_time_f, "na_to_loc")
            na_sink_link_edges = build_na_link_edges(na_sink, layer_locations, travel_time_f, "loc_to_na")

            self._links[("na", layer)] = pl.concat([na_source_link_edges, na_sink_link_edges])

        return self

    def __repr__(self):
        layers = "\n".join(f"\t\t{n}: {t}" for n, t in self.layers.items())
        links = "\n".join(f"\t\t{low} |--> {up}" for low, up in self.links)

        return f"""Network(\n\tlayers=(\n{layers}\n\t), links=(\n{links}\n\t)\n)"""


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


def build_na_link_edges(
    na_loc_id: str,
    locations_gdf: gpd.GeoDataFrame,
    travel_time_f: TravelTimeFactory,
    direction: Literal["na_to_loc", "loc_to_na"],
) -> pl.DataFrame:
    if direction not in ["na_to_loc", "loc_to_na"]:
        raise ValueError(f"Invalid direction {direction}, must be 'na_to_loc' or 'loc_to_na'")

    origins = na_loc_id if direction == "na_to_loc" else locations_gdf["loc_id"]
    destinations = locations_gdf["loc_id"] if direction == "na_to_loc" else na_loc_id

    edges = pl.DataFrame({
        "orig_loc_id": origins,
        "dest_loc_id": destinations,
    })

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
