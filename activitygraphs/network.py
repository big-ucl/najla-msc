import pickle
import shutil
from abc import ABC
from collections.abc import Callable, Iterable
from enum import StrEnum
from pathlib import Path
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
    PTNodeType,
)
from activitygraphs.config import GenevaDataConfig
from activitygraphs.routing import TravelTimeCalculator
from activitygraphs.utils import (
    check_geometry_shapes,
    check_schema,
    convert_locations_to_point_geometry,
    extract_unique_loc_ids,
)

NA_LON, NA_LAT = 6.1709475192397605, 46.24348817355701
NA, NA_SOURCE, NA_SINK = "NA", "NA_SOURCE", "NA_SINK"

type PTLayerBuilder = Callable[[pl.DataFrame, PTNodeType], tuple[pl.DataFrame, pl.DataFrame]]
type TravelTimeFactory = float | pl.Expr | pl.DataFrame | TravelTimeCalculator | Callable[[str, str], float]


class NetworkData(ABC):
    user_journeys_df: pl.DataFrame
    locations_gdf: gpd.GeoDataFrame


class LayerType(StrEnum):
    PUBLIC_TRANSPORT = "public_transport"
    POINT = "point"
    PLANAR = "planar"
    NA = "na"

    def __repr__(self):
        return self.name


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

    @classmethod
    def load(cls, layer_dir: Path) -> Self:
        with open(layer_dir / "attrs.pickle", "rb") as f:
            attrs = pickle.load(f)

        edge_list = pl.read_parquet(layer_dir / "edge_df.parquet")

        return cls(attrs["name"], attrs["type"], attrs["loc_ids"], edge_list)

    def save(self, layers_dir: Path):
        layer_dir = self._dir(layers_dir)
        layer_dir.mkdir(parents=True, exist_ok=True)

        attrs = {"name": self.name, "type": self.type, "loc_ids": self.loc_ids}

        with open(layer_dir / "attrs.pickle", "wb") as f:
            # noinspection PyTypeChecker
            pickle.dump(attrs, f)

        self.edge_list.write_parquet(layer_dir / "edge_df.parquet")

    def _dir(self, layers_dir: Path) -> Path:
        return layers_dir / f"{self.type.name}-{self.name}"


class PTLayer(Layer):
    def __init__(
        self,
        name: str,
        loc_ids: Iterable[str],
        node_type: PTNodeType,
        pt_edge_df: pl.DataFrame,
        transfer_edge_df: pl.DataFrame,
    ):
        edge_cols = EDGE_LIST_SCHEMA.keys()
        edge_df = pl.concat([pt_edge_df.select(*edge_cols), transfer_edge_df.select(*edge_cols)])

        super().__init__(name, LayerType.PUBLIC_TRANSPORT, loc_ids, edge_df)

        self.node_type = node_type
        self.pt_edge_df = check_schema(pt_edge_df, PT_EDGE_LIST_SCHEMA)
        self.transfer_edge_df = check_schema(transfer_edge_df, TRANSFER_EDGE_LIST_SCHEMA)

    @classmethod
    def load(cls, layer_dir) -> Self:
        with open(layer_dir / "attrs.pickle", "rb") as f:
            attrs = pickle.load(f)

        with open(layer_dir / "pt_attrs.pickle", "rb") as f:
            pt_attrs = pickle.load(f)

        pt_edge_df = pl.read_parquet(layer_dir / "pt_edge_df.parquet", schema=PT_EDGE_LIST_SCHEMA)
        transfer_edge_df = pl.read_parquet(layer_dir / "transfer_edge_df.parquet", schema=TRANSFER_EDGE_LIST_SCHEMA)

        return cls(attrs["name"], attrs["loc_ids"], pt_attrs["node_type"], pt_edge_df, transfer_edge_df)

    def save(self, layers_dir: Path):
        super().save(layers_dir)
        layer_dir = self._dir(layers_dir)

        with open(layer_dir / "pt_attrs.pickle", "wb") as f:
            pickle.dump({"node_type": self.node_type}, f)

        self.pt_edge_df.write_parquet(layer_dir / "pt_edge_df.parquet")
        self.transfer_edge_df.write_parquet(layer_dir / "transfer_edge_df.parquet")


def load_layer(layer_dir: Path) -> Layer:
    with open(layer_dir / "attrs.pickle", "rb") as f:
        attrs = pickle.load(f)

    layer_type: LayerType = attrs["type"]

    match layer_type:
        case LayerType.PUBLIC_TRANSPORT:
            return PTLayer.load(layer_dir)
        case LayerType.NA | LayerType.PLANAR | LayerType.POINT:
            return Layer.load(layer_dir)


class Network:
    def __init__(
        self,
        locations_gdf: gpd.GeoDataFrame,
        user_journeys_df: pl.DataFrame,
        layers: dict[str, Layer],
        links: dict[tuple[str, str], pl.DataFrame],
    ):
        self._locations_gdf = check_schema(locations_gdf, LOCATIONS_SCHEMA)
        self._user_journeys = check_schema(user_journeys_df, USER_JOURNEY_SCHEMA)
        self._layers = layers.copy()
        self._links = links.copy()

    @classmethod
    def empty_network(cls, network_data: NetworkData) -> Self:
        return cls(network_data.locations_gdf, network_data.user_journeys_df, {}, {})

    @property
    def locations_gdf(self) -> gpd.GeoDataFrame:
        return self._locations_gdf.copy()

    @property
    def locations_df(self) -> pl.DataFrame:
        return utils.gdf_to_polars(self._locations_gdf)

    @property
    def location_types(self) -> list[str]:
        return self._locations_gdf["type"].unique().tolist()

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
        pt_layer_builder: PTLayerBuilder | None = None,
        pt_edge_df: pl.DataFrame | None = None,
        transfer_edge_df: pl.DataFrame | None = None,
        pt_node_type: PTNodeType = PTNodeType.ONE_PER_ROUTE,
    ) -> Self:
        if pt_edge_df is None and transfer_edge_df is None and pt_layer_builder is not None:
            pt_edge_df, transfer_edge_df = pt_layer_builder(self.locations_df, pt_node_type)
        elif pt_layer_builder is None:
            raise ValueError("No PTLayerBuilder provided, cannot build edges.")
        elif pt_edge_df is None or transfer_edge_df is None:
            raise ValueError("Arguments `pt_edge_df` and `transfer_edge_df` must be both None or both DataFrames")

        if pt_node_type == PTNodeType.ONE_PER_STOP:
            _check_only_one_route_per_node(pt_edge_df, transfer_edge_df)

        if loc_ids is None:
            loc_ids = extract_unique_loc_ids(pt_edge_df, transfer_edge_df)

        loc_ids = self._check_layer_loc_ids(name, loc_ids)

        self._layers[name] = PTLayer(name, loc_ids, pt_node_type, pt_edge_df, transfer_edge_df)
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
        ]).reset_index(drop=True)

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

    @classmethod
    def _dir(cls, cfg: GenevaDataConfig, project_root: Path | None = None, name: str | None = None) -> Path:
        project_root = project_root if project_root is not None else Path(".")
        suffix = "" if name is None else f"-{name}"
        data_dir = project_root / cfg.paths.processed / f"{cls.__name__}{suffix}"

        return data_dir

    @classmethod
    def exists_on_disk(cls, cfg: GenevaDataConfig, project_root: Path | None = None, name: str | None = None):
        return cls._dir(cfg, project_root, name).exists()

    @classmethod
    def load(cls, cfg: GenevaDataConfig, project_root: Path | None = None, name: str | None = None) -> Self:
        data_dir = cls._dir(cfg, project_root, name)

        if data_dir.exists():
            layers = cls._load_layers(data_dir / "layers")
            links = cls._load_links(data_dir / "links")

            user_journeys_df = pl.read_parquet(data_dir / "user_journeys_df.parquet")
            locations_gdf = gpd.read_parquet(data_dir / "locations_gdf.parquet")

            return cls(locations_gdf, user_journeys_df, layers, links)
        else:
            raise ValueError(f"Data directory does not exist: {data_dir}")

    @classmethod
    def _load_layers(cls, layers_dir: Path) -> dict[str, Layer]:
        layers = {}
        for layer_dir in layers_dir.iterdir():
            layer = load_layer(layer_dir)
            layers[layer.name] = layer

        return layers

    @classmethod
    def _load_links(cls, links_dir: Path) -> dict[tuple[str, str], pl.DataFrame]:
        links = {}
        for link_dir in links_dir.iterdir():
            with open(link_dir / "attrs.pickle", "rb") as f:
                attrs = pickle.load(f)

            link_edge_df = pl.read_parquet(link_dir / "link_edges.parquet", schema=LINK_EDGE_LIST_SCHEMA)
            links[(attrs["lower"], attrs["upper"])] = link_edge_df

        return links

    def save(
        self,
        cfg: GenevaDataConfig,
        project_root: Path | None = None,
        name: str | None = None,
        can_overwrite: bool = False,
    ):
        data_dir = self._dir(cfg, project_root, name)
        layers_dir = data_dir / "layers"
        links_dir = data_dir / "links"

        data_dir.mkdir(parents=True, exist_ok=can_overwrite)

        if can_overwrite and layers_dir.exists():
            shutil.rmtree(layers_dir)

        if can_overwrite and links_dir.exists():
            shutil.rmtree(links_dir)

        layers_dir.mkdir(parents=True, exist_ok=can_overwrite)
        links_dir.mkdir(parents=True, exist_ok=can_overwrite)

        for _, layer in self._layers.items():
            layer.save(layers_dir)

        for (lower, upper), link_edge_df in self._links.items():
            link_dir = links_dir / f"{lower}-{upper}"
            link_dir.mkdir(parents=True, exist_ok=True)

            attrs = {"lower": lower, "upper": upper}
            with open(link_dir / "attrs.pickle", "wb") as f:
                pickle.dump(attrs, f)

            link_edge_df.write_parquet(link_dir / "link_edges.parquet")

        self._user_journeys.write_parquet(data_dir / "user_journeys_df.parquet")
        self._locations_gdf.to_parquet(data_dir / "locations_gdf.parquet")


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


def _check_only_one_route_per_node(pt_edge_df: pl.DataFrame, transfer_edge_df: pl.DataFrame):
    def check_df(name: str, df: pl.DataFrame, loc_id_col: str, route_id_col: str):
        counts = (
            df
            .select(loc_id_col, route_id_col)
            .unique()
            .group_by(loc_id_col)
            .agg(count=pl.len(), route_id=pl.col(route_id_col))
        )
        mismatches = counts.filter(pl.col("count") != 1)

        if len(mismatches) > 0:
            raise ValueError(
                f"Some nodes ({loc_id_col}, {route_id_col}) in {name} have more than one route: \n{mismatches}"
            )

    check_df("pt_edge_df", pt_edge_df, "orig_loc_id", "route_id")
    check_df("pt_edge_df", pt_edge_df, "dest_loc_id", "route_id")
    check_df("transfer_edge_df", transfer_edge_df, "orig_loc_id", "orig_route_id")
    check_df("transfer_edge_df", transfer_edge_df, "dest_loc_id", "dest_route_id")


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
        geometries
        .apply(_find_neighbours, sindex=sindex, geometries=geometries)
        .rename("neighbour")
        .explode()
        .reset_index()
    )

    if exclude_self:
        neighbours = neighbours[neighbours["loc_id"] != neighbours["neighbour"]]

    return pl.DataFrame(neighbours)
