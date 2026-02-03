import math
from collections.abc import Generator, Hashable, Sequence
from enum import Enum
from typing import Callable, Mapping, Type

import geopandas as gpd
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from geopandas import GeoDataFrame
from torch_geometric.data import HeteroData

from activitygraphs.base import USER_JOURNEY_SCHEMA, Mode
from activitygraphs.data.gtfs import PARENT_STOP_ROUTE_ID
from activitygraphs.network import LayerType, Network
from activitygraphs.utils import check_schema

type Encoder = Callable[[pd.Series | pl.Series], torch.Tensor]

type LocID = str | tuple[str, str]
type NodeMapping = Mapping[LocID, int]
type LocIDMapping = Mapping[LocID, tuple[LayerType, int]]

type NodeProcessor = Callable[
    [Network, Sequence[str], Mapping[str, Encoder] | None], tuple[NodeMapping, torch.Tensor | None]
]
type EdgeProcessor = Callable[
    [Network, Sequence[str], NodeMapping, Mapping[str, Encoder] | None],
    dict[str, tuple[torch.Tensor, torch.Tensor]],
]

EMBEDDING_DIM = 3
LAYER_NAME_COL = "layer_name"


class EnumEncoder:
    def __init__(self, enum_cls: Type[Enum] | list[Hashable], embedding_dim: int):
        self._enum_cls = enum_cls
        self.mapping = {enum: index for index, enum in enumerate(enum_cls)}

        self.num_embeddings = len(self.mapping)
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)

    def __call__(self, enum_series: pd.Series | pl.Series) -> torch.Tensor:
        enum_series = enum_series if isinstance(enum_series, pd.Series) else enum_series.to_pandas()

        indices = enum_series.map(self.mapping).to_numpy()
        if indices.dtype != "int":
            raise ValueError("Conversion to indices failed, some values not in mapping ")

        device = self.embedding.weight.device
        indices = torch.tensor(indices, device=device, dtype=torch.long)
        return self.embedding(indices)


class PTStopEncoder:
    pass


class RouteEncoder:
    pass


class TimeOfDayEncoder:
    # noinspection PyUnresolvedReferences
    def __call__(self, tod_series: pd.Series | pl.Series) -> torch.Tensor:
        tod_series = tod_series if isinstance(tod_series, pl.Series) else pl.from_pandas(tod_series)
        total_seconds = (
            3600 * tod_series.dt.hour().cast(pl.UInt32)
            + 60 * tod_series.dt.minute().cast(pl.UInt32)
            + tod_series.dt.second().cast(pl.UInt32)
        )

        seconds_in_day = 24 * 60 * 60
        sines = (2 * math.pi * total_seconds / seconds_in_day).sin()
        cosines = (2 * math.pi * total_seconds / seconds_in_day).cos()

        return torch.stack([sines.to_torch(), cosines.to_torch()], dim=1)


class GeometryEncoder:
    def __call__(self, geometry_series: pd.Series | pl.Series) -> torch.Tensor:
        if not isinstance(geometry_series, pd.Series) or geometry_series.dtype != "geometry":
            raise ValueError(
                f"Geometry must be of type gpd.Series with dtype=geometry,"
                f"found {type(geometry_series)} with dtype={geometry_series.dtype}"
            )

        geometry_series = gpd.GeoSeries(geometry_series)
        planar_crs = geometry_series.estimate_utm_crs()
        area = torch.tensor(geometry_series.to_crs(planar_crs).area.to_numpy())

        return torch.stack([area], dim=1)


def network_to_pyg(network: Network, embedding_dim=EMBEDDING_DIM) -> HeteroData:
    data = HeteroData()

    layer_name_encoder = EnumEncoder(network.layers.keys(), embedding_dim)
    loc_type_encoder = EnumEncoder(network.location_types, embedding_dim)
    geometry_encoder = GeometryEncoder()
    node_encoders = {"type": loc_type_encoder, LAYER_NAME_COL: layer_name_encoder, "geometry": geometry_encoder}

    route_mode_encoder = EnumEncoder(Mode, embedding_dim)
    time_of_day_encoder = TimeOfDayEncoder()
    edge_encoders = {
        "route_mode": route_mode_encoder,
        "first_departure_time": time_of_day_encoder,
        "last_departure_time": time_of_day_encoder,
    }

    node_mappings: dict[LayerType, NodeMapping] = {}
    for layer_type in set(network.layers.values()):
        layer_mapping = _process_layers_by_type(data, network, layer_type, node_encoders, edge_encoders)
        node_mappings[layer_type] = layer_mapping

    link_types = set((network[l1].type, network[l2].type) for l1, l2 in network.links)
    for lower_type, upper_type in link_types:
        relations = _process_layer_links(network, lower_type, upper_type, node_mappings)

        for (lower, relation_name, upper), (edge_index, edge_attrs) in relations.items():
            data[lower, relation_name, upper].edge_index = edge_index
            data[lower, relation_name, upper].edge_attrs = edge_attrs

    data.graph_metadata = _create_graph_metadata(node_mappings)

    return data


def _create_graph_metadata(
    node_mappings: dict[LayerType, NodeMapping],
) -> dict[str, LocIDMapping]:
    full_loc_id_mapping = {}

    for layer_type, mapping in node_mappings.items():
        for loc_id, node_index in mapping.items():
            full_loc_id_mapping[loc_id] = (layer_type, node_index)

            # If PT `parent` node, add a direct loc_id -> node mapping
            if isinstance(loc_id, tuple) and loc_id[1] == PARENT_STOP_ROUTE_ID:
                full_loc_id_mapping[loc_id[0]] = (layer_type, node_index)

    loc_id_mapping = {k: v for k, v in full_loc_id_mapping.items() if isinstance(k, str)}

    return {"full_loc_id_mapping": full_loc_id_mapping, "loc_id_mapping": loc_id_mapping}


def add_labels_to_pyg(
    data: HeteroData, user_journeys_df: pl.DataFrame, separate_na_source_sink: bool
) -> Generator[HeteroData]:
    check_schema(user_journeys_df, USER_JOURNEY_SCHEMA)

    loc_id_mapping: LocIDMapping = data.graph_metadata["loc_id_mapping"]
    loc_id_to_layer_type = {k: t for k, (t, _) in loc_id_mapping.items()}
    loc_id_to_node_index = {k: i for k, (_, i) in loc_id_mapping.items()}

    user_attributes = _build_user_attributes(user_journeys_df, separate_na_source_sink)
    visited_locations = _build_visited_locations(user_journeys_df, separate_na_source_sink)
    visited_locations = visited_locations.filter(pl.col("user_id").is_in(user_attributes["user_id"].implode()))

    for (user_id,), user_visited_locations in visited_locations.group_by("user_id"):
        user_attrs = user_attributes.filter(user_id=user_id)

        data = data.clone()
        data.graph_metadata["user_id"] = user_id

        _add_location_indicator_features(data, user_attrs, "home_loc_id", loc_id_to_layer_type, loc_id_to_node_index)
        _add_visited_location_labels(data, user_visited_locations, loc_id_to_layer_type, loc_id_to_node_index)

        yield data


def _build_user_attributes(user_journeys_df: pl.DataFrame, separate_na_source_sink: bool) -> pl.DataFrame:
    home_locations = _extract_locations(
        user_journeys_df, "od_lieu_domicile", separate_na_source_sink, is_unique=True, loc_col="home_loc_id"
    )
    work_locations = _extract_locations(
        user_journeys_df, "od_lieu_travail", separate_na_source_sink, is_unique=False, loc_col="work_loc_ids"
    )
    study_locations = _extract_locations(
        user_journeys_df, "od_lieu_etude", separate_na_source_sink, is_unique=False, loc_col="study_loc_ids"
    )

    return (
        user_journeys_df
        .select(pl.col("user_id").unique())
        .join(home_locations, on="user_id", how="inner")
        .join(work_locations, on="user_id", how="left")
        .join(study_locations, on="user_id", how="left")
    )


def _extract_locations(
    user_journeys_df: pl.DataFrame,
    purpose: str,
    separate_na_source_sink: bool,
    is_unique: bool = True,
    loc_col: str = "loc_id",
) -> pl.DataFrame:
    departure_locs = user_journeys_df.filter(
        pl.col("dep_purpose") == purpose, pl.col("leg_id") == pl.col("leg_id").min().over("journey_id")
    ).select("user_id", pl.col("dep_loc_id").alias(loc_col))
    arrival_locs = user_journeys_df.filter(
        pl.col("arr_purpose") == purpose, pl.col("leg_id") == pl.col("leg_id").max().over("journey_id")
    ).select("user_id", pl.col("arr_loc_id").alias(loc_col))

    if separate_na_source_sink:
        departure_locs = departure_locs.with_columns(pl.col(loc_col).replace({"NA": "NA_SOURCE"}))
        arrival_locs = arrival_locs.with_columns(pl.col(loc_col).replace({"NA": "NA_SINK"}))

    locations = pl.concat([departure_locs, arrival_locs]).group_by("user_id").agg(pl.col(loc_col).unique())

    if is_unique:
        locations = locations.filter(pl.col(loc_col).list.len() <= 1).with_columns(pl.col(loc_col).list.item())

    return locations


def _build_visited_locations(user_journeys_df: pl.DataFrame, separate_na_source_sink: bool) -> pl.DataFrame:
    def replace_na_with(col: str, value: str):
        return pl.col(col).replace({"NA": value}) if separate_na_source_sink else pl.col(col)

    origin_and_intermediate = user_journeys_df.select(
        "user_id",
        "journey_id",
        "leg_id",
        loc_id=replace_na_with("dep_loc_id", "NA_SOURCE"),
        is_origin=pl.col("leg_id") == 0,
        is_destination=False,
        is_endpoint=pl.col("leg_id") == 0,
    )

    destinations = (
        user_journeys_df
        .group_by("user_id", "journey_id")
        .agg(pl.all().last())
        .select(
            "user_id",
            "journey_id",
            "leg_id",
            loc_id=replace_na_with("arr_loc_id", "NA_SINK"),
            is_origin=False,
            is_destination=True,
            is_endpoint=True,
        )
    )

    visited_locations = (
        pl
        .concat([origin_and_intermediate, destinations])
        .unique(maintain_order=True)
        .sort("user_id", "journey_id", "leg_id", "is_destination", ~pl.col("is_origin"))
        .with_columns(pl.col("leg_id").rank(method="ordinal").over("user_id", "journey_id") - 1)
        .rename({"leg_id": "seq_num"})
    )

    return visited_locations


def _add_location_indicator_features(
    data: HeteroData,
    user_attributes: pl.DataFrame,
    loc_id_column: str,
    loc_id_to_layer_type: dict[str, LayerType],
    loc_id_to_node_index: dict[str, int],
) -> None:
    user_ids = user_attributes["user_id"].unique()
    assert len(user_ids) == 1, f"Only a single user ID is allowed per graph, found {user_ids}"

    visited_locations = user_attributes.select(pl.col(loc_id_column).explode())
    visited_indices = visited_locations.select(
        loc_id_column,
        layer_type=pl.col(loc_id_column).replace_strict(loc_id_to_layer_type),
        node_index=pl.col(loc_id_column).replace_strict(loc_id_to_node_index),
    )

    for layer_type in data.node_types:
        df = visited_indices.filter(layer_type=layer_type)

        x = torch.zeros(data[layer_type].num_nodes)
        x[df["node_index"].to_numpy()] = 1

        data[layer_type].x = torch.cat([data[layer_type].x, x.unsqueeze(1)], dim=1)


def _add_visited_location_labels(
    data: HeteroData,
    user_visited_locations: pl.DataFrame,
    loc_id_to_layer_type: dict[str, LayerType],
    loc_id_to_node_index: dict[str, int],
) -> None:
    user_ids = user_visited_locations["user_id"].unique()
    assert len(user_ids) == 1, f"Only a single user ID is allowed per graph, found {user_ids}"

    visited_indices = user_visited_locations.select(
        "user_id",
        "loc_id",
        layer_type=pl.col("loc_id").replace_strict(loc_id_to_layer_type),
        node_index=pl.col("loc_id").replace_strict(loc_id_to_node_index),
    )

    for layer_type in data.node_types:
        df = visited_indices.filter(layer_type=layer_type)

        y = torch.zeros(data[layer_type].num_nodes)
        y[df["node_index"].to_numpy()] = 1

        data[layer_type].y = y


# =====================================
# Layer processing
# =====================================


def _process_layers_by_type(
    data: HeteroData,
    network: Network,
    layer_type: LayerType,
    node_encoders: Mapping[str, Encoder] | None = None,
    edge_encoders: Mapping[str, Encoder] | None = None,
) -> Mapping[LayerType | tuple[str, str], int]:
    assert not data[layer_type]

    node_processor = _select_node_processor(layer_type)
    edge_processor = _select_edge_processor(layer_type)

    layer_names = [name for name, l_type in network.layers.items() if l_type == layer_type]
    node_mapping, node_x = node_processor(network, layer_names, node_encoders)
    internal_edges = edge_processor(network, layer_names, node_mapping, edge_encoders)

    data[layer_type].x = node_x
    data[layer_type].num_nodes = len(node_mapping)

    for relation_name, (edge_index, edge_attrs) in internal_edges.items():
        data[layer_type, relation_name, layer_type].edge_index = edge_index
        data[layer_type, relation_name, layer_type].edge_attrs = edge_attrs

    return node_mapping


def _select_node_processor(layer_type: LayerType) -> NodeProcessor:
    match layer_type:
        case LayerType.PUBLIC_TRANSPORT:
            return _process_pt_layer_nodes
        case _:
            return _process_base_layer_nodes


def _select_edge_processor(layer_type: LayerType) -> EdgeProcessor:
    match layer_type:
        case LayerType.PUBLIC_TRANSPORT:
            return _process_pt_layer_edges
        case _:
            return _process_base_layer_edges


# =====================================
# Node processing
# =====================================


def _process_base_layer_nodes(
    network: Network, layer_names: Sequence[str], node_encoders: Mapping[str, Encoder] | None = None
) -> tuple[dict[str, int], torch.Tensor | None]:
    layer_locations_gdf = _combine_layer_locations(network, layer_names)

    # Map `loc_id` to indices in PyG graph and create node features
    node_mapping = layer_locations_gdf.reset_index(drop=True).reset_index().set_index("loc_id")["index"].to_dict()
    x = _encode_node_features(layer_locations_gdf, node_encoders)

    return node_mapping, x


def _process_pt_layer_nodes(
    network: Network, layer_names: Sequence[str], node_encoders: Mapping[str, Encoder] | None = None
) -> tuple[dict[tuple[str, str], int], torch.Tensor | None]:
    layer_locations_gdf = _combine_layer_locations(network, layer_names)
    pt_layers = [network.get_pt_layer(name) for name in layer_names]

    # Find all unique (loc_id, route_id) pairs and create a dataframe with their info.
    pt_origins = pl.concat(layer.pt_edge_df.select(loc_id="orig_loc_id", route_id="route_id") for layer in pt_layers)
    tr_origins = pl.concat(
        layer.transfer_edge_df.select(loc_id="orig_loc_id", route_id="orig_route_id") for layer in pt_layers
    )
    pt_destinations = pl.concat(
        layer.pt_edge_df.select(loc_id="dest_loc_id", route_id="route_id") for layer in pt_layers
    )
    tr_destinations = pl.concat(
        layer.transfer_edge_df.select(loc_id="dest_loc_id", route_id="dest_route_id") for layer in pt_layers
    )

    pt_locations = pl.concat([pt_origins, tr_origins, pt_destinations, tr_destinations]).unique()

    # Make sure stops without routes going through them still have a (loc_id, `parent`) node
    # noinspection PyTypeChecker
    pt_locations_gdf: gpd.GeoDataFrame = pt_locations.to_pandas().merge(layer_locations_gdf, on="loc_id", how="right")
    pt_locations_gdf["route_id"] = pt_locations_gdf["route_id"].fillna(PARENT_STOP_ROUTE_ID)

    # Create mapping from (loc_id, route_id) to node index and encode node features
    node_mapping = (
        pt_locations_gdf.reset_index(drop=True).reset_index().set_index(["loc_id", "route_id"])["index"].to_dict()
    )
    x = _encode_node_features(pt_locations_gdf, node_encoders)

    return node_mapping, x


def _combine_layer_locations(network: Network, layer_names: Sequence[str]) -> gpd.GeoDataFrame:
    layer_locations = {name: network.get_layer_locations(name) for name in layer_names}

    # Combine layer locations together and add a column identifying each layer by name
    layer_names_series = pd.concat(
        (pd.Series(name).repeat(len(gdf)) for name, gdf in layer_locations.items()), ignore_index=True
    )

    # noinspection PyTypeChecker
    layer_locations_gdf: gpd.GeoDataFrame = pd.concat(layer_locations.values(), ignore_index=True)
    layer_locations_gdf[LAYER_NAME_COL] = layer_names_series

    return layer_locations_gdf


def _encode_node_features(
    layer_locations_gdf: GeoDataFrame, encoders: Mapping[str, Encoder] | None
) -> torch.Tensor | None:
    if encoders is None:
        return None

    xs = [encoder(layer_locations_gdf[col]) for col, encoder in encoders.items() if col in layer_locations_gdf]
    x = torch.cat(xs, dim=-1)

    return x


# =====================================
# Layer edge processing (internal)
# =====================================


def _process_base_layer_edges(
    network: Network,
    layer_names: Sequence[str],
    node_mapping: Mapping[str, int],
    edge_encoders: Mapping[str, Encoder] | None,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    layers = [network[name] for name in layer_names]

    orig_index_expr = pl.col("orig_loc_id").replace_strict(node_mapping)
    dest_index_expr = pl.col("dest_loc_id").replace_strict(node_mapping)

    edge_list = pl.concat(layer.edge_list for layer in layers)

    edge_indices = edge_list.select(orig_index_expr, dest_index_expr).to_torch().T
    edge_attrs = _encode_edge_features(edge_list, ["travel_time_min"], edge_encoders)

    return {"travel": (edge_indices, edge_attrs)}


def _process_pt_layer_edges(
    network: Network,
    layer_names: Sequence[str],
    node_mapping: Mapping[LocID, int],
    edge_encoders: Mapping[str, Encoder] | None,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    layers = [network.get_pt_layer(name) for name in layer_names]

    pt_orig_idx_expr = pl.concat_list("orig_loc_id", "route_id").replace_strict(node_mapping)
    pt_dest_idx_expr = pl.concat_list("dest_loc_id", "route_id").replace_strict(node_mapping)
    tr_orig_idx_expr = pl.concat_list("orig_loc_id", "orig_route_id").replace_strict(node_mapping)
    tr_dest_idx_expr = pl.concat_list("dest_loc_id", "dest_route_id").replace_strict(node_mapping)

    pt_edge_list = pl.concat(layer.pt_edge_df for layer in layers)
    pt_edge_indices = pt_edge_list.select(pt_orig_idx_expr, pt_dest_idx_expr).to_torch().T
    pt_edge_attrs = _encode_edge_features(
        pt_edge_list,
        ["travel_time_min", "avg_dwell_time_min", "daily_trip_count", "avg_headway_min"],
        edge_encoders,
    )

    tr_edges = pl.concat(layer.transfer_edge_df for layer in layers)
    tr_edge_indices = tr_edges.select(tr_orig_idx_expr, tr_dest_idx_expr).to_torch().T
    tr_edge_attrs = _encode_edge_features(
        tr_edges,
        ["travel_time_min"],
        edge_encoders,
    )

    return {
        "pt": (pt_edge_indices, pt_edge_attrs),
        "transfer": (tr_edge_indices, tr_edge_attrs),
    }


def _encode_edge_features(
    edge_df: pl.DataFrame, numerical_features: list[str], edge_encoders: Mapping[str, Encoder] | None
) -> torch.Tensor:
    edge_attrs = edge_df.select(*numerical_features).to_torch()

    if edge_encoders:
        encoded_attrs = [encoder(edge_df[col]) for col, encoder in edge_encoders.items() if col in edge_df]
        edge_attrs = torch.cat([edge_attrs, *encoded_attrs], dim=1)

    return edge_attrs


# =====================================
# Inter-layer link processing
# =====================================


def _process_layer_links(
    network: Network, lower_type: LayerType, upper_type: LayerType, node_mapping: Mapping[str, NodeMapping]
) -> dict[tuple[str, str, str], tuple[torch.Tensor, torch.Tensor]]:
    matching_links = [
        (ll, ul) for ll, ul in network.links if network[ll].type == lower_type and network[ul].type == upper_type
    ]

    link_edge_df = pl.concat(network.get_links(ll, ul) for ll, ul in matching_links)
    lower_mapping = _node_mapping_to_dict(node_mapping[lower_type])
    upper_mapping = _node_mapping_to_dict(node_mapping[upper_type])

    upwards_edges = link_edge_df.filter(pl.col("orig_loc_id").is_in(lower_mapping))
    downwards_edges = link_edge_df.filter(~pl.col("orig_loc_id").is_in(lower_mapping))

    upwards_edge_attrs = upwards_edges.select("travel_time_min").to_torch()
    upwards_edge_indices = upwards_edges.select(
        pl.col("orig_loc_id").replace_strict(lower_mapping), pl.col("dest_loc_id").replace_strict(upper_mapping)
    ).to_torch()

    if lower_type == upper_type:
        assert downwards_edges.is_empty()
        return {(lower_type, "links", upper_type): (upwards_edge_indices.T, upwards_edge_attrs)}

    downwards_edge_attrs = downwards_edges.select("travel_time_min").to_torch()
    downwards_edge_indices = downwards_edges.select(
        pl.col("orig_loc_id").replace_strict(upper_mapping), pl.col("dest_loc_id").replace_strict(lower_mapping)
    ).to_torch()

    rel_up, rel_down = ("contains", "is_contained_by") if lower_type == LayerType.PLANAR else ("links_to", "links_from")

    return {
        (lower_type, rel_up, upper_type): (upwards_edge_indices.T, upwards_edge_attrs),
        (upper_type, rel_down, lower_type): (downwards_edge_indices.T, downwards_edge_attrs),
    }


def _node_mapping_to_dict(node_mapping: NodeMapping, parent_key: str = PARENT_STOP_ROUTE_ID) -> dict[str, int]:
    if all(isinstance(key, str) for key in node_mapping.keys()):
        return dict(node_mapping.items())

    return {key[0]: item for key, item in node_mapping.items() if key[1] == parent_key}
