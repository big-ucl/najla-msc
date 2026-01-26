from collections.abc import Hashable, Sequence
from enum import Enum
from typing import Callable, Mapping, Type

import geopandas as gpd
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from geopandas import GeoDataFrame
from torch_geometric.data import HeteroData
from torch_geometric.io.planetoid import edge_index_from_dict

from activitygraphs.base import Mode
from activitygraphs.network import LayerType, Network

type Encoder = Callable[[pd.Series | pl.Series], torch.Tensor]
type NodeProcessor = Callable[
    [Network, Sequence[str], Mapping[str, Encoder] | None], tuple[dict[str, int], torch.Tensor | None]
]
type EdgeProcessor = Callable[
    [Network, Sequence[str], Mapping[str, int], Mapping[str, Encoder] | None],
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
    pass


class GeometryEncoder:
    pass


def network_to_pyg(network: Network, embedding_dim=EMBEDDING_DIM) -> HeteroData:
    data = HeteroData()

    layer_name_encoder = EnumEncoder(network.layers.keys(), embedding_dim)
    loc_type_encoder = EnumEncoder(network.location_types, embedding_dim)
    node_encoders = {"type": loc_type_encoder, LAYER_NAME_COL: layer_name_encoder}

    route_mode_encoder = EnumEncoder(Mode, embedding_dim)
    edge_encoders = {"route_mode": route_mode_encoder}

    for layer_type in set(network.layers.values()):
        _process_layers_by_type(data, network, layer_type, node_encoders, edge_encoders)

    return data


def _process_layers_by_type(
    data: HeteroData,
    network: Network,
    layer_type: LayerType,
    node_encoders: Mapping[str, Encoder] | None = None,
    edge_encoders: Mapping[str, Encoder] | None = None,
):
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
    # By construction, every loc_id has a (loc_id, `parent`) node, plus perhaps additional (loc_id, route_id) nodes
    # TODO: add assert check for presence of (loc_id, `parent`) at every stop
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
    pt_locations_gdf = pt_locations.to_pandas().merge(layer_locations_gdf, on="loc_id")

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

    layer_locations_gdf = pd.concat(layer_locations.values(), ignore_index=True)
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
    node_mapping: Mapping[str, int],
    edge_encoders: Mapping[str, Encoder] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
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
