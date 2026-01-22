from collections.abc import Hashable, Sequence
from enum import Enum
from typing import Callable, Mapping, Type

import geopandas as gpd
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from activitygraphs.network import Layer, LayerType, Network

type Encoder = Callable[[pd.Series | pl.Series], torch.Tensor]
type NodeProcessor = Callable[
    [Mapping[str, gpd.GeoDataFrame], Mapping[str, Encoder] | None], tuple[dict[str, int], torch.Tensor | None]
]
type EdgeProcessor = Callable[[Sequence[Layer], Mapping[str, int]], tuple[torch.Tensor, torch.Tensor]]

EMBEDDING_DIM = 3
LAYER_NAME_COL = "layer_name"


class EnumEncoder:
    def __init__(self, enum_cls: Type[Enum] | list[Hashable], embedding_dim: int):
        self._enum_cls = enum_cls
        self.mapping = {enum: index for index, enum in enumerate(enum_cls)}

        self.num_embeddings = len(self.mapping)
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)

    def __call__(self, enum_series: pd.Series):
        indices = enum_series.map(self.mapping).to_numpy()
        if indices.dtype != "int":
            raise ValueError("Conversion to indices failed, some values not in mapping ")

        device = self.embedding.weight.device
        indices = torch.tensor(indices, device=device, dtype=torch.long)
        return self.embedding(indices)


class GeometryEncoder:
    pass


def network_to_pyg(network: Network, embedding_dim=EMBEDDING_DIM) -> HeteroData:
    data = HeteroData()

    layer_name_encoder = EnumEncoder(network.layers.keys(), embedding_dim)
    loc_type_encoder = EnumEncoder(network.location_types, embedding_dim)
    encoders = {"type": loc_type_encoder, LAYER_NAME_COL: layer_name_encoder}

    for layer_type in set(network.layers.values()):
        if layer_type != LayerType.PUBLIC_TRANSPORT:
            _process_layers_by_type(data, network, layer_type, encoders)

    return data


def _process_layers_by_type(
    data: HeteroData, network: Network, layer_type: LayerType, encoders: Mapping[str, Encoder] | None = None
):
    assert not data[layer_type]

    node_processor = _select_node_processor(layer_type)
    edge_processor = _select_edge_processor(layer_type)

    layer_names = [name for name, l_type in network.layers.items() if l_type == layer_type]
    layer_locations = {name: network.get_layer_locations(name) for name in layer_names}
    layers = [network[name] for name in layer_names]

    node_mapping, node_x = node_processor(layer_locations, encoders)
    edge_list, edge_attrs = edge_processor(layers, node_mapping)

    data[layer_type].x = node_x
    data[layer_type].edge_list = edge_list
    data[layer_type].edge_attrs = edge_attrs
    data[layer_type].num_nodes = len(node_mapping)


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
    layer_locations: Mapping[str, gpd.GeoDataFrame], encoders: Mapping[str, Encoder] | None = None
) -> tuple[dict[str, int], torch.Tensor | None]:
    # Combine layer locations together and add a column identifying each layer by name
    layer_names = pd.concat(
        (pd.Series(name).repeat(len(gdf)) for name, gdf in layer_locations.items()), ignore_index=True
    )

    layer_locations_gdf = pd.concat(layer_locations.values(), ignore_index=True)
    layer_locations_gdf[LAYER_NAME_COL] = layer_names

    # Map `loc_id` to indices in PyG graph
    node_mapping = layer_locations_gdf.reset_index(drop=True).reset_index().set_index("loc_id")["index"].to_dict()

    # Encode location features (if they exist)
    x = None
    if encoders is not None:
        xs = [encoder(layer_locations_gdf[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    return node_mapping, x


def _process_pt_layer_nodes(
    layer_locations: Mapping[str, gpd.GeoDataFrame], encoders: Mapping[str, Encoder] | None = None
) -> tuple[dict[str, int], torch.Tensor | None]:
    raise NotImplementedError()


def _process_base_layer_edges(
    layers: Sequence[Layer], node_mapping: Mapping[str, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_index_expr = pl.col("orig_loc_id").replace_strict(node_mapping)
    dest_index_expr = pl.col("dest_loc_id").replace_strict(node_mapping)

    edge_lists = [layer.edge_list.select(orig_index_expr, dest_index_expr).to_torch().T for layer in layers]
    edge_attrs = [layer.edge_list.select("travel_time_min").to_torch() for layer in layers]

    return torch.cat(edge_lists, dim=1), torch.cat(edge_attrs)


def _process_pt_layer_edges(
    layers: Sequence[Layer], node_mapping: Mapping[str, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError()
