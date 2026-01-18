from collections.abc import Hashable
from enum import Enum
from typing import Callable, Mapping, Type

import geopandas as gpd
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from activitygraphs.network import Layer, Network

type Encoder = Callable[[pd.Series | pl.Series], torch.Tensor]


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


def network_to_pyg(network: Network) -> HeteroData:
    pass


def _process_layer_nodes(
    layer_locations_gdf: gpd.GeoDataFrame, encoders: Mapping[str, Encoder] | None = None
) -> tuple[dict[str, int], torch.Tensor | None]:
    node_mapping = layer_locations_gdf.reset_index(drop=True).reset_index().set_index("loc_id")["index"].to_dict()

    x = None
    if encoders is not None:
        xs = [encoder(layer_locations_gdf[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    return node_mapping, x


def _process_layer_edges(layer: Layer, node_mapping: Mapping[str, int]):
    edge_list = (
        layer.edge_list
        .select(pl.col("orig_loc_id").replace_strict(node_mapping), pl.col("dest_loc_id").replace_strict(node_mapping))
        .to_torch()
        .T
    )

    edge_attrs = layer.edge_list.select("travel_time_min").to_torch()

    return edge_list, edge_attrs
