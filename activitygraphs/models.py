import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn

import synthetic


class SimpleGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = gnn.GCNConv(in_channels, hidden_channels)
        self.conv2 = gnn.GCNConv(hidden_channels, out_channels)

    def forward(self, batch):
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr.squeeze()

        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)

        return x


class Benchmark(nn.Module):
    pass


class EqualProbablity(Benchmark):
    def __init__(self, normalize=False):
        super().__init__()
        self.normalize = normalize

    def forward(self, batch):
        logits = torch.ones_like(batch.x) - batch.x

        if self.normalize:
            logits = logits.reshape((len(batch), -1))
            logits = logits / logits.sum(dim=1).unsqueeze(1)
            logits = logits.reshape(batch.x.shape)

        return logits


class BestGuess(Benchmark):
    def __init__(self, all_schedule_graphs: pl.DataFrame):
        super().__init__()
        self.all_schedule_graphs = all_schedule_graphs

    def forward(self, batch):
        xs = batch.x.reshape((len(batch), -1))
        graph_xs = batch.graph_x

        ys = torch.cat([self._predict(graph_x, x) for graph_x, x in zip(graph_xs, xs, strict=True)])
        return ys.reshape(batch.x.shape)

    def _predict(self, graph_x, x):
        df = self.all_schedule_graphs

        x_cols = [pl.col(col) for col in df.columns if col.startswith("from_")]
        y_cols = [pl.col(col) for col in df.columns if col.startswith("to_")]
        exprs = [col == x for col, x in zip(x_cols, x)]

        conditioned = df.filter(pl.col("sequence_num") == graph_x, *exprs)
        conditioned = conditioned.select(*[y_col - x_col for x_col, y_col in zip(x_cols, y_cols)])
        probs = conditioned.sum() / len(conditioned)

        return probs.to_torch()

    @classmethod
    def from_graph(cls, graph: synthetic.SyntheticGraph) -> "BestGuess":
        all_schedules = synthetic.compute_all_possible_schedules(graph)
        return cls(all_schedules)
