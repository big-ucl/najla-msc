import polars as pl
import synthetic
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn


class SimpleGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = gnn.GCNConv(in_channels, hidden_channels)
        self.conv2 = gnn.GCNConv(hidden_channels, out_channels)
        self.lin = gnn.Linear(hidden_channels, out_channels)

    def forward(self, batch):
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr.squeeze()

        x = self.conv1(x, edge_index, 1 / edge_attr)
        x = F.relu(x)
        # TODO x = self.lin(x, edge_index, edge_attr)
        x = self.conv2(x, edge_index, 1 / edge_attr)

        return x


class Benchmark(nn.Module):
    pass


class EqualProbablity(Benchmark):
    def __init__(self, normalize=True):
        super().__init__()
        self.normalize = normalize

    def forward(self, batch):
        logits = torch.ones_like(batch.x) - batch.x

        if self.normalize:
            logits = logits.reshape((batch.graph_x.shape[0], -1))
            logits = logits / logits.sum(dim=1).unsqueeze(1)
            logits = logits.reshape(batch.x.shape)

        return logits


class BestGuess(Benchmark):
    def __init__(self, all_schedule_graphs: pl.DataFrame):
        super().__init__()
        self.all_schedule_graphs = all_schedule_graphs

    def forward(self, batch):
        graph_xs = batch.graph_x
        xs = batch.x.reshape((graph_xs.shape[0], -1))
        ytest = batch.y.reshape((graph_xs.shape[0], -1))

        ys = torch.cat([self._predict(graph_x, x, y) for graph_x, x, y in zip(graph_xs, xs, ytest, strict=True)])
        return ys.reshape(batch.x.shape)

    def _predict(self, graph_x, x, y):
        df = self.all_schedule_graphs

        x_cols = [pl.col(col) for col in df.columns if col.startswith("from_")]
        y_cols = [pl.col(col) for col in df.columns if col.startswith("to_")]
        exprs = [col == x for col, x in zip(x_cols, x)]

        conditioned = df.filter(pl.col("sequence_num") == graph_x, *exprs)
        conditioned = conditioned.select(*[y_col - x_col for x_col, y_col in zip(x_cols, y_cols)])
        probs = conditioned.sum() / len(conditioned)

        if len(conditioned) == 0:
            raise ValueError(f"Schedule does not exist for seq_num={graph_x}, {x=}")

        return probs.to_torch().float()

    @classmethod
    def from_graph(cls, graph: synthetic.SyntheticGraph) -> "BestGuess":
        all_schedules = synthetic.compute_all_possible_schedules(graph)
        return cls(all_schedules)
