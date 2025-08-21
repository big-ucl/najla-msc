import polars as pl
import synthetic
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn


class SimpleGCN(nn.Module):
    """A simple two-layer GCN model with edge weights from the PyG tutorial"""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        """
        Args:
            in_channels (int): number of node features
            hidden_channels (int): number of hidden channels
            out_channels (int): number of output per node
        """
        super().__init__()
        self.conv1 = gnn.GCNConv(in_channels, hidden_channels)
        self.conv2 = gnn.GCNConv(hidden_channels, out_channels)

    def forward(self, batch):
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr.squeeze()

        x = self.conv1(x, edge_index, 1 / edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, 1 / edge_attr)

        return x


class MLP(nn.Module):
    def __init__(self, n_nodes: int, n_graph_x: int, hidden_channels: int, num_layers: int):
        super().__init__()
        self.n_nodes = n_nodes
        self.mlp = gnn.MLP(
            in_channels=n_nodes + n_graph_x,
            hidden_channels=hidden_channels,
            out_channels=n_nodes,
            num_layers=num_layers,
        )

    def forward(self, batch):
        X = batch.x.reshape((-1, self.n_nodes))
        X = torch.cat([batch.graph_x.unsqueeze(1), X], dim=1)

        return self.mlp(X).reshape((-1, 1))


class Benchmark(nn.Module):
    """Baseline for models that require no training and are used for evaluation"""

    pass


class EqualProbablity(Benchmark):
    """Model that outputs equal probability for all nodes not already selected"""

    def __init__(self, normalize: bool = True):
        """
        Args:
            normalize (bool, optional): Output sum to 1 if True, otherwise binary indicators. Defaults to True.
        """
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
    """Model that computes conditional probablities over all valid schedules. Should provide the best possible loss."""

    def __init__(self, all_schedule_graphs: pl.DataFrame, strict: bool = True):
        """_summary_

        Args:
            all_schedule_graphs (pl.DataFrame):
                A dataframe of all possible steps in a schedule, with a `person_id` index column, a `sequenece_num`
                column, a `from_loc_id_N` indicator for each node `N`, and a `to_loc_id_N` indicator column for each
                node `N`.
            strict (bool, optional): Raises an error if schedule does not exist. Defaults to True.
        """
        super().__init__()
        self.all_schedule_graphs = all_schedule_graphs
        self.strict = strict

    def forward(self, batch):
        graph_xs = batch.graph_x
        xs = batch.x.reshape((graph_xs.shape[0], -1))
        y_targets = batch.y.reshape((graph_xs.shape[0], -1))
        ps = batch.person_id.reshape((graph_xs.shape[0], -1))

        ys = torch.cat([
            self._predict(graph_x.item(), x, y.item(), p.item())
            for graph_x, x, y, p in zip(graph_xs, xs, y_targets, ps, strict=True)
        ])
        return ys.reshape(batch.x.shape)

    def _predict(self, graph_x, x, y, p):
        df = self.all_schedule_graphs

        x_cols = [pl.col(col) for col in df.columns if col.startswith("from_")]
        y_cols = [pl.col(col) for col in df.columns if col.startswith("to_")]
        exprs = [col == x for col, x in zip(x_cols, x)]

        conditioned = df.filter(pl.col("sequence_num") == graph_x, *exprs)
        conditioned = conditioned.select(*[y_col - x_col for x_col, y_col in zip(x_cols, y_cols)])
        probs = conditioned.sum() / len(conditioned)

        if self.strict and len(conditioned) == 0:
            raise ValueError(
                f"Schedule does not exist for {p=} seq_num={graph_x}, {x=}, {y=}. Check equal path lengths."
            )

        return probs.to_torch().float()

    @classmethod
    def from_graph(cls, graph: synthetic.SyntheticGraph) -> "BestGuess":
        """Creates a BestGuess model from a SyntheticGraph by computing all possible schedules

        Args:
            graph (synthetic.SyntheticGraph): the synthetic graph over which to compute schedules

        Returns:
            BestGuess: a new BestGuess instance over the graph
        """
        all_schedules = synthetic.compute_all_possible_schedules(graph)
        return cls(all_schedules)
