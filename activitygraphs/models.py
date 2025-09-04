import polars as pl
import synthetic
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from synthetic import SyntheticGenerator


class NodeLabelVGAE(gnn.VGAE):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__(encoder, decoder)

    def forward(self, data):
        return super().forward(data.x, data)

    def infer(self, data):
        mu, log_std = super().forward(data.x_infer, data)
        z = super().reparametrize(mu, log_std)
        return super().decode(z)


class GCNEncoder(nn.Module):
    def __init__(self, in_node_channels, in_graph_channels, hidden_channels, num_layers, latent_channels, dropout=0.2):
        super().__init__()

        if num_layers <= 1:
            raise ValueError(f"GCNEncoder needs at least 2 layers, got {num_layers=}")

        in_channels = in_node_channels + in_graph_channels

        self.gcn_shared = gnn.GCN(in_channels, hidden_channels, num_layers - 1, hidden_channels, dropout=dropout)
        self.gcn_mu = gnn.GCNConv(hidden_channels, latent_channels)
        self.gcn_log_std = gnn.GCNConv(hidden_channels, latent_channels)

    def forward(self, x, data):
        # Inject graph features into node features
        graph_x = data.graph_x[data.batch].unsqueeze(1)
        node_x = torch.cat([x, graph_x], dim=1)

        edge_index = data.edge_index
        edge_weight = data.edge_weight

        x = self.gcn_shared(node_x, edge_index, edge_weight)
        x = F.relu(x)

        mu = self.gcn_mu(x, edge_index, edge_weight)
        log_std = self.gcn_log_std(x, edge_index, edge_weight)

        return mu, log_std


class MLPDecoder(nn.Module):
    def __init__(self, latent_channels, hidden_channels, num_layers, out_channels, dropout=0.2):
        super().__init__()

        layer_channels = (
            [(latent_channels, hidden_channels)]
            + [(hidden_channels, hidden_channels)] * (num_layers - 1)
            + [(hidden_channels, out_channels)]
        )

        modules = nn.Sequential()
        for i, (in_channels, out_channels) in enumerate(layer_channels):
            modules.add_module(f"dense_{i}", nn.Linear(in_channels, out_channels))
            modules.add_module(f"relu_{i}", nn.ReLU())
            modules.add_module(f"dropout_{i}", nn.Dropout(dropout))

        self.mlp = modules

    def forward(self, z):
        return self.mlp(z)


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


class EqualProbability(Benchmark):
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
    """Model that computes conditional probabilities over all valid schedules. Should provide the best possible loss."""

    def __init__(self, all_schedule_graphs: pl.DataFrame, strict: bool = True):
        """_summary_

        Args:
            all_schedule_graphs (pl.DataFrame):
                A dataframe of all possible steps in a schedule, with a `person_id` index column, a `sequence_num`
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

        ys = torch.cat([self._predict(graph_x.item(), x) for graph_x, x in zip(graph_xs, xs, strict=True)])
        return ys.reshape(batch.x.shape)

    def _predict(self, graph_x, x):
        df = self.all_schedule_graphs

        x_cols = [pl.col(col) for col in df.columns if col.startswith("from_")]
        y_cols = [pl.col(col) for col in df.columns if col.startswith("to_")]
        exprs = [col == x for col, x in zip(x_cols, x)]

        conditioned = df.filter(pl.col("sequence_num") == graph_x, *exprs)
        conditioned = conditioned.select(*[y_col - x_col for x_col, y_col in zip(x_cols, y_cols)])
        probs = conditioned.sum() / len(conditioned)

        if self.strict and len(conditioned) == 0:
            return (torch.ones_like(x) / len(x)).unsqueeze(0)

        return probs.to_torch().float()

    @classmethod
    def from_graph(cls, graph: synthetic.SyntheticGraph) -> "BestGuess":
        """Creates a BestGuess model from a SyntheticGraph by computing all possible schedules

        Args:
            graph (synthetic.SyntheticGraph): the synthetic graph over which to compute schedules

        Returns:
            BestGuess: a new BestGuess instance over the graph
        """
        all_schedules = synthetic.compute_all_possible_schedules(graph, SyntheticGenerator.AVAILABLE_SCHEDULES)
        return cls(all_schedules)
