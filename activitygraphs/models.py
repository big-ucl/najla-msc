import polars as pl
import synthetic
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from synthetic import SyntheticGenerator
from torch_geometric.utils import to_dense_batch


class FiftyFifty(nn.Module):
    def forward(self, x, data):
        return self.decode(data), None, None

    def decode(self, data):
        y, _ = to_dense_batch(data.y, data.batch)
        return torch.zeros_like(y)


class NodeLabelVGAE(gnn.VGAE):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__(encoder, decoder)

    def forward(self, data):
        return super().forward(data.x, data)

    def infer(self, data):
        mu, log_std = super().forward(data.x_infer, data)
        z = super().reparametrize(mu, log_std)
        return super().decode(z)


class VAE(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(*args, **kwargs)
        z = self.reparameterize(mu, log_var)
        y = self.decoder(z)
        return y, mu, log_var

    def encode(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(*args, **kwargs)

    @staticmethod
    def reparameterize(mu, log_var) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(log_var)

        return mu + epsilon * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        pass


class GCNEncoder(nn.Module):
    def __init__(self, in_node_channels, in_graph_channels, hidden_channels, num_layers, latent_channels, dropout=0.2):
        super().__init__()

        if num_layers <= 1:
            raise ValueError(f"GCNEncoder needs at least 2 layers, got {num_layers=}")

        in_channels = in_node_channels + in_graph_channels

        self.gcn_shared = gnn.GCN(in_channels, hidden_channels, num_layers - 1, hidden_channels, dropout=dropout)
        self.mu = nn.Linear(hidden_channels, latent_channels)
        self.log_var = nn.Linear(hidden_channels, latent_channels)

    def forward(self, x, data):
        # Inject graph features into node features
        graph_x = data.graph_x[data.batch].unsqueeze(1)
        node_x = torch.cat([x, graph_x], dim=1)

        edge_index = data.edge_index
        edge_weight = data.edge_weight

        x = self.gcn_shared(node_x, edge_index, edge_weight)
        x = F.relu(x)
        x = gnn.global_max_pool(x, data.batch)

        mu = self.mu(x)
        log_var = self.log_var(x)

        return mu, log_var


class MLPEncoder(nn.Module):
    def __init__(
        self,
        in_num_nodes,
        in_num_node_features,
        in_num_graph_features,
        hidden_channels,
        num_layers,
        latent_channels,
        dropout=0.2,
    ):
        super().__init__()

        if num_layers <= 1:
            raise ValueError(f"MLP encoder needs at least 2 layers, got {num_layers=}")

        in_channels = in_num_nodes * in_num_node_features + in_num_graph_features

        self.mlp_shared = MLP(in_channels, hidden_channels, num_layers - 1, hidden_channels, dropout=dropout)
        self.mu = nn.Linear(hidden_channels, latent_channels)
        self.log_var = nn.Linear(hidden_channels, latent_channels)

    def forward(self, x, data):
        batched_x, _ = to_dense_batch(x, data.batch)
        node_x = batched_x.flatten(start_dim=1)
        graph_x = data.graph_x.unsqueeze(1)
        combined_x = torch.cat([node_x, graph_x], dim=1)

        x = self.mlp_shared(combined_x)
        x = F.relu(x)

        mu = self.mu(x)
        log_var = self.log_var(x)

        return mu, log_var


class MLPDecoder(nn.Module):
    def __init__(self, latent_channels, hidden_channels, num_layers, out_nodes, out_classes, dropout=0.2):
        super().__init__()

        self.out_nodes = out_nodes
        self.out_classes = out_classes

        out_channels = out_nodes * out_classes

        self.mlp = MLP(latent_channels, hidden_channels, num_layers, out_channels, dropout=dropout)

    def forward(self, z):
        y = self.mlp(z)
        y = y.reshape(y.shape[0], -1, self.out_classes)

        return y


class MLP(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_layers, out_channels, dropout=0.2):
        super().__init__()

        layer_channels = (
            [(input_channels, hidden_channels)]
            + [(hidden_channels, hidden_channels)] * (num_layers - 1)
            + [(hidden_channels, out_channels)]
        )

        modules = nn.Sequential()
        for i, (in_channels, out_channels) in enumerate(layer_channels):
            modules.add_module(f"linear_{i}", nn.Linear(in_channels, out_channels))

            if i < len(layer_channels) - 1:
                modules.add_module(f"relu_{i}", nn.ReLU())
                modules.add_module(f"dropout_{i}", nn.Dropout(dropout))

        self.mlp = modules

    def forward(self, x):
        return self.mlp(x)


class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_loss = float("inf")
        self.no_improvement_count = 0

    def check(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                if self.verbose:
                    print(f"Early stopping: no improvement. Validation loss={val_loss:.4f}")
                return True
        return False


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
