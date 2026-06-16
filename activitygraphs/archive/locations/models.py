import polars as pl
import synthetic
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from synthetic import SyntheticGenerator
from torch_geometric.utils import to_dense_batch


class FiftyFifty(nn.Module):
    """
    Description: A trivial baseline model that always predicts zero probability for every node
    (i.e. predicts no location is visited). This serves as a sanity-check lower-bound baseline.
    Any real model should outperform this. The output format matches the VAE API (returns a
    3-tuple) so it can be used with the same evaluation utilities.
    """
    def forward(self, x, data):
        """
        Description: Runs the trivial forward pass, returning all-zero predictions.

        Input:
          - x: Ignored. Accepts any input for API compatibility.
          - data (torch_geometric.data.Data): The batch of graphs. Used only to get the output shape.

        Output:
          - (tuple[torch.Tensor, None, None]): A 3-tuple where the first element is a tensor of
            zeros (same shape as data.y in dense form), and the remaining two are None (for API
            compatibility with VAE models that return mu and log_std).
        """
        return self.decode(data), None, None

    def decode(self, data):
        """
        Description: Produces the all-zero output tensor. Converts ground-truth labels to a dense
        batch format and creates a zero tensor of the same shape.

        Input:
          - data (torch_geometric.data.Data): The batch of graphs, used to access ground-truth
            labels and batch assignment.

        Output:
          - (torch.Tensor): Zero tensor of shape (batch_size, n_nodes), representing zero
            predicted probability for every node for every person.
        """
        y, _ = to_dense_batch(data.y, data.batch)  # Convert to dense batch shape (B, V)
        return torch.zeros_like(y)  # Return zeros — predicting no locations visited


class NodeLabelVGAE(gnn.VGAE):
    """
    Description: A Variational Graph Autoencoder (VGAE) adapted for node label prediction.
    Extends PyG's built-in VGAE class to accept PyG Data batch objects directly. The model
    encodes node features into a latent distribution, samples a latent vector, and decodes
    it back to node label predictions. Provides a separate `infer` method that uses only
    home-location inputs (simulating prediction without knowing the full schedule).
    """
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        """
        Description: Initialises the NodeLabelVGAE with an encoder and a decoder module.

        Input:
          - encoder (nn.Module): The GNN encoder that maps node features to (mu, log_std).
          - decoder (nn.Module): The decoder that maps latent vectors to node label predictions.
        """
        super().__init__(encoder, decoder)

    def forward(self, data):
        """
        Description: Runs the full VGAE forward pass during training: encodes the node features
        (with full label information visible) to get the latent distribution, then decodes.

        Input:
          - data (torch_geometric.data.Data): A batch of graph data objects, using `data.x`
            which contains full node features + all ground-truth labels.

        Output:
          - (tuple): Returns the output of the parent VGAE forward, typically
            (reconstruction, mu, log_std).
        """
        return super().forward(data.x, data)

    def infer(self, data):
        """
        Description: Runs inference (prediction) using only the home-location inputs, simulating
        the scenario where we only know where a person lives and want to predict the rest of their
        activity schedule. Uses `data.x_infer` which contains node features + home-only labels.

        Input:
          - data (torch_geometric.data.Data): A batch of graph data objects, using `data.x_infer`
            which contains node features + home-only node labels (partial observation).

        Output:
          - (torch.Tensor): Decoded node label predictions, shape=(batch_size, n_nodes, n_classes).
        """
        mu, log_std = super().forward(data.x_infer, data)  # Encode with home-only features
        z = super().reparametrize(mu, log_std)  # Sample latent vector using reparameterisation trick
        return super().decode(z)  # Decode to predicted node labels


class VAE(nn.Module):
    """
    Description: A general-purpose Variational Autoencoder (VAE) base class for node label
    prediction. Wraps an encoder module and a decoder module. The encoder maps input features
    to a latent distribution (mu, log_var), the reparameterisation trick samples a latent vector
    z, and the decoder maps z to output predictions. This class can be subclassed to use different
    encoder/decoder architectures (e.g. GCN, MLP).
    """
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        """
        Description: Initialises the VAE with the given encoder and decoder modules.

        Input:
          - encoder (nn.Module): Any module that accepts input features and returns (mu, log_var).
          - decoder (nn.Module): Any module that accepts a latent vector z and returns predictions.
        """
        super().__init__()
        self.encoder = encoder  # Maps input features to latent mean and log-variance
        self.decoder = decoder  # Maps sampled latent vector to output predictions

    def forward(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Description: Runs the full VAE forward pass: encode → reparameterise → decode.

        Input:
          - *args, **kwargs: Forwarded to self.encode (and thus self.encoder). Typically includes
            node feature tensor and a Data batch object.

        Output:
          - (tuple[torch.Tensor, torch.Tensor, torch.Tensor]): A 3-tuple of:
              y       : model output / reconstruction (e.g. node label predictions)
              mu      : latent mean, shape=(batch, latent_dim)
              log_var : latent log-variance, shape=(batch, latent_dim)
        """
        mu, log_var = self.encode(*args, **kwargs)  # Encode inputs to latent distribution parameters
        z = self.reparameterize(mu, log_var)  # Sample latent vector (differentiable via reparameterisation)
        y = self.decoder(z)  # Decode latent vector to output predictions
        return y, mu, log_var

    def encode(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Description: Runs the encoder module to produce the latent distribution parameters.

        Input:
          - *args, **kwargs: Forwarded directly to self.encoder.

        Output:
          - (tuple[torch.Tensor, torch.Tensor]): A tuple of (mu, log_var), both of shape
            (batch_size, latent_dim).
        """
        return self.encoder(*args, **kwargs)

    @staticmethod
    def reparameterize(mu, log_var) -> torch.Tensor:
        """
        Description: Applies the reparameterisation trick: z = mu + epsilon * std, where
        epsilon ~ N(0, 1) and std = exp(0.5 * log_var). This allows gradients to flow through
        the sampling operation during backpropagation.

        Input:
          - mu (torch.Tensor): Mean of the latent distribution, shape=(batch, latent_dim).
          - log_var (torch.Tensor): Log-variance of the latent distribution, shape=(batch, latent_dim).

        Output:
          - (torch.Tensor): Sampled latent vector z, same shape as mu, drawn from N(mu, std^2).
        """
        std = torch.exp(0.5 * log_var)  # Convert log-variance to standard deviation
        epsilon = torch.randn_like(log_var)  # Sample noise from standard normal, same shape as log_var

        return mu + epsilon * std  # Reparameterised sample: differentiable w.r.t. mu and log_var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Description: Placeholder decode method. Subclasses should override this if they need
        a custom decode step beyond calling self.decoder(z).

        Input:
          - z (torch.Tensor): Latent vector, shape=(batch, latent_dim).

        Output:
          - None (this base implementation does not return anything — override in subclasses).
        """
        pass


class GCNEncoder(nn.Module):
    """
    Description: A Graph Convolutional Network (GCN) encoder for the VAE. Processes node features
    using multiple GCN layers, injects person-level (graph-level) features by concatenating them
    onto each node, applies global max pooling to produce a single graph-level representation,
    then projects to latent mean (mu) and log-variance (log_var) vectors. This gives a
    graph-level latent representation that captures the entire activity graph structure.
    """
    def __init__(self, in_node_channels, in_graph_channels, hidden_channels, num_layers, latent_channels, dropout=0.2):
        """
        Description: Initialises the GCN encoder architecture.

        Input:
          - in_node_channels (int): Number of input features per node (e.g. node type indicators + label).
          - in_graph_channels (int): Number of person/graph-level features (e.g. income).
          - hidden_channels (int): Number of hidden units in each GCN layer.
          - num_layers (int): Total number of layers including the two output projection layers.
            Must be >= 2.
          - latent_channels (int): Dimensionality of the latent space (size of mu and log_var vectors).
          - dropout (float): Dropout rate applied between GCN layers for regularisation. Defaults to 0.2.
        """
        super().__init__()

        if num_layers <= 1:
            raise ValueError(f"GCNEncoder needs at least 2 layers, got {num_layers=}")

        # Combined input size: node features + graph features injected per node
        in_channels = in_node_channels + in_graph_channels

        # Shared GCN backbone processes all nodes, producing hidden representations
        self.gcn_shared = gnn.GCN(in_channels, hidden_channels, num_layers - 1, hidden_channels, dropout=dropout)
        # Linear layers to project pooled graph representation to latent distribution parameters
        self.mu = nn.Linear(hidden_channels, latent_channels)  # Projects to latent mean
        self.log_var = nn.Linear(hidden_channels, latent_channels)  # Projects to latent log-variance

    def forward(self, x, data):
        """
        Description: Runs the GCN encoder forward pass. Injects person-level features into each
        node's feature vector, applies GCN message passing, pools all nodes into one vector via
        global max pooling, then computes mu and log_var.

        Input:
          - x (torch.Tensor): Node feature matrix, shape=(total_nodes_in_batch, in_node_channels).
          - data (torch_geometric.data.Data): The batch of graph data, used to access:
              data.graph_x   : person-level features, shape=(batch_size, in_graph_channels)
              data.batch     : node-to-graph assignment vector, shape=(total_nodes,)
              data.edge_index: graph connectivity, shape=(2, E)
              data.edge_weight: edge weights, shape=(E,) or None

        Output:
          - (tuple[torch.Tensor, torch.Tensor]): (mu, log_var), both of shape=(batch_size, latent_channels).
        """
        # Inject graph features into node features — broadcast person-level features to all their nodes
        graph_x = data.graph_x[data.batch].unsqueeze(1)  # Shape: (total_nodes, 1, in_graph_channels)
        node_x = torch.cat([x, graph_x], dim=1)  # Concatenate node and person features per node

        edge_index = data.edge_index  # Sparse edge connectivity for GCN message passing
        edge_weight = data.edge_weight  # Optional edge weights (e.g. travel distances)

        x = self.gcn_shared(node_x, edge_index, edge_weight)  # Apply GCN layers
        x = F.relu(x)  # Non-linear activation
        x = gnn.global_max_pool(x, data.batch)  # Aggregate node representations to graph level

        mu = self.mu(x)  # Project to latent mean
        log_var = self.log_var(x)  # Project to latent log-variance

        return mu, log_var


class MLPEncoder(nn.Module):
    """
    Description: A Multi-Layer Perceptron (MLP) encoder for the VAE that ignores graph structure.
    Instead of using GCN message passing, it flattens all node features into a single vector per
    person and concatenates person-level features, then processes everything through a standard
    feedforward network. This serves as a graph-unaware baseline to compare against the GCN encoder.
    """
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
        """
        Description: Initialises the MLP encoder architecture.

        Input:
          - in_num_nodes (int): Number of nodes in the graph (V). Used to compute flattened input size.
          - in_num_node_features (int): Number of features per node (F). The full node feature
            vector is V * F after flattening.
          - in_num_graph_features (int): Number of person/graph-level features (P) (e.g. income).
          - hidden_channels (int): Number of hidden units in each MLP layer.
          - num_layers (int): Total number of layers including the two output projection layers.
            Must be >= 2.
          - latent_channels (int): Dimensionality of the latent space.
          - dropout (float): Dropout rate for regularisation. Defaults to 0.2.
        """
        super().__init__()

        if num_layers <= 1:
            raise ValueError(f"MLP encoder needs at least 2 layers, got {num_layers=}")

        # Flattened input: all node features concatenated + person-level features
        in_channels = in_num_nodes * in_num_node_features + in_num_graph_features

        # Shared MLP backbone processes the full flattened graph representation
        self.mlp_shared = MLP(in_channels, hidden_channels, num_layers - 1, hidden_channels, dropout=dropout)
        # Linear projection layers for the latent distribution parameters
        self.mu = nn.Linear(hidden_channels, latent_channels)  # Projects to latent mean
        self.log_var = nn.Linear(hidden_channels, latent_channels)  # Projects to latent log-variance

    def forward(self, x, data):
        """
        Description: Runs the MLP encoder forward pass. Converts sparse node features to a dense
        batch, flattens all nodes into one vector per person, concatenates person-level features,
        then computes mu and log_var.

        Input:
          - x (torch.Tensor): Node feature matrix, shape=(total_nodes_in_batch, in_node_features).
          - data (torch_geometric.data.Data): The batch of graph data, used to access:
              data.batch   : node-to-graph assignment vector
              data.graph_x : person-level features, shape=(batch_size, in_graph_features)

        Output:
          - (tuple[torch.Tensor, torch.Tensor]): (mu, log_var), both of shape=(batch_size, latent_channels).
        """
        batched_x, _ = to_dense_batch(x, data.batch)  # Convert to dense: shape=(batch, n_nodes, features)
        node_x = batched_x.flatten(start_dim=1)  # Flatten to (batch, n_nodes * features)
        combined_x = torch.cat([node_x, data.graph_x], dim=1)  # Append person-level features

        x = self.mlp_shared(combined_x)  # Apply MLP layers
        x = F.relu(x)  # Non-linear activation before final projection

        mu = self.mu(x)  # Project to latent mean
        log_var = self.log_var(x)  # Project to latent log-variance

        return mu, log_var


class MLPDecoder(nn.Module):
    """
    Description: A Multi-Layer Perceptron (MLP) decoder for the VAE that maps a latent vector z
    back to node-level output predictions. Produces raw logits (unnormalised scores) for each
    (node, class) pair, which can then be passed through sigmoid or softmax to get probabilities.
    This decoder ignores graph structure entirely — it predicts all nodes simultaneously from z.
    """
    def __init__(self, latent_channels, hidden_channels, num_layers, out_nodes, out_classes, dropout=0.2):
        """
        Description: Initialises the MLP decoder architecture.

        Input:
          - latent_channels (int): Dimensionality of the latent space (input to the decoder).
          - hidden_channels (int): Number of hidden units in each MLP layer.
          - num_layers (int): Number of layers in the MLP.
          - out_nodes (int): Number of output nodes (locations / graph nodes) V.
          - out_classes (int): Number of output classes per node (label types) L.
            Total output size = out_nodes * out_classes.
          - dropout (float): Dropout rate for regularisation. Defaults to 0.2.
        """
        super().__init__()

        self.out_nodes = out_nodes  # Number of nodes in the graph (stored for reshape)
        self.out_classes = out_classes  # Number of label classes per node (stored for reshape)

        out_channels = out_nodes * out_classes  # Total flattened output size: one value per (node, class)

        # MLP that takes the latent vector and outputs a flat prediction vector
        self.mlp = MLP(latent_channels, hidden_channels, num_layers, out_channels, dropout=dropout)

    def forward(self, z):
        """
        Description: Decodes a latent vector z to node label predictions. Passes z through the MLP
        and reshapes the flat output to a (batch, n_nodes, n_classes) tensor.

        Input:
          - z (torch.Tensor): Latent vector sampled from the posterior, shape=(batch, latent_channels).

        Output:
          - (torch.Tensor): Raw logit predictions, shape=(batch, out_nodes, out_classes).
            These are unnormalised — apply sigmoid for binary classification.
        """
        y = self.mlp(z)  # Apply MLP: (batch, latent_channels) -> (batch, out_nodes * out_classes)
        y = y.reshape(y.shape[0], -1, self.out_classes)  # Reshape to (batch, out_nodes, out_classes)

        return y


class MLP(nn.Module):
    """
    Description: A general-purpose fully-connected Multi-Layer Perceptron (MLP). Builds a stack of
    linear layers with ReLU activations and dropout for regularisation. The final output layer has
    no activation or dropout, so it produces raw outputs (logits) suitable for further processing
    (e.g. softmax or sigmoid in a loss function).

    Architecture:
        input_channels -> hidden_channels -> ... (num_layers - 1 hidden layers) ... -> out_channels
    """
    def __init__(self, input_channels, hidden_channels, num_layers, out_channels, dropout=0.2):
        """
        Description: Builds the MLP architecture dynamically as a nn.Sequential module.

        Input:
          - input_channels (int): Number of input features.
          - hidden_channels (int): Number of units in each hidden layer.
          - num_layers (int): Number of hidden layers between input and output.
          - out_channels (int): Number of output units.
          - dropout (float): Dropout probability applied after each hidden activation. Defaults to 0.2.
        """
        super().__init__()

        # Construct the list of (in, out) channel pairs for each linear layer
        # First layer: input_channels -> hidden_channels
        # Middle layers: hidden_channels -> hidden_channels (repeated num_layers - 1 times)
        # Final layer: hidden_channels -> out_channels
        layer_channels = (
            [(input_channels, hidden_channels)]
            + [(hidden_channels, hidden_channels)] * (num_layers - 1)
            + [(hidden_channels, out_channels)]
        )

        modules = nn.Sequential()  # Container that will hold all layer modules in order
        for i, (in_channels, out_channels) in enumerate(layer_channels):
            modules.add_module(f"linear_{i}", nn.Linear(in_channels, out_channels))

            # Add ReLU and Dropout after every layer except the final output layer
            if i < len(layer_channels) - 1:
                modules.add_module(f"relu_{i}", nn.ReLU())
                modules.add_module(f"dropout_{i}", nn.Dropout(dropout))

        self.mlp = modules  # The complete sequential MLP stack

    def forward(self, x):
        """
        Description: Runs the input tensor through all MLP layers.

        Input:
          - x (torch.Tensor): Input tensor, shape=(batch, input_channels).

        Output:
          - (torch.Tensor): Output tensor, shape=(batch, out_channels). Raw logits (no final activation).
        """
        return self.mlp(x)


class EarlyStopping:
    """
    Description: A utility class that monitors validation loss and signals when training should
    stop early to prevent overfitting. Training is stopped if the validation loss does not
    improve by at least `delta` over `patience` consecutive epochs.
    """
    def __init__(self, patience=5, delta=0, verbose=False):
        """
        Description: Initialises the early stopping tracker.

        Input:
          - patience (int): Number of epochs to wait without improvement before stopping.
            Defaults to 5.
          - delta (float): Minimum improvement in validation loss to be considered an improvement.
            Defaults to 0 (any improvement counts).
          - verbose (bool): If True, prints a message when early stopping is triggered.
            Defaults to False.
        """
        self.patience = patience  # How many bad epochs to tolerate before stopping
        self.delta = delta  # Minimum required improvement magnitude
        self.verbose = verbose  # Whether to print a message on early stopping
        self.best_loss = float("inf")  # Best validation loss seen so far (initialised to infinity)
        self.no_improvement_count = 0  # Counter of consecutive epochs without improvement

    def check(self, val_loss: float) -> bool:
        """
        Description: Checks whether training should stop based on the current validation loss.
        Updates the best loss and counter. Returns True (stop training) if there has been no
        improvement for `patience` consecutive epochs.

        Input:
          - val_loss (float): The validation loss at the current epoch.

        Output:
          - (bool): True if training should stop (patience exceeded), False otherwise.
        """
        if val_loss < self.best_loss - self.delta:
            # Improvement found: update the best loss and reset the counter
            self.best_loss = val_loss
            self.no_improvement_count = 0
        else:
            # No meaningful improvement: increment the patience counter
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                if self.verbose:
                    print(f"Early stopping: no improvement. Validation loss={val_loss:.4f}")
                return True  # Signal that training should stop
        return False  # Training should continue


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
        """
        Description: Runs the two-layer GCN forward pass. Edge weights are inverted (1/distance)
        so that closer neighbours have stronger influence during message passing. Applies ReLU
        activation between the two convolutional layers.

        Input:
          - batch (torch_geometric.data.Data): A batch of graph data containing:
              batch.x         : node features, shape=(total_nodes, in_channels)
              batch.edge_index: graph connectivity, shape=(2, E)
              batch.edge_attr : edge attributes (e.g. distances), shape=(E, 1) or (E,)

        Output:
          - (torch.Tensor): Node-level output logits, shape=(total_nodes, out_channels).
        """
        x = batch.x  # Node feature matrix
        edge_index = batch.edge_index  # Sparse edge connectivity (COO format)
        edge_attr = batch.edge_attr.squeeze()  # Edge attributes as a 1D vector (e.g. distances)

        x = self.conv1(x, edge_index, 1 / edge_attr)  # First GCN layer: weight by inverse distance
        x = F.relu(x)  # Non-linear activation
        x = self.conv2(x, edge_index, 1 / edge_attr)  # Second GCN layer: weight by inverse distance

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
        """
        Description: Produces equal-probability predictions for all nodes not already selected
        (i.e. nodes with x = 0). Nodes already selected (x = 1) receive a probability of 0.
        If normalize=True, the outputs sum to 1 (proper probability distribution over unselected
        nodes); otherwise they are binary indicators.

        Input:
          - batch (torch_geometric.data.Data): A batch of graph data containing:
              batch.x      : node features (with label indicators at the end), shape=(total_nodes, F)
              batch.graph_x: person-level features, used to infer batch size

        Output:
          - (torch.Tensor): Probability or indicator values per node, shape=(total_nodes, F).
        """
        # Flip the current node labels: unselected nodes (x=0) get 1, selected nodes (x=1) get 0
        logits = torch.ones_like(batch.x) - batch.x

        if self.normalize:
            # Reshape to (batch_size, n_nodes * features) and normalise so outputs sum to 1 per person
            logits = logits.reshape((batch.graph_x.shape[0], -1))
            logits = logits / logits.sum(dim=1).unsqueeze(1)  # Row-wise normalisation
            logits = logits.reshape(batch.x.shape)  # Restore original shape

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
        """
        Description: Predicts the next visited location for each person in the batch by looking
        up the empirical conditional probability from the precomputed all-schedules DataFrame.
        For each person, conditions on their current sequence position (graph_x) and current
        schedule state (x), then returns the empirical frequency of each next location.

        Input:
          - batch (torch_geometric.data.Data): A batch of graph data containing:
              batch.graph_x: current sequence step per person (graph-level scalar), shape=(B, 1)
              batch.x      : current node state indicators per person, shape=(total_nodes, F)

        Output:
          - (torch.Tensor): Predicted probabilities for each node being visited next,
            same shape as batch.x.
        """
        graph_xs = batch.graph_x  # Current sequence position for each person in the batch
        xs = batch.x.reshape((graph_xs.shape[0], -1))  # Flatten node features to (batch, n_nodes * F)

        # For each person, look up their conditional probability vector from the schedule database
        ys = torch.cat([self._predict(graph_x.item(), x) for graph_x, x in zip(graph_xs, xs, strict=True)])
        return ys.reshape(batch.x.shape)  # Restore to original batch shape

    def _predict(self, graph_x, x):
        """
        Description: Internal method that computes the empirical conditional probability for a
        single person. Filters the precomputed schedules DataFrame to rows where the sequence
        position equals graph_x AND the current state matches x, then computes what fraction
        of those rows have each node as the NEXT location.

        Input:
          - graph_x (int/float): The current sequence step (which step in the schedule, as a scalar).
          - x (torch.Tensor): The current node state vector for this person, flat 1D tensor.

        Output:
          - (torch.Tensor): Probability vector of shape (1, n_nodes) representing the empirical
            probability that each node is visited next, given the current state.
        """
        df = self.all_schedule_graphs  # Precomputed DataFrame of all possible schedules

        # Column selectors: "from_" columns = current state, "to_" columns = next state
        x_cols = [pl.col(col) for col in df.columns if col.startswith("from_")]
        y_cols = [pl.col(col) for col in df.columns if col.startswith("to_")]
        # Build filter expressions: check if each "from_" column matches the person's current state
        exprs = [col == x for col, x in zip(x_cols, x)]

        # Filter to rows that match the current sequence step AND current node state
        conditioned = df.filter(pl.col("sequence_num") == graph_x, *exprs)
        # Compute the NEW visits (to_col - from_col gives only newly added nodes at this step)
        conditioned = conditioned.select(*[y_col - x_col for x_col, y_col in zip(x_cols, y_cols)])
        # Compute empirical probabilities as fraction of matching schedules that visit each node
        probs = conditioned.sum() / len(conditioned)

        # If strict mode and no matching schedules, fall back to uniform distribution
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
