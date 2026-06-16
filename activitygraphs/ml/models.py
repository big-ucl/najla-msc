"""
PyTorch GNN architectures used for node-level activity prediction.

The main model used in this research is ``GATSkip``, which combines:
  1. A node-wise MLP pre-processing step.
  2. Several Graph Attention (GAT) convolution layers that aggregate neighbour
     information (message passing).
  3. A skip connection that feeds the *original* input features directly to the
     final MLP, so the network can always fall back to the raw features.

Other architectures (``GCN``, ``NodeMLP``, ``GCNRes``, ``GraphTransformer``)
are provided for comparison / ablation studies.

Beginner note
-------------
A "GNN layer" (or "graph convolution") updates each node's vector by mixing its
current vector with those of its neighbours.  Stacking several such layers lets
information flow farther across the graph (each layer = one extra hop).
"""

import functools      # used to pre-fill arguments of GATConv (functools.partial)
import itertools      # used for pairwise iteration when building layer lists

import torch
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.nn.conv import GCNConv, GPSConv, GATConv


def build_module_list(
    module_f: torch.nn.Module,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    hidden_channels: int | None = None,
    **module_kwargs,
) -> torch.nn.ModuleList:
    """
    Description: Build a stack of identical neural-network layers with
    the right input/output sizes automatically wired between them.

    Given a layer constructor (e.g. ``torch.nn.Linear`` or ``GATConv``),
    this function creates ``num_layers`` instances arranged so that:
      - The first layer maps  ``in_channels``     → ``hidden_channels``
      - Middle layers map     ``hidden_channels`` → ``hidden_channels``
      - The last layer maps   ``hidden_channels`` → ``out_channels``

    For a single layer (``num_layers=1``) no ``hidden_channels`` is needed
    because the layer goes directly from ``in_channels`` to ``out_channels``.

    Input:
      - module_f (callable): A constructor with signature
        ``(in_size, out_size, **kwargs) → nn.Module``.  For example
        ``torch.nn.Linear`` or ``GATConv``.
      - num_layers (int): How many layers to create.  Must be at least 1.
      - in_channels (int): Width of the input to the very first layer.
      - out_channels (int): Width of the output of the very last layer.
      - hidden_channels (int | None): Width of all intermediate layers.
        Must be provided when ``num_layers > 1``.
      - **module_kwargs: Any extra keyword arguments forwarded to every
        ``module_f`` call (e.g. ``edge_dim=4`` for GAT layers).

    Output:
      - (torch.nn.ModuleList): A list of ``num_layers`` layer objects
        properly registered with PyTorch so their parameters are trainable.
    """
    if num_layers > 1 and hidden_channels is None:
        raise ValueError(f"Hidden channels not provided for {num_layers=} and {hidden_channels=}")

    if num_layers < 1:
        raise ValueError(f"Invalid number of layers: {num_layers=}")

    # Build the full channel size sequence: [in, hidden, hidden, ..., out].
    # itertools.pairwise will then give us consecutive (in, out) pairs for each layer.
    layers = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]

    # Empty container — PyTorch tracks parameters of sub-modules added here
    modules = torch.nn.ModuleList()

    # Iterate over consecutive size pairs and create one layer per pair
    for num_in, num_out in itertools.pairwise(layers):
        modules.append(module_f(num_in, num_out, **module_kwargs))

    return modules


class GCN(torch.nn.Module):
    """
    Description: Multi-layer Graph Convolutional Network (GCN).

    Stacks several graph convolution layers, each of which updates every
    node's feature vector by averaging the vectors of its neighbours (and
    itself).  Between all-but-the-last layer the following operations are
    applied in order:
      1. Dropout (randomly zero out activations to reduce over-fitting).
      2. Graph convolution (message passing over edges).
      3. Optional residual addition (adds the pre-conv vector back in).
      4. Leaky-ReLU activation (non-linearity that allows small negative values).

    The last layer applies only dropout + convolution (no activation), so
    the final output can be fed into a loss function or another module.

    Attributes:
      - residuals (bool): Whether to add skip connections inside the GCN.
      - dropout (float): Dropout probability applied before each conv layer.
      - convs (ModuleList): The list of graph convolution layers.
    """

    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.2,
        residuals: bool = False,
        conv: pyg.nn.conv.MessagePassing = None,
    ):
        """
        Description: Initialise the GCN with the desired depth and widths.

        Input:
          - num_layers (int): Number of graph convolution layers (depth).
          - in_channels (int): Number of input features per node.
          - hidden_channels (int): Feature width of all intermediate layers.
          - out_channels (int): Feature width of the final layer's output.
          - dropout (float): Probability of zeroing a feature during training
            (default 0.2 means 20% of values are dropped).
          - residuals (bool): If True, add the layer's input back to its output
            (requires ``in_channels == hidden_channels`` for size compatibility).
          - conv: The graph convolution class to use.  Defaults to ``GCNConv``.

        Output:
          - (GCN): Initialised GCN model ready to call ``.forward()``.
        """
        super().__init__()

        # Default convolution type is the standard GCN convolution
        conv = GCNConv if conv is None else conv

        if residuals and in_channels != hidden_channels:
            raise ValueError("Number of in channels must match number of hidden channels")

        # Store whether to use residual connections in forward pass
        self.residuals = residuals
        # Store dropout probability for use in forward pass
        self.dropout = dropout
        # Create all convolution layers with proper channel sizes
        self.convs = build_module_list(conv, num_layers, in_channels, out_channels, hidden_channels)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor | None = None, batch=None
    ) -> torch.Tensor:
        """
        Description: Run the input node features through all GCN layers.

        Processes all-but-last layers with dropout + conv + optional residual +
        leaky-ReLU, then the last layer with dropout + conv only.

        Input:
          - x (torch.Tensor): Node feature matrix, shape [num_nodes, in_channels].
          - edge_index (torch.Tensor): Graph connectivity as a pair of index
            lists [2, num_edges] (COO sparse format).
          - edge_attr (torch.Tensor | None): Optional edge feature matrix,
            shape [num_edges, edge_feature_dim].  Pass ``None`` if unused.
          - batch (torch.Tensor | None): Batch assignment vector mapping each
            node to its graph index; used when multiple graphs are stacked.
            Shape [num_nodes].  Pass ``None`` for a single graph.

        Output:
          - (torch.Tensor): Updated node feature matrix,
            shape [num_nodes, out_channels].
        """
        for conv in self.convs[:-1]:
            x_res = x  # save input for optional residual addition

            # Apply dropout: randomly zero out a fraction of node features
            x = F.dropout(x, p=self.dropout, training=self.training)

            # Apply graph convolution (message passing)
            if edge_attr is not None:
                x = conv(x, edge_index, edge_attr)
            else:
                x = conv(x, edge_index)

            # Optional residual connection: add back the pre-conv features
            if self.residuals:
                x = x + x_res

            # Non-linear activation (like ReLU but allows small negative outputs)
            x = F.leaky_relu(x)

        # Final layer: dropout + conv, no activation (raw output for loss/next module)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if edge_attr is not None:
            x = self.convs[-1](x, edge_index, edge_attr)
        else:
            x = self.convs[-1](x, edge_index)

        return x


class NodeMLP(torch.nn.Module):
    """
    Description: Node-wise Multi-Layer Perceptron (MLP).

    Applies the same linear + activation stack independently to each node's
    feature vector.  This is equivalent to a standard MLP run in parallel
    across all nodes — no information flows between nodes (no edges are used).

    Useful as:
      - A pre-processing head before GNN layers (to project features into a
        common hidden space).
      - A post-processing head after GNN layers (to produce the final output).

    Attributes:
      - dropout (float): Probability of zeroing out a feature during training.
      - lins (ModuleList): The list of ``torch.nn.Linear`` layers.
    """

    def __init__(
        self, num_layers: int, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.2
    ):
        """
        Description: Initialise the NodeMLP with the desired depth and widths.

        Input:
          - num_layers (int): Number of linear layers.
          - in_channels (int): Number of input features per node.
          - hidden_channels (int): Width of all intermediate layers.
          - out_channels (int): Width of the final output per node.
          - dropout (float): Dropout probability (default 0.2).

        Output:
          - (NodeMLP): Initialised NodeMLP model.
        """
        super().__init__()

        # Store dropout probability used in forward pass
        self.dropout = dropout
        # Build a list of Linear layers with proper input/output sizes
        self.lins = build_module_list(torch.nn.Linear, num_layers, in_channels, out_channels, hidden_channels)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor | None = None, batch=None
    ) -> torch.Tensor:
        """
        Description: Pass node features through the MLP layers.

        All-but-last layers apply dropout → linear → ReLU.
        The last layer applies dropout → linear (no activation).

        Input:
          - x (torch.Tensor): Node feature matrix, shape [num_nodes, in_channels].
          - edge_index (torch.Tensor): Ignored — present only to match the common
            GNN forward signature.
          - edge_attr (torch.Tensor | None): Ignored — present for API compatibility.
          - batch (torch.Tensor | None): Ignored — present for API compatibility.

        Output:
          - (torch.Tensor): Transformed node features, shape [num_nodes, out_channels].
        """
        for lin in self.lins[:-1]:
            # Apply dropout, then the linear transformation, then ReLU activation
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = lin(x).relu()

        # Final layer: only dropout + linear, no activation
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)

        return x


class FullyConnectedMLP(torch.nn.Module):
    """
    Description: Per-user "omniscient" MLP that sees the entire graph at once.

    Unlike GNNs which process one node at a time (with neighbourhood context),
    this model concatenates *all* node features of a user's graph into a single
    long vector and processes them jointly through an MLP.

    Purpose: serves as an approximate upper-bound on what is achievable with
    a GNN — a GNN with enough layers should approach this performance without
    requiring a fixed graph size at inference time.

    Limitation: requires that every user graph has *exactly* ``num_nodes``
    nodes (which is true here because all users share the same network graph).
    Edge information (``edge_index`` / ``edge_attr``) is ignored.

    Attributes:
      - num_nodes (int): Fixed number of nodes per graph.
      - in_features (int): Number of input features per node.
      - mlp (NodeMLP): The MLP that maps the flat feature vector to per-node
        logits.
    """

    def __init__(
        self,
        num_nodes: int,
        in_features: int,
        hidden_channels: int,
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        """
        Description: Initialise the FullyConnectedMLP.

        Input:
          - num_nodes (int): Fixed number of nodes in every graph.  All batched
            graphs must have exactly this many nodes.
          - in_features (int): Number of input features per node.
          - hidden_channels (int): Width of the MLP's hidden layers.
          - num_layers (int): Number of MLP layers (default 3).
          - dropout (float): Dropout probability (default 0.2).

        Output:
          - (FullyConnectedMLP): Initialised model.
        """
        super().__init__()

        # Number of nodes expected per graph (used for reshape operations)
        self.num_nodes = num_nodes
        # Expected feature dimension per node (used for validation)
        self.in_features = in_features
        # MLP takes flattened all-node features and predicts one logit per node
        self.mlp = NodeMLP(
            num_layers,
            in_channels=num_nodes * in_features,   # input = all nodes concatenated
            hidden_channels=hidden_channels,
            out_channels=num_nodes,                 # output = one score per node
            dropout=dropout,
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor | None = None, batch=None
    ) -> torch.Tensor:
        """
        Description: Flatten the whole graph's node features per user and
        predict a visit score for every node simultaneously.

        Input:
          - x (torch.Tensor): Node feature matrix for all graphs in the batch,
            shape [total_nodes, in_features].  ``total_nodes`` must be a
            multiple of ``num_nodes``.
          - edge_index (torch.Tensor): Ignored.
          - edge_attr (torch.Tensor | None): Ignored.
          - batch (torch.Tensor | None): Ignored.

        Output:
          - (torch.Tensor): Per-node logits (unnormalised log-odds),
            shape [total_nodes, 1].
        """
        # Unpack shape for validation and reshaping
        total_nodes, num_features = x.shape

        if total_nodes % self.num_nodes != 0:
            raise ValueError(f"Expected a multiple of {self.num_nodes} nodes per batch, got {total_nodes}.")
        if num_features != self.in_features:
            raise ValueError(f"Expected {self.in_features} input features, got {num_features}.")

        # Derive how many user graphs are packed in this batch
        num_graphs = total_nodes // self.num_nodes

        # Reshape: [total_nodes, features] → [num_graphs, num_nodes * features]
        # Each row is one user's complete flattened graph feature vector
        flat = x.reshape(num_graphs, self.num_nodes * num_features)

        # Run MLP: [num_graphs, num_nodes * features] → [num_graphs, num_nodes]
        logits = self.mlp(flat, edge_index)  # [num_graphs, num_nodes]

        # Reshape back to the standard [total_nodes, 1] format for compatibility
        return logits.reshape(total_nodes, 1)


class GCNPlus(torch.nn.Module):
    """
    Description: GCN followed by a node-wise MLP post-processing head.

    A simple two-stage architecture:
      1. GCN layers perform graph-based message passing to produce
         node embeddings that incorporate neighbourhood context.
      2. A per-node MLP refines the embeddings to produce the final output.

    This is an ablation-style variant: unlike ``GCNSkip``, there is no skip
    connection from the raw inputs to the output MLP.

    Attributes:
      - gcn (GCN): The graph convolution stage.
      - lin (NodeMLP): The per-node output MLP stage.
    """

    def __init__(
        self, num_gcn: int, num_lin: int, in_channels: int, hidden_channels: int, out_channels: int, dropout=0.2
    ):
        """
        Description: Initialise GCNPlus.

        Input:
          - num_gcn (int): Number of graph convolution layers.
          - num_lin (int): Number of linear (MLP) layers in the post-processing head.
          - in_channels (int): Number of input features per node.
          - hidden_channels (int): Feature width used in both the GCN and MLP.
          - out_channels (int): Number of output features per node (typically 1
            for binary prediction).
          - dropout (float): Dropout probability (default 0.2).

        Output:
          - (GCNPlus): Initialised model.
        """
        super().__init__()

        # GCN stage: maps in_channels → hidden_channels using graph convolutions
        self.gcn = GCN(num_gcn, in_channels, hidden_channels, hidden_channels, dropout)
        # MLP stage: maps hidden_channels → out_channels (no graph info, per-node only)
        self.lin = NodeMLP(num_lin, hidden_channels, hidden_channels, out_channels, dropout)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor | None = None, batch=None
    ) -> torch.Tensor:
        """
        Description: Run features through GCN then MLP.

        Input:
          - x (torch.Tensor): Node feature matrix, shape [num_nodes, in_channels].
          - edge_index (torch.Tensor): Graph connectivity, shape [2, num_edges].
          - edge_attr (torch.Tensor | None): Optional edge features.
          - batch (torch.Tensor | None): Batch assignment vector.

        Output:
          - (torch.Tensor): Per-node logits, shape [num_nodes, out_channels].
        """
        # GCN stage + ReLU activation before handing off to MLP
        x = self.gcn(x, edge_index, edge_attr).relu()
        # MLP stage: refine per-node embeddings into final logits
        x = self.lin(x, edge_index)

        return x


class GCNRes(torch.nn.Module):
    """
    Description: GCN with pre-/post-processing MLPs and residual connections.

    Three-stage architecture:
      1. ``pre_lin``:  MLP that projects raw node features into the hidden space.
      2. ``convs``:    GCN with residual connections in every conv layer (adds the
                       pre-conv vector back after each message-passing step, helping
                       gradients flow through deep stacks).
      3. ``post_lin``: MLP that maps the GCN output to the final predictions.

    Attributes:
      - dropout (float): Dropout probability shared across all sub-modules.
      - pre_lin (NodeMLP): Input projection MLP.
      - convs (GCN): Graph convolution with residuals.
      - post_lin (NodeMLP): Output MLP.
    """

    def __init__(
        self,
        num_pre_layers: int,
        num_gcn_layers: int,
        num_post_layers: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.2,
    ):
        """
        Description: Initialise GCNRes.

        Input:
          - num_pre_layers (int): Number of MLP layers in the input projection.
          - num_gcn_layers (int): Number of GCN layers.
          - num_post_layers (int): Number of MLP layers in the output head.
          - in_channels (int): Number of input features per node.
          - hidden_channels (int): Width used throughout the network.
          - out_channels (int): Width of the final output per node.
          - dropout (float): Dropout probability (default 0.2).

        Output:
          - (GCNRes): Initialised model.
        """
        super().__init__()

        # Shared dropout probability (stored for reference, used inside sub-modules)
        self.dropout = dropout
        # Pre-processing MLP: in_channels → hidden_channels (node-wise)
        self.pre_lin = NodeMLP(num_pre_layers, in_channels, hidden_channels, hidden_channels, dropout)
        # GCN with residual add-backs: hidden_channels → hidden_channels
        self.convs = GCN(num_gcn_layers, hidden_channels, hidden_channels, hidden_channels, dropout, residuals=True)
        # Post-processing MLP: hidden_channels → out_channels (node-wise)
        self.post_lin = NodeMLP(num_post_layers, hidden_channels, hidden_channels, out_channels, dropout)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor | None = None, batch=None
    ) -> torch.Tensor:
        """
        Description: Pass node features through pre-MLP → GCN → post-MLP.

        Input:
          - x (torch.Tensor): Node feature matrix, shape [num_nodes, in_channels].
          - edge_index (torch.Tensor): Graph connectivity, shape [2, num_edges].
          - edge_attr (torch.Tensor | None): Edge features (not used by this model).
          - batch (torch.Tensor | None): Batch assignment vector.

        Output:
          - (torch.Tensor): Per-node logits, shape [num_nodes, out_channels].
        """
        # Stage 1: project raw features into the hidden space without graph info
        x = self.pre_lin(x, edge_index)
        # Stage 2: graph convolutions with residuals (no edge attributes here)
        x = self.convs(x, edge_index)
        # Stage 3: per-node refinement to produce final outputs
        x = self.post_lin(x, edge_index)

        return x


class GCNSkip(torch.nn.Module):
    """
    Description: GCN with a long-range skip connection from input to output MLP.

    Three-stage architecture with an important twist: the raw input features
    ``x`` are saved at the start, then concatenated to the GCN output before
    the final MLP.  This "skip connection" ensures the model can always
    recover information present in the original features even if the GCN
    layers accidentally discard it.

    Data flow::

        x  ──► pre_lin ──► GCN convs ──► cat([GCN_out, x]) ──► post_lin ──► output
        │                                       ▲
        └───────────────────────────────────────┘  (skip)

    Attributes:
      - dropout (float): Dropout probability.
      - pre_lin (NodeMLP): Input projection MLP.
      - convs (GCN): Graph convolution stage.
      - post_lin (NodeMLP): Output MLP whose input width is
        ``hidden_channels + in_channels`` due to the concatenation.
    """

    def __init__(
        self,
        num_pre_layers: int,
        num_gcn_layers: int,
        num_post_layers: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.2,
        residuals: bool = False,
        conv: pyg.nn.conv.MessagePassing | None = None,
    ):
        """
        Description: Initialise GCNSkip.

        Input:
          - num_pre_layers (int): Depth of the pre-processing MLP.
          - num_gcn_layers (int): Number of graph convolution layers.
          - num_post_layers (int): Depth of the output MLP.
          - in_channels (int): Number of input features per node.
          - hidden_channels (int): Feature width for the hidden layers.
          - out_channels (int): Number of outputs per node (typically 1).
          - dropout (float): Dropout probability (default 0.2).
          - residuals (bool): Enable residual connections *inside* the GCN
            layers (default False).
          - conv: Graph convolution class to use (default ``GCNConv``).

        Output:
          - (GCNSkip): Initialised model.
        """
        super().__init__()

        # Dropout probability used in all sub-modules
        self.dropout = dropout
        # Pre-processing MLP: raw features → hidden space
        self.pre_lin = NodeMLP(num_pre_layers, in_channels, hidden_channels, hidden_channels, dropout)
        # GCN: hidden_channels → hidden_channels (with optional residuals)
        self.convs = GCN(
            num_gcn_layers, hidden_channels, hidden_channels, hidden_channels, dropout, residuals=residuals, conv=conv
        )
        # Post-processing MLP: (hidden + original features) → output
        # Input is wider because the skip-concatenation appends the original in_channels
        self.post_lin = NodeMLP(num_post_layers, hidden_channels + in_channels, hidden_channels, out_channels, dropout)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor | None = None, batch=None
    ) -> torch.Tensor:
        """
        Description: Run the skip-connection GCN forward pass.

        Input:
          - x (torch.Tensor): Node feature matrix, shape [num_nodes, in_channels].
          - edge_index (torch.Tensor): Graph connectivity, shape [2, num_edges].
          - edge_attr (torch.Tensor | None): Optional edge features.
          - batch (torch.Tensor | None): Batch assignment vector.

        Output:
          - (torch.Tensor): Per-node logits, shape [num_nodes, out_channels].
        """
        # Save original features — will be concatenated back after the GCN
        x_skip = x

        # Stage 1: MLP projection into hidden space
        x = self.pre_lin(x, edge_index)
        # Stage 2: graph message passing (uses edge features if provided)
        x = self.convs(x, edge_index, edge_attr)

        # Stage 3: concatenate GCN output with saved raw features along the feature axis
        x = torch.cat([x, x_skip], dim=1)
        # Stage 4: output MLP processes the enriched [hidden + raw] representation
        x = self.post_lin(x, edge_index)

        return x


class GATSkip(GCNSkip):
    """
    Description: Primary model — GCNSkip with Graph Attention (GAT) convolutions.

    This is the main architecture used in the research project.  It extends
    ``GCNSkip`` by replacing the standard GCN message-passing with *Graph
    Attention* convolution (GATConv), which:
      - Assigns a learnable *attention weight* to each neighbour.
      - Incorporates *edge features* (e.g. road length, travel time) into
        the attention computation via the ``edge_dim`` argument.

    Compared to vanilla GCN, GAT is more expressive because it can focus on
    the most informative neighbours rather than averaging them equally.

    The skip connection (from ``GCNSkip``) is preserved unchanged.
    """

    def __init__(
        self,
        num_pre_layers: int,
        num_gcn_layers: int,
        num_post_layers: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        edge_dim: int,
        dropout: float = 0.2,
        residuals: bool = False,
    ):
        """
        Description: Initialise GATSkip.

        Input:
          - num_pre_layers (int): Depth of the pre-processing MLP.
          - num_gcn_layers (int): Number of GAT convolution layers.
          - num_post_layers (int): Depth of the output MLP.
          - in_channels (int): Number of input features per node.
          - hidden_channels (int): Feature width of all hidden layers.
          - out_channels (int): Number of output features per node (usually 1).
          - edge_dim (int): Dimensionality of the edge feature vectors; GAT
            uses these in its attention computation.
          - dropout (float): Dropout probability (default 0.2).
          - residuals (bool): Enable residual connections inside the GAT layers
            (default False).

        Output:
          - (GATSkip): Initialised model.
        """
        super().__init__(
            num_pre_layers,
            num_gcn_layers,
            num_post_layers,
            in_channels,
            hidden_channels,
            out_channels,
            dropout,
            residuals,
            # Use GATConv instead of GCNConv; functools.partial pre-fills edge_dim
            # so build_module_list can call GATConv(in, out, edge_dim=edge_dim).
            conv=functools.partial(GATConv, edge_dim=edge_dim),
        )


class GPSLayer(torch.nn.Module):
    """
    Description: A single GPS (General, Powerful, Scalable) graph transformer layer.

    Each GPS layer combines two complementary attention mechanisms:
      1. **Local message-passing** via GATConv: each node attends to its direct
         neighbours in the graph (using edge features).
      2. **Global attention** via Performer: each node can attend to *any* other
         node in the same graph, regardless of graph distance.

    The Performer approximates full self-attention in linear time, making it
    feasible for larger graphs.

    Attributes:
      - conv (GPSConv): The combined local+global convolution layer from
        PyTorch Geometric.
    """

    def __init__(self, hidden_channels, edge_dim, num_heads=4, dropout=0.2):
        """
        Description: Initialise a single GPS layer.

        Input:
          - hidden_channels (int): Feature width (input and output width of
            this layer are both ``hidden_channels``).
          - edge_dim (int): Dimensionality of the edge feature vectors used by
            the local GATConv component.
          - num_heads (int): Number of attention heads for the global Performer
            attention (default 4).
          - dropout (float): Dropout probability applied inside GPSConv
            (default 0.2).

        Output:
          - (GPSLayer): Initialised GPS layer.
        """
        super().__init__()
        # GPSConv combines a local GATConv with global Performer attention
        self.conv = GPSConv(
            channels=hidden_channels,
            # Local component: 1-head GAT with edge features; no self-loops
            # because GPSConv adds the residual connection itself
            conv=GATConv(hidden_channels, hidden_channels, heads=1, edge_dim=edge_dim, add_self_loops=False),
            heads=num_heads,        # number of global attention heads
            dropout=dropout,
            attn_type="performer",  # linear-time approximation to full attention
        )

    def forward(self, x, edge_index, edge_attr, batch):
        """
        Description: Apply local GAT + global Performer attention to the node features.

        Input:
          - x (torch.Tensor): Node feature matrix, shape [num_nodes, hidden_channels].
          - edge_index (torch.Tensor): Graph connectivity, shape [2, num_edges].
          - edge_attr (torch.Tensor): Edge feature matrix, shape [num_edges, edge_dim].
          - batch (torch.Tensor): Batch assignment vector, shape [num_nodes].

        Output:
          - (torch.Tensor): Updated node features, shape [num_nodes, hidden_channels].
        """
        return self.conv(x, edge_index, batch, edge_attr=edge_attr)


class GraphTransformer(torch.nn.Module):
    """
    Description: Full Graph Transformer using the GPS architecture.

    Stacks multiple ``GPSLayer`` blocks (each combining local GAT and global
    Performer attention) between two projection heads:

      1. ``input_proj``:  Linear layer that maps raw node features → hidden space.
      2. ``edge_proj``:   Linear layer that maps raw edge features → hidden space
                          (so the GPS layers see edge features of the same width).
      3. GPS layers:      ``num_layers`` sequential GPS layers.
      4. ``output``:      Two-layer MLP that maps the final hidden state → output.

    This architecture can in principle capture long-range dependencies across
    the entire graph (via global attention) without needing many layers.

    Attributes:
      - input_proj (torch.nn.Linear): Node feature projection.
      - edge_proj (torch.nn.Linear): Edge feature projection.
      - layers (ModuleList): The stack of GPS layers.
      - output (Sequential): The output MLP head.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        edge_dim,
        num_layers=4,
        num_heads=8,
        dropout=0.2,
    ):
        """
        Description: Initialise the Graph Transformer.

        Input:
          - in_channels (int): Number of raw input features per node.
          - hidden_channels (int): Width of all internal representations.
          - out_channels (int): Number of output features per node (typically 1).
          - edge_dim (int): Dimensionality of the raw edge feature vectors.
          - num_layers (int): Number of GPS transformer layers (default 4).
          - num_heads (int): Number of global attention heads per layer (default 8).
          - dropout (float): Dropout probability (default 0.2).

        Output:
          - (GraphTransformer): Initialised model.
        """
        super().__init__()

        # Linear projection: brings raw node features to the hidden dimension
        self.input_proj = torch.nn.Linear(in_channels, hidden_channels)
        # Linear projection: brings raw edge features to the hidden dimension
        # so GPS layers receive edge attributes of the expected width
        self.edge_proj = torch.nn.Linear(edge_dim, hidden_channels)

        # Stack of GPS layers; each layer keeps the same hidden_channels width
        self.layers = torch.nn.ModuleList([
            GPSLayer(hidden_channels, hidden_channels, num_heads, dropout) for _ in range(num_layers)
        ])

        # Output MLP: hidden → hidden (ReLU) → out (no activation)
        self.output = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        """
        Description: Run the full Graph Transformer forward pass.

        Input:
          - x (torch.Tensor): Node feature matrix, shape [num_nodes, in_channels].
          - edge_index (torch.Tensor): Graph connectivity, shape [2, num_edges].
          - edge_attr (torch.Tensor): Edge feature matrix,
            shape [num_edges, edge_dim].
          - batch (torch.Tensor): Batch assignment vector, shape [num_nodes].

        Output:
          - (torch.Tensor): Per-node logits, shape [num_nodes, out_channels].
        """
        # Project node features to the hidden space
        x = self.input_proj(x)
        # Project edge features to the hidden space (consumed by GPS layers)
        edge_attr = self.edge_proj(edge_attr)

        # Apply each GPS layer sequentially; x is updated in place each time
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr, batch)

        # Final MLP: map last hidden state to per-node output logits
        return self.output(x)
