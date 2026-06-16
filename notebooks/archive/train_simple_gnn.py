"""
Module: train_simple_gnn.py

Description:
    A Marimo interactive notebook for quickly training and testing a simple Heterogeneous
    Graph Neural Network (HeteroGNN) on the Geneva PyG activity dataset.

    The model uses multiple stacked HeteroConv layers (each containing a GAT convolution
    per edge type), followed by a shared linear output layer.  It is trained with binary
    cross-entropy loss against visited-location binary labels.

    This notebook is an archive/prototype and is superseded by the full Lightning-based
    training pipeline in the main activitygraphs package.

Dependencies:
    - activitygraphs library (ActivityDataset)
    - PyTorch Geometric (HeteroConv, GATConv, Linear)
    - Marimo, tqdm
"""

import marimo

__generated_with = "0.19.9"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    import torch

    from pathlib import Path

    from activitygraphs.config import load_config

    project_root = Path(mo.notebook_dir().parent)
    cfg = load_config(project_root)


@app.cell
def _():
    """
    Description: Load the per-user annotated PyG ``ActivityDataset`` from disk. Uses the
    ``"stops"`` network variant (one PyG node per PT stop). The dataset contains one
    ``HeteroData`` object per user with node features and visited-location labels.

    Input:
      - (none): depends on ``cfg`` and ``project_root`` from the setup block.

    Output:
      - dataset (ActivityDataset): the loaded PyG dataset (iterable of HeteroData objects).
    """
    from archive.locations import ActivityDataset

    network_name = "stops"  # use the simpler one-node-per-stop network variant

    dataset = ActivityDataset.from_files(cfg.data, project_root, name=network_name)
    dataset
    return (dataset,)


@app.cell
def _():
    """
    Description: Define the HeteroGNN model class — a heterogeneous graph neural network
    that stacks HeteroConv layers (one GATConv per edge type) and applies a shared linear
    output head.  Also imports the required PyG and PyTorch modules.

    Input:
      - (none): depends on ``torch`` from the notebook setup block.

    Output:
      - F (module): ``torch.nn.functional`` for activation functions.
      - HeteroGNN (class): the heterogeneous GNN model class, ready for instantiation.
    """
    import torch.nn.functional as F
    from torch_geometric.nn import HeteroConv, Linear, GATConv

    class HeteroGNN(torch.nn.Module):
        """
        Description:
            A heterogeneous graph neural network that processes multi-relational graphs
            (e.g. the Geneva transport network with multiple node and edge types).

            Each layer applies a separate Graph Attention Convolution (GATConv) for every
            edge type in the graph.  After all layers, a shared linear head maps the
            hidden representations to a single output value per node.

        Input (constructor):
          - metadata (tuple): a (node_types, edge_types) tuple from pyg.HeteroData.metadata().
                node_types is a list of node-type strings.
                edge_types is a list of (src_type, relation, dst_type) triples.
          - hidden_channels (int): dimensionality of the hidden node embedding in each layer.
          - out_channels (int): number of output dimensions per node (1 for binary prediction).
          - num_layers (int): how many HeteroConv layers to stack.
          - output_types (list[str] | None): node types to include in the output. Defaults to
                all node types if None.
        """
        def __init__(self, metadata, hidden_channels, out_channels, num_layers, output_types=None):
            """
            Description: Initialise the HeteroGNN with lazy GATConv layers for each edge type
            and a shared linear output head.

            Input:
              - metadata (tuple): (node_types, edge_types) from pyg.HeteroData.metadata().
              - hidden_channels (int): hidden embedding size for each GATConv layer.
              - out_channels (int): number of output values per node (1 for binary).
              - num_layers (int): number of stacked HeteroConv layers.
              - output_types (list[str] | None): node types to include in the output; defaults
                    to all node types.

            Output:
              - (none): sets self.convs, self.lin, and self.output_types.
            """
            super().__init__()

            # Unpack the two parts of the PyG heterogeneous graph metadata
            node_types, edge_types = metadata

            # Build one HeteroConv layer per requested depth.
            # Each HeteroConv contains one GATConv per edge type.
            # (-1, -1) as in_channels means lazy initialisation (determined at first forward pass).
            self.convs = torch.nn.ModuleList()
            for _ in range(num_layers):
                conv = HeteroConv({
                    edge_type: GATConv((-1, -1), hidden_channels, add_self_loops=False) for edge_type in edge_types
                })
                self.convs.append(conv)

            # Shared linear output head: maps hidden_channels -> out_channels for all node types
            self.lin = Linear(hidden_channels, out_channels)

            # Subset of node types to produce predictions for
            self.output_types = output_types if output_types is not None else node_types

        def forward(self, x_dict, edge_index_dict, edge_attr_dict):
            """
            Description:
                Runs the forward pass of the heterogeneous GNN.

            Input:
              - x_dict (dict[str, Tensor]): node feature tensors keyed by node type.
              - edge_index_dict (dict[tuple, Tensor]): edge index tensors keyed by
                    (src_type, relation, dst_type) triple.
              - edge_attr_dict (dict[tuple, Tensor]): edge attribute tensors keyed by
                    (src_type, relation, dst_type) triple.

            Output:
              - (dict[str, Tensor]): output logits per node type in self.output_types.
                    Each tensor has shape (num_nodes_of_type, out_channels).
            """
            # Apply each stacked HeteroConv layer followed by a leaky-ReLU non-linearity
            for conv in self.convs:
                x_dict = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
                # Leaky ReLU activation applied independently to every node type
                x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}

            # Record how many nodes each output type has (for splitting after cat)
            splits = [x_dict[t].shape[0] for t in self.output_types]

            # Concatenate embeddings of all output node types along the node dimension
            hs = torch.cat([x_dict[t] for t in self.output_types], dim=0)

            # Apply the shared linear head to all nodes at once
            out = self.lin(hs)

            # Split back into per-type tensors and package as a dictionary
            out = torch.split(out, splits)
            out = {t: o for t, o in zip(self.output_types, out)}

            return out

    return F, HeteroGNN


@app.cell
def _(F, HeteroGNN, dataset):
    """
    Description: Instantiate the HeteroGNN model, initialise its lazy weight layers, set up
    the Adam optimiser, and run one full training epoch. The cell also defines the ``train``
    helper function used for each epoch.

    Input:
      - F (module): ``torch.nn.functional`` for the loss computation.
      - HeteroGNN (class): the model class defined in the previous cell.
      - dataset (ActivityDataset): the loaded PyG dataset for metadata and batching.

    Output:
      - data (HeteroData): the first batch from the DataLoader (used for graph metadata).
      - losses (list[float]): per-batch losses from the first training epoch.
      - model (HeteroGNN): the trained model after one epoch.
    """
    from tqdm import tqdm
    from torch_geometric.loader import DataLoader

    # Use GPU if available, otherwise fall back to CPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # DataLoader batches multiple HeteroData graphs into a single large graph for efficiency
    loader = DataLoader(dataset, batch_size=32)
    # Peek at the first batch to obtain graph metadata (node/edge types and shapes)
    data = next(iter(loader)).to(device)

    # Build the model: 2 hidden layers, 64 hidden channels, 1 output per node (binary logit)
    model = HeteroGNN(data.metadata(), hidden_channels=64, out_channels=1, num_layers=2).to(device)

    # Run a single forward pass without gradients to trigger lazy weight initialisation
    # in Linear layers that use (-1) as in_channels.
    with torch.no_grad():  # Initialize lazy modules.
        x_dict = {k: x.float() for k, x in data.x_dict.items()}
        _ = model(x_dict, data.edge_index_dict, data.edge_attr_dict)

    # Adam optimiser with weight decay for regularisation
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)

    def train(model, optimizer, loader):
        """
        Description:
            Runs one full training epoch over the dataset, computing binary cross-entropy
            loss against the visited-location labels and back-propagating gradients.

        Input:
          - model (HeteroGNN): the model to train (modified in-place).
          - optimizer (torch.optim.Optimizer): the optimiser to use for parameter updates.
          - loader (DataLoader): the DataLoader providing batched HeteroData graphs.

        Output:
          - losses (list[float]): the scalar loss value for each batch in this epoch.
        """
        model.train()
        optimizer.zero_grad()  # clear any stale gradients from the previous epoch

        losses = []  # collect per-batch losses for monitoring

        for batch in tqdm(loader):
            batch = batch.to(device)

            # Cast node features to float32 (the dataset stores them in their native dtype)
            x_dict = {k: x.float() for k, x in batch.x_dict.items()}
            # y_dict contains the binary visit labels per node type
            y_dict = {k: y for k, y in batch.y_dict.items()}
            # Forward pass: obtain per-node logits keyed by output node type
            out = model(x_dict, batch.edge_index_dict, batch.edge_attr_dict)

            # Concatenate outputs and labels across all output node types into flat vectors
            out_cat = torch.cat([out[t] for t in model.output_types])
            y_cat = torch.cat([y_dict[t] for t in model.output_types])

            # Binary cross-entropy with logits: numerically stable combined sigmoid + BCE
            loss = F.binary_cross_entropy_with_logits(out_cat.squeeze(), y_cat)
            loss.backward()   # compute gradients
            optimizer.step()  # update model parameters

            losses.append(float(loss.detach()))  # detach to avoid keeping the computation graph

        return losses

    # Run one training epoch and collect the per-batch losses
    losses = train(model, optimizer, loader)
    return data, losses, model


@app.cell
def _(losses):
    """
    Description: Display the total loss (sum of all per-batch losses) from the first
    training epoch as a quick health check. Very high values suggest an initialisation
    problem; values near zero might indicate vanishing gradients.

    Input:
      - losses (list[float]): per-batch BCE losses from the training epoch.

    Output:
      - (none): renders the summed loss value in the notebook UI.
    """
    sum(losses)
    return


@app.cell
def _():
    """
    Description: Empty placeholder cell — reserved for future additions to the notebook.
    Returns nothing and performs no computation.
    """
    return


@app.cell
def _():
    """
    Description: Add self-loops to the graph so that each node aggregates its own
    features during message passing in addition to those of its neighbours. This is a
    common preprocessing step for GCN and GAT models.

    Input:
      - (none): uses ``data`` from the enclosing scope (the first batch from the loader).

    Output:
      - data (HeteroData): the graph with self-loops added to all edge types.
    """
    import torch_geometric.transforms as T

    data = T.AddSelfLoops()(data)
    return (data,)


@app.cell
def _(data, model):
    """
    Description: Run a single forward pass on the self-loop-augmented graph to verify
    that the model still produces valid outputs after the transform. Displays the output
    logit dictionary in the notebook UI.

    Input:
      - data (HeteroData): the self-loop-augmented graph.
      - model (HeteroGNN): the trained model.

    Output:
      - (none): renders the per-node-type output dictionary in the notebook UI.
    """
    _x_dict = {k: x.float() for k, x in data.x_dict.items()}  # cast features to float32

    model(_x_dict, data.edge_index_dict, data.edge_attr_dict)
    return


@app.cell
def _():
    """
    Description: Empty placeholder cell — reserved for future additions to the notebook.
    Returns nothing and performs no computation.
    """
    return


if __name__ == "__main__":
    app.run()
