"""
Frequency-based (no learning) baselines for visit prediction.

These baselines assign visit probabilities based purely on empirical
visit-frequency statistics observed in the training data, without any
learned parameters (except the stored logit tensors).

They implement the same forward signature as the GNN models so they can be
evaluated with the standard ``ActivityGraphModule`` / ``evaluate_baseline``
pipeline.

Baselines provided
------------------
UniformBaseline       -- P(visit) = 0.5 for every node and user.
GlobalBaseline        -- P(visit) = global visit rate across all training data.
NodeBaseline          -- P(visit | node = n) = per-node visit frequency.
ConditionalNodeBaseline -- P(visit | node = n, home = h) = per-node frequency
                          conditioned on the user's home node.
"""

import torch
import torch_geometric as pyg


def inverse_sigmoid(prob):
    """
    Description: Convert a probability to its logit (log-odds) representation.

    The logit is the inverse of the sigmoid function:
      ``logit(p) = log(p / (1 - p))``

    This is useful because the ``ActivityGraphModule`` expects raw logits
    (not probabilities) in the model output.

    Input:
      - prob (torch.Tensor): Probability tensor with values in (0, 1).

    Output:
      - (torch.Tensor): Logit tensor of the same shape.
    """
    return torch.log(prob / (1 - prob))


def extract_is_home(x: torch.Tensor, is_home_idx: int) -> torch.Tensor:
    """
    Description: Extract a boolean mask that is True for home nodes.

    Reads column ``is_home_idx`` from the node feature matrix ``x`` and
    converts it to a boolean tensor: any value > 0 is treated as home.

    Input:
      - x (torch.Tensor): Node feature matrix, shape [num_nodes, num_features].
      - is_home_idx (int): Column index of the ``is_home`` binary indicator.

    Output:
      - (torch.Tensor): Boolean tensor of shape [num_nodes]; True = home node.
    """
    return (x[..., is_home_idx] > 0.0).bool()


def compute_ptr_from_batch(batch: torch.Tensor):
    """
    Description: Compute a ``ptr`` array (graph boundary indices) from the
    batch assignment vector.

    ``ptr`` is a standard PyG concept: ``ptr[g]`` is the index of the first
    node belonging to graph ``g`` in the concatenated batch.  This function
    reconstructs ``ptr`` when only the flat ``batch`` vector is available
    (and ``batch.ptr`` may not be set).

    Input:
      - batch (torch.Tensor): Integer batch assignment vector, shape
        [num_nodes].  Entry ``r`` holds the graph index for node ``r``.

    Output:
      - (torch.Tensor): Integer pointer array of shape [num_graphs + 1].
        ``ptr[g]`` = first row index of graph ``g``;
        ``ptr[-1]`` = total number of nodes.
    """
    # Start with zeros; size = num_graphs + 1 (the +2 covers [0, ..., total])
    ptr = torch.zeros(batch.max().item() + 2, dtype=torch.long, device=batch.device)
    # Cumulative sum of node counts per graph gives the graph start positions
    ptr[1:] = batch.bincount().cumsum(0)

    return ptr


class UniformBaseline(torch.nn.Module):
    """
    Description: Simplest possible baseline — assigns P(visit) = 0.5 to every
    node for every user.

    Outputs a logit of 0 for all nodes (sigmoid(0) = 0.5).  Serves as a
    lower-bound: any model that learns something useful should outperform
    this baseline.

    No parameters; no fitting required.
    """

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        Description: Return a logit of 0 (probability 0.5) for every node.

        Input:
          - x (torch.Tensor): Node feature matrix, shape [num_nodes, *].
            Only used to infer the number of nodes and the device.
          - edge_index, edge_attr, batch: Ignored.

        Output:
          - (torch.Tensor): Zero tensor, shape [num_nodes, 1].
        """
        # All zeros → sigmoid(0) = 0.5 for every node
        return torch.zeros(x.shape[0], 1, device=x.device)


class GlobalBaseline(torch.nn.Module):
    """
    Description: Baseline that assigns the same visit probability to every
    node, equal to the *global* average visit rate across all training users
    and nodes.

    Unlike ``UniformBaseline`` (always 0.5), this baseline adapts to the
    actual proportion of node-visits in the training data.  If 10 % of all
    node-user pairs are visited, every node gets logit(0.1) ≈ -2.2.

    Attributes:
      - logit (torch.Tensor | None): The scalar logit computed by ``fit()``.
        ``None`` until ``fit()`` is called.
    """

    def __init__(self):
        """
        Description: Initialise the GlobalBaseline (no parameters to learn).

        Output:
          - (GlobalBaseline): Unfitted baseline (call ``fit()`` before use).
        """
        super().__init__()
        # Will be set to a scalar logit by fit(); None until then
        self.logit = None

    def fit(self, loader: pyg.loader.DataLoader):
        """
        Description: Compute the global node-visit rate from the training
        DataLoader and store it as a logit.

        Input:
          - loader (pyg.loader.DataLoader): Training DataLoader.  The method
            iterates over all batches to count total positives and total nodes.

        Output:
          - (GlobalBaseline): Returns ``self`` for method chaining.
        """
        num_pos = 0     # running count of positive (visited) node-user pairs
        num_total = 0   # running count of all node-user pairs

        for batch in loader:
            num_pos += batch.y.sum().item()
            num_total += batch.y.numel()

        # Clamp to avoid log(0) when computing the logit
        p = torch.tensor(num_pos / num_total).clamp(1e-6, 1 - 1e-6)
        # Convert to logit for use as a raw model output
        self.logit = inverse_sigmoid(p)

        return self

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        Description: Return the global visit-rate logit for every node.

        Input:
          - x (torch.Tensor): Node feature matrix (used only for size / device).
          - edge_index, edge_attr, batch: Ignored.

        Output:
          - (torch.Tensor): Constant logit repeated for every node,
            shape [num_nodes, 1].
        """
        # Expand the scalar logit to [num_nodes, 1] and move to the correct device
        return self.logit.expand(x.shape[0], 1).to(x.device)


class NodeBaseline(torch.nn.Module):
    """
    Description: Baseline that assigns each node its *marginal* visit
    probability observed in the training data, independent of the user.

    For each network node ``n``, the model estimates:
      ``P(visit | node = n) = (number of users who visited n) / (total users)``

    This is a stronger baseline than ``GlobalBaseline`` because it captures
    the fact that some locations (e.g. the city centre) are visited by many
    users while others (e.g. industrial zones) are rarely visited.

    Attributes:
      - num_nodes (int): Number of nodes in the network graph.
      - logits (torch.Tensor | None): Per-node logit vector, shape [num_nodes].
        ``None`` until ``fit()`` is called.
    """

    def __init__(self, num_nodes: int):
        """
        Description: Initialise NodeBaseline.

        Input:
          - num_nodes (int): Number of nodes in the shared network graph.

        Output:
          - (NodeBaseline): Unfitted baseline.
        """
        super().__init__()
        # Fixed number of nodes in the network graph (used for array sizing)
        self.num_nodes = num_nodes
        # Per-node logit vector; set by fit()
        self.logits = None

    def fit(self, loader: pyg.loader.DataLoader):
        """
        Description: Compute per-node visit rates from the training DataLoader
        and store them as logits.

        Input:
          - loader (pyg.loader.DataLoader): Training DataLoader.

        Output:
          - (NodeBaseline): Returns ``self`` for method chaining.
        """
        # Running count of visits for each node
        visit_counts = torch.zeros(self.num_nodes)
        # Running count of how many users had this node in their graph
        graph_counts = torch.zeros(self.num_nodes)

        for batch in loader:
            # Intra-graph node index for each row in batch.x
            node_indices = torch.arange(batch.num_nodes) - batch.ptr[batch.batch]
            # Accumulate visit counts by node index
            visit_counts.scatter_add_(0, node_indices, batch.y.float().squeeze().cpu())
            # Accumulate graph counts (each node appears once per user graph)
            graph_counts.scatter_add_(0, node_indices, torch.ones(batch.num_nodes))

        # Compute per-node visit rate; clamp to avoid log(0)
        p = (visit_counts / graph_counts.clamp(min=1)).clamp(1e-6, 1 - 1e-6)
        # Convert to logits for raw model output compatibility
        self.logits = inverse_sigmoid(p)

        return self

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        Description: Return the pre-computed per-node logit for every node
        in the batch.

        Input:
          - x (torch.Tensor): Node feature matrix [num_nodes, *].
          - edge_index, edge_attr: Ignored.
          - batch (torch.Tensor): Batch assignment vector [num_nodes].

        Output:
          - (torch.Tensor): Per-node logits, shape [num_nodes, 1].
        """
        # Reconstruct the ptr array from the flat batch vector
        ptr = compute_ptr_from_batch(batch)
        # Map each row back to its intra-graph node index
        node_indices = torch.arange(x.shape[0], device=x.device) - ptr[batch]
        # Index the stored logits by the intra-graph node index
        return self.logits.to(x.device)[node_indices].unsqueeze(1)


class ConditionalNodeBaseline(torch.nn.Module):
    """
    Description: Strongest frequency baseline — assigns each node a visit
    probability conditioned on *which node is the user's home*.

    Estimates:
      ``P(visit | node = n, home = h) = count(n visited & home = h) / count(home = h)``

    This creates a ``num_nodes × num_nodes`` lookup table: for every
    possible home node ``h`` there is a separate visit-rate vector over all
    nodes.  Users with the same home location share the same logit vector.

    Attributes:
      - num_nodes (int): Number of nodes in the network graph.
      - is_home_idx (int): Column index of ``is_home`` in the node feature matrix.
      - logits (torch.Tensor | None): Logit matrix, shape [num_nodes, num_nodes]
        (indexed as [node, home]).  ``None`` until ``fit()`` is called.
    """

    def __init__(self, num_nodes: int, is_home_idx: int):
        """
        Description: Initialise ConditionalNodeBaseline.

        Input:
          - num_nodes (int): Number of nodes in the shared network graph.
          - is_home_idx (int): Column index of the binary ``is_home`` feature
            in the node feature matrix ``x``.

        Output:
          - (ConditionalNodeBaseline): Unfitted baseline.
        """
        super().__init__()
        # Number of nodes (determines matrix dimensions)
        self.num_nodes = num_nodes
        # Which column of x holds the is_home binary indicator
        self.is_home_idx = is_home_idx
        # Logit matrix: logits[n, h] = logit(P(visit n | home = h))
        # Shape: [num_nodes, num_nodes] (node, home)
        self.logits = None

    def fit(self, loader: pyg.loader.DataLoader):
        """
        Description: Compute per-node per-home visit rates and store as
        logits in a [num_nodes, num_nodes] matrix.

        Input:
          - loader (pyg.loader.DataLoader): Training DataLoader.

        Output:
          - (ConditionalNodeBaseline): Returns ``self`` for method chaining.
        """
        # Accumulates how many times node n was visited given home h
        visit_counts = torch.zeros(self.num_nodes, self.num_nodes)
        # Accumulates how many users have home h
        graph_counts = torch.zeros(self.num_nodes)

        for batch in loader:
            # Intra-graph node index for every node in the batch
            node_indices = torch.arange(batch.num_nodes) - batch.ptr[batch.batch]
            # Boolean mask: True for home nodes
            home_mask = extract_is_home(batch.x, self.is_home_idx)

            # Process each graph in the batch individually (home is graph-specific)
            for i in range(batch.num_graphs):
                graph_mask = batch.batch == i   # nodes belonging to graph i
                home_nodes = home_mask[graph_mask].nonzero()

                # Skip if no home node found (should not happen with well-formed data)
                if len(home_nodes) == 0:
                    continue

                # Use the first identified home node (there should only be one)
                home = home_nodes[0].item()
                # Accumulate visit counts for each node given this home
                visit_counts[node_indices[graph_mask], home] += batch.y[graph_mask].squeeze().float().cpu()
                # Increment the count for this home node
                graph_counts[home] += 1

        # Compute conditional probabilities: visits / users-with-that-home
        p = visit_counts / graph_counts.clamp(min=1).unsqueeze(0)
        p = p.clamp(1e-6, 1 - 1e-6)   # avoid log(0)
        # Convert to logit matrix for raw output compatibility
        self.logits = inverse_sigmoid(p)

        return self

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        Description: Look up the conditional logit for each node given its
        user's home node.

        Input:
          - x (torch.Tensor): Node feature matrix [num_nodes, *];
            column ``is_home_idx`` is used to identify each user's home.
          - edge_index, edge_attr: Ignored.
          - batch (torch.Tensor): Batch assignment vector [num_nodes].

        Output:
          - (torch.Tensor): Per-node logits, shape [num_nodes, 1].
        """
        # Compute the ptr array (graph boundary indices) from the batch vector
        ptr = compute_ptr_from_batch(batch)
        # Intra-graph node index for each row in x
        node_indices = torch.arange(x.shape[0], device=x.device) - ptr[batch]
        # Boolean mask: True for home nodes
        home_mask = extract_is_home(x, self.is_home_idx)
        # One entry per graph: which node index is the home for that graph?
        home_indices = torch.zeros(batch.max().item() + 1, dtype=torch.long, device=x.device)

        for i in range(batch.max().item() + 1):
            graph_mask = batch == i                    # nodes in graph i
            home_nodes = home_mask[graph_mask].nonzero()
            # Default to node 0 if no home found; otherwise use the first home
            home_indices[i] = 0 if len(home_nodes) == 0 else home_nodes[0].item()

        # For each node, determine its graph's home node index
        node_home = home_indices[batch]

        # Index the logit matrix: logits[node_idx, home_idx]
        return self.logits.to(x.device)[node_indices, node_home.cpu()].unsqueeze(1).to(x.device)
