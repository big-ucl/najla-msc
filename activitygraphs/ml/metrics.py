"""
Diagnostic metrics: contiguity-hop distance and hop-band ranking metrics.

Motivation
----------
Global ranking metrics (Recall@k, NDCG@k) are dominated by nodes close to
the user's home, because those nodes are both commonly visited *and* already
well-predicted by the graph structure.  To detect whether a model also
learns patterns for locations far from home (beyond its message-passing
receptive field), ``ActivityGraphModule`` groups each node by its
*contiguity-hop distance* from the user's home node and evaluates Recall@k
and NDCG@k separately within each group (hop band).

Contents
--------
DEFAULT_HOP_BANDS          -- Predefined hop-distance bands used in experiments.
compute_home_hop_distance  -- Compute the all-pairs shortest-path distance matrix.
build_hop_band_metrics     -- Build one torchmetrics collection per hop band.
RetrievalRPrecision        -- Custom metric: recall at R (user-specific cutoff).
"""

from collections.abc import Collection

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path
from torchmetrics import MetricCollection, Metric
from torchmetrics.retrieval import RetrievalNormalizedDCG, RetrievalRecall

# Hop bands: (label, low_hops_inclusive, high_hops_inclusive)
# These five bands partition the full range of possible hop distances.
# "0-2" covers the immediate neighbourhood; "13+" covers distant nodes.
DEFAULT_HOP_BANDS: tuple[tuple[str, float, float], ...] = (
    ("0-2", 0, 2),
    ("3-5", 3, 5),
    ("6-8", 6, 8),
    ("9-12", 9, 12),
    ("13+", 13, float("inf")),
)


def compute_home_hop_distance(edge_index: torch.Tensor, num_nodes: int) -> np.ndarray:
    """
    Description: Compute the all-pairs shortest hop-distance matrix for the
    network graph.

    The hop distance between two nodes is the minimum number of edges in any
    path connecting them (i.e. the number of "hops" in the graph).  The
    graph is treated as undirected (edges can be traversed in both directions)
    and unweighted (each edge costs 1 hop regardless of physical length).

    This matrix is used by ``ActivityGraphModule._update_hop_band_metrics``
    to determine how far each node is from the user's home in terms of graph
    structure, not Euclidean distance.

    Input:
      - edge_index (torch.Tensor): PyG COO edge index, shape [2, num_edges].
        Row 0 = source node indices; row 1 = destination node indices.
      - num_nodes (int): Total number of nodes in the graph.

    Output:
      - (np.ndarray): Float matrix, shape [num_nodes, num_nodes].
        Entry ``[u, v]`` = number of hops from node ``u`` to node ``v``.
        Unreachable pairs have value ``inf``.
    """
    # Extract source and destination arrays from the COO edge index
    src = edge_index[0].cpu().numpy()   # source node indices [num_edges]
    dst = edge_index[1].cpu().numpy()   # destination node indices [num_edges]

    # Build a scipy sparse adjacency matrix in COO format
    # All edge weights are 1 (unweighted graph)
    adjacency = coo_matrix((np.ones(len(src)), (src, dst)), shape=(num_nodes, num_nodes))

    # Compute all-pairs shortest paths using Dijkstra on the undirected graph
    return shortest_path(adjacency, method="D", unweighted=True, directed=False)


def build_hop_band_metrics(hop_bands: Collection[tuple[str, float, float]], k: int) -> torch.nn.ModuleDict:
    """
    Description: Build one torchmetrics ``MetricCollection`` per hop band,
    each containing ``recall@k`` and ``ndcg@k``.

    The returned ``ModuleDict`` maps each hop-band label to its collection.
    Each collection's logged keys are prefixed with ``test_hop_{label}_`` so
    they appear in the test results row as e.g.
    ``test_hop_3-5_recall@5``, ``test_hop_3-5_ndcg@5``.

    ``empty_target_action="skip"`` means that if a user has no positive nodes
    in a band, they are skipped (not counted as 0 recall), which avoids
    biasing the band-level average towards zero on sparse bands.

    Input:
      - hop_bands (Collection): Iterable of ``(label, low, high)`` tuples.
        ``label`` is a display string; ``low`` and ``high`` are inclusive
        hop-count bounds.
      - k (int): Rank cutoff for Recall@k and NDCG@k.

    Output:
      - (torch.nn.ModuleDict): Mapping from hop-band label string to its
        ``MetricCollection``.  Registered with PyTorch so metrics move to
        the correct device automatically.
    """
    return torch.nn.ModuleDict({
        label: MetricCollection(
            {
                f"recall@{k}": RetrievalRecall(top_k=k, empty_target_action="skip"),
                f"ndcg@{k}": RetrievalNormalizedDCG(top_k=k, empty_target_action="skip"),
            },
            # Prefix ensures logged key names encode the band and metric name
            prefix=f"test_hop_{label}_",
        )
        for label, _, _ in hop_bands
    })


class RetrievalRPrecision(Metric):
    """
    Description: Custom torchmetric computing per-user R-Precision (also
    called "recall at R").

    For each user ``i``:
      1. Let ``R = |RG_i|`` be the number of locations the user actually
         visited (the "realised set size").
      2. Take the model's top-R predicted nodes.
      3. Compute ``hits / R``: the fraction of the user's visited locations
         that appear in the top-R predictions.
      4. Average over all users with ``R > 0``.

    Why R-Precision instead of fixed @5?
    ``k = 5`` makes sense for Geneva (where most users visit 2–4 locations)
    but would be very tight for Toronto (where users visit ~26 locations on
    average).  By setting ``k = R`` per user, this metric stays interpretable
    and comparable across datasets.

    Attributes:
      - higher_is_better (bool): True — higher R-Precision is better.
      - score_sum (Tensor): Accumulated sum of per-user R-Precision scores.
      - n_users (Tensor): Number of users with at least one positive.
    """

    higher_is_better = True  # higher R-Precision is better

    def __init__(self) -> None:
        """
        Description: Initialise the metric with two accumulators.

        Output:
          - (RetrievalRPrecision): Ready metric, reset to zero state.
        """
        super().__init__()
        # Running sum of per-user R-Precision scores (numerator for the average)
        self.add_state("score_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        # Running count of users who had at least one positive label
        self.add_state("n_users", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, indexes: torch.Tensor) -> None:
        """
        Description: Update the accumulators for a batch of predictions.

        Processes each user separately to allow user-specific cutoffs.

        Input:
          - preds (torch.Tensor): Predicted scores (probabilities or logits)
            for all nodes in the batch.  Shape [num_nodes].
          - target (torch.Tensor): Binary labels (0/1) for all nodes.
            Shape [num_nodes].
          - indexes (torch.Tensor): User ID for each node, used to group
            nodes by user.  Shape [num_nodes].

        Output:
          - None (updates accumulated state).
        """
        for idx in torch.unique(indexes):
            # Isolate this user's nodes
            mask = indexes == idx
            t = target[mask]           # binary labels for this user's nodes
            r = int(t.sum())           # R = number of truly visited nodes
            if r == 0:
                continue               # skip users with no positive labels
            # Select the top-R predicted nodes (capped at total node count)
            top = preds[mask].topk(min(r, t.numel())).indices
            # Compute this user's recall: how many of the top-R are actual visits?
            self.score_sum = self.score_sum + t[top].sum() / r
            self.n_users = self.n_users + 1

    def compute(self) -> torch.Tensor:
        """
        Description: Compute the average R-Precision across all users seen
        so far.

        Output:
          - (torch.Tensor): Scalar average R-Precision.  Returns 0 if no
            users with positives have been seen.
        """
        # Divide accumulated score by number of users (clamp to avoid division by zero)
        return self.score_sum / self.n_users.clamp(min=1)
