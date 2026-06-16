"""
ActivityGraphModule: PyTorch Lightning training loop for GNN-based activity prediction.

This module defines the ``ActivityGraphModule`` class, which wraps *any*
graph neural network model (e.g. ``GATSkip``) in a Lightning
``LightningModule`` to handle:

  - **Training**: weighted binary cross-entropy loss with optional L1
    regularisation.
  - **Validation / Test metrics**: per-epoch ranking metrics (R-Precision,
    Recall@k, NDCG@k, Precision@k), calibration error, and optional
    hop-band metrics that measure performance grouped by graph distance
    from the user's home node.
  - **Popularity injection**: optionally adds a per-node visit frequency
    term to the model output (as either a feature column or a logit offset).
  - **Optimiser**: AdamW with optional ReduceLROnPlateau scheduling.

Helper functions
----------------
  ``extract_features``          -- Assembles the full node feature matrix from a batch.
  ``extracted_features_dim``    -- Computes the feature width for that assembly.
  ``create_pop_logit_column``   -- Builds a per-node popularity column aligned to a batch.
  ``_compute_node_index_within_graph`` -- Maps each node to its intra-graph index 0..N-1.
"""

from collections.abc import Collection
from typing import Literal, cast

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric as pyg
from torch import Tensor
from torchmetrics import MeanMetric, MetricCollection, SumMetric
from torchmetrics.classification import BinaryCalibrationError
from torchmetrics.retrieval import RetrievalNormalizedDCG, RetrievalPrecision, RetrievalRecall

from activitygraphs.ml.dataset import ActivityDataset
from activitygraphs.ml.metrics import DEFAULT_HOP_BANDS, build_hop_band_metrics, RetrievalRPrecision
from activitygraphs.ml.sampling import poisson_sampling


def extract_features(
    batch: pyg.data.Data | pyg.data.Batch,
    full_info: bool,
    use_demographics: bool = True,
    pop_logit: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Description: Assemble the final node feature matrix that will be fed to
    the GNN model.

    The feature matrix is built by optionally concatenating extra columns to
    the base node features ``batch.x``:

      1. ``batch.x`` — always included.  Contains network features and the
         per-user spatial features (including the ``is_home`` indicator).
      2. **Pop-logit column** (optional): if ``pop_logit`` is provided and
         ``pop_mode == "feature"``, the z-scored per-node popularity logit is
         appended.
      3. **Demographics** (optional): if ``use_demographics`` is True, the
         per-user demographic vector (``batch.graph_x``) is broadcast from
         one row per graph to one row per node and concatenated.
      4. **Distance from home** (optional): if ``full_info`` is True, the
         per-node distance-from-home column (``batch.distances``) is appended
         as the last feature.  Used by the distance-augmented MLP baseline
         only; kept behind a flag so it does not leak into the GNN models.

    Input:
      - batch (pyg.data.Data | pyg.data.Batch): A single graph or a batched
        collection of graphs.
      - full_info (bool): Whether to include the distance-from-home column.
      - use_demographics (bool): Whether to broadcast and concatenate user
        demographics (default True).
      - pop_logit (torch.Tensor | None): Popularity logit vector of shape
        [num_nodes_in_network].  Pass ``None`` to skip this column.

    Output:
      - (torch.Tensor): Assembled node feature matrix,
        shape [total_nodes, assembled_feature_dim].
    """
    # Start with the base node features (network + spatial already concatenated)
    x = cast(torch.Tensor, batch.x)

    # Optionally prepend a z-scored popularity-logit column
    if pop_logit is not None:
        pop_logit_col = create_pop_logit_column(pop_logit, batch, standardize=True)
        x = torch.cat([x, pop_logit_col], dim=-1)

    # Optionally broadcast per-graph demographics to per-node and concatenate
    if use_demographics and batch.graph_x is not None:
        if batch.batch is not None:
            # Batched: batch.batch maps each node to its graph index
            demo = batch.graph_x[batch.batch]  # shape: [total_nodes, num_demo_features]
        else:
            # Single graph: expand the single row to all nodes
            demo = batch.graph_x.expand(x.shape[0], -1)
        x = torch.cat([x, demo], dim=-1)

    # If full_info mode is off, return early without the distance column
    if not full_info:
        return x

    # Append the per-node distance-from-home as the last column
    return torch.cat([x, batch.distances.to(x.dtype)], dim=-1)


def extracted_features_dim(
    dataset: ActivityDataset, use_demographics: bool = True, full_info: bool = False, use_pop_feature: bool = False
) -> int:
    """
    Description: Compute the width (number of columns) of the feature matrix
    that ``extract_features`` would produce for this dataset.

    Use this to determine ``in_channels`` when constructing a model so the
    first layer size matches the assembled feature dimension.

    Input:
      - dataset (ActivityDataset): The dataset whose feature dimensions are
        used for the calculation.
      - use_demographics (bool): Whether demographics will be concatenated
        (same flag as in ``extract_features``).
      - full_info (bool): Whether the distance-from-home column will be
        appended (same flag as in ``extract_features``).
      - use_pop_feature (bool): Whether the popularity-logit column will be
        prepended (True when ``pop_mode == "feature"``).

    Output:
      - (int): Total number of feature columns per node.
    """
    # Start with the base dataset feature count (network + spatial)
    dim = dataset.num_features

    # Add 1 for the popularity logit column if it will be used as a feature
    if use_pop_feature:
        dim += 1
    # Add the number of demographic columns if demographics will be concatenated
    if use_demographics:
        dim += dataset.demographics.shape[1]
    # Add the distance-from-home column if full_info mode is enabled
    if full_info:
        dim += dataset.distances.shape[-1]

    return dim


def create_pop_logit_column(pop_logit: torch.Tensor, batch, standardize: bool):
    """
    Description: Build a per-node popularity logit column aligned to the
    nodes in ``batch``, shape [num_nodes_in_batch, 1].

    Because the popularity vector is indexed by *network node index* (0 to
    ``num_nodes - 1``), and a batched ``batch.x`` concatenates multiple
    graphs sequentially, we need to map each row in ``batch.x`` back to its
    node index within its graph before indexing into ``pop_logit``.

    If ``standardize`` is True, the column is z-scored using the global mean
    and standard deviation of ``pop_logit`` so it is on a similar scale to
    other features.  This is used when the popularity is injected as an input
    *feature* column.  When used as a logit *offset* (``pop_mode="offset"``)
    it should not be standardised so the scale matches the model's logits.

    Input:
      - pop_logit (torch.Tensor): Per-node popularity log-odds, shape [num_nodes].
      - batch: A PyG Data / Batch object.
      - standardize (bool): If True, z-score the values before returning.

    Output:
      - (torch.Tensor): Column vector of shape [num_nodes_in_batch, 1].
    """
    # Map each row in batch.x to its intra-graph node index, then index pop_logit
    pop_logit_col = pop_logit[_compute_node_index_within_graph(batch, pop_logit.device)]

    # Optionally z-score for use as an input feature
    if standardize:
        pop_logit_col = (pop_logit_col - pop_logit.mean()) / pop_logit.std().clamp(min=1e-6)

    # Add a trailing dimension so the column can be concatenated with x
    return pop_logit_col.unsqueeze(-1)


def _compute_node_index_within_graph(batch, device: torch.device) -> torch.Tensor:
    """
    Description: Compute the intra-graph node index (0 to N-1) for every
    node in the batch, where N is the number of nodes per graph.

    When multiple graphs are stacked into a batch, ``batch.x`` contains rows
    for graph 0 first, then graph 1, etc.  Row ``r`` in ``batch.x`` might
    be node 0 of graph 2.  This function tells us what that *within-graph*
    index is (here: 0).  This is needed to look up per-node quantities (like
    the popularity logit) that are indexed by node position within a graph.

    Input:
      - batch: A PyG Data / Batch object with ``batch.ptr`` and ``batch.batch``
        attributes (only present when batching > 1 graph).
      - device (torch.device): The device to create the index tensor on.

    Output:
      - (torch.Tensor): Integer tensor of shape [num_nodes_in_batch].
        Entry ``r`` is the within-graph node index for row ``r`` of ``batch.x``.
    """
    n = batch.num_nodes  # total number of nodes across all graphs in the batch

    # Single-graph case: all nodes are already indexed 0..N-1
    if getattr(batch, "batch", None) is None:
        return torch.arange(n, device=device)

    # Batched case: subtract the starting offset of each graph from the global index.
    # batch.ptr[g] is the starting row index of graph g in batch.x.
    return torch.arange(n, device=device) - batch.ptr.to(device)[batch.batch]


class ActivityGraphModule(L.LightningModule):
    """
    Description: PyTorch Lightning module wrapping any GNN model for
    node-level binary visit prediction on activity graphs.

    This class manages the full training lifecycle:
      - **Training step**: forward pass → BCE loss (+ optional L1 reg) →
        backprop via AdamW.
      - **Validation step**: BCE loss + ranking metrics (R-Precision,
        Recall@k, NDCG@k, Precision@k) + calibration error.
      - **Test step**: same as validation + hop-band metrics + predicted/true
        set size + Poisson-sampled recall.
      - **Optimiser**: AdamW with optional ReduceLROnPlateau scheduler.

    The model must implement
    ``forward(x, edge_index, edge_attr, batch) → logits [num_nodes, 1]``.

    Attributes (selection)
    ----------------------
    - model: The wrapped GNN (any ``nn.Module`` with the right signature).
    - lr (float): Initial AdamW learning rate.
    - pos_weight (Tensor): Registered buffer — positive-class BCE weight.
    - reg (str | None): Regularisation type (``"l1"`` or ``None``).
    - lambda_reg (float): L1 coefficient.
    - full_info (bool): Whether to include distance-from-home in features.
    - k (int): Rank cutoff for @k metrics.
    - weight_decay (float): AdamW weight decay.
    - schedule_lr (bool): Whether to use ReduceLROnPlateau.
    - val_metrics / test_metrics (MetricCollection): Ranking metrics.
    - val_calibration / test_calibration (BinaryCalibrationError): Calib.
    - home_hop_distance (Tensor | None): Registered buffer — all-pairs hop
      distance matrix used for hop-band metrics.
    - hop_band_test_metrics: Per-hop-band torchmetrics collections.
    - hop_band_bce / hop_band_pos: Per-hop-band BCE / positive-count metrics.
    - test_pred_size / test_true_size (MeanMetric): Expected set size tracking.
    - test_sampled_recall / test_sampled_size (MeanMetric): Poisson sample stats.
    - pop_mode (str): How to inject the popularity signal (``"none"`` /
      ``"offset"`` / ``"feature"``).
    - pop_logit (Tensor | None): Registered buffer — per-node popularity logits.
    """

    pos_weight: torch.Tensor  # registered buffer; annotated so it types as Tensor, not Tensor | Module

    def __init__(
        self,
        model: torch.nn.Module,
        lr: float,
        pos_weight: torch.Tensor,
        reg: str | None = None,
        lambda_reg: float = 0.01,
        full_info: bool = False,
        k: int = 5,
        weight_decay: float = 1e-4,
        schedule_lr: bool = False,
        home_hop_distance: np.ndarray | torch.Tensor | None = None,
        is_home_idx: int | None = None,
        hop_bands: Collection[tuple[str, float, float]] = DEFAULT_HOP_BANDS,
        pop_logit: torch.Tensor | None = None,
        pop_mode: Literal["none", "offset", "feature"] = "none",
    ):
        """
        Description: Initialise the ActivityGraphModule.

        Input:
          - model (torch.nn.Module): Any GNN model with signature
            ``forward(x, edge_index, edge_attr, batch) → logits``.
          - lr (float): Initial AdamW learning rate (e.g. 0.001).
          - pos_weight (torch.Tensor): Scalar positive-class BCE weight
            (computed by ``_compute_training_weights``).
          - reg (str | None): Regularisation type; ``"l1"`` adds L1 norm
            of all model parameters to the loss.
          - lambda_reg (float): L1 regularisation coefficient (default 0.01).
          - full_info (bool): If True, the distance-from-home column is
            appended to node features (for the MLP-distance baseline only).
          - k (int): Rank cutoff for Recall@k, NDCG@k, Precision@k metrics.
          - weight_decay (float): AdamW weight-decay parameter (default 1e-4).
          - schedule_lr (bool): Enable ReduceLROnPlateau scheduling on
            ``val_bce`` (default False).
          - home_hop_distance (np.ndarray | torch.Tensor | None): All-pairs
            hop distance matrix [num_nodes, num_nodes]; enables hop-band
            diagnostics if provided.  Requires ``is_home_idx``.
          - is_home_idx (int | None): Column index of ``is_home`` in ``x``;
            needed by hop-band metrics to locate each user's home node.
          - hop_bands (Collection): Sequence of ``(label, low, high)`` tuples
            defining hop-distance bands for diagnostics.
          - pop_logit (torch.Tensor | None): Per-node popularity log-odds,
            shape [num_nodes].  Required when ``pop_mode != "none"``.
          - pop_mode (str): ``"none"`` — no popularity injection;
            ``"offset"`` — add pop_logit to the model's output logits;
            ``"feature"`` — prepend z-scored pop_logit as input feature.

        Output:
          - (ActivityGraphModule): Initialised Lightning module.
        """
        super().__init__()
        # The GNN model whose parameters are optimised during training
        self.model = model
        # Initial learning rate for AdamW
        self.lr = lr
        # Register pos_weight as a buffer so it moves to the right device with the module
        self.register_buffer("pos_weight", pos_weight)
        # Regularisation type ("l1" or None)
        self.reg = reg
        # L1 penalty coefficient (used when reg == "l1")
        self.lambda_reg = lambda_reg
        # Whether to include distance-from-home in the feature matrix
        self.full_info = full_info
        # Rank cutoff for @k evaluation metrics
        self.k = k
        # AdamW weight-decay parameter
        self.weight_decay = weight_decay
        # Whether to add a ReduceLROnPlateau scheduler
        self.schedule_lr = schedule_lr

        # Metrics: General setup
        # Build a shared MetricCollection with the four ranking metrics;
        # clone it separately for validation and test to avoid state mixing.
        metrics = MetricCollection({
            "r_precision": RetrievalRPrecision(),
            f"recall@{k}": RetrievalRecall(top_k=k, empty_target_action="skip"),
            f"ndcg@{k}": RetrievalNormalizedDCG(top_k=k, empty_target_action="skip"),
            f"precision@{k}": RetrievalPrecision(top_k=k, empty_target_action="skip"),
        })

        # Separate clones so train and test metric states don't interfere
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

        # Calibration error: measures how well predicted probabilities match true frequencies
        self.val_calibration = BinaryCalibrationError(n_bins=15, norm="l1")
        self.test_calibration = BinaryCalibrationError(n_bins=15, norm="l1")

        # Metrics: Optional distance-from-home hop-band metrics (test time only).
        # is_home_idx: which column of x is the binary is_home flag
        self.is_home_idx = is_home_idx
        # hop_bands: list of (label, low_hop, high_hop) tuples
        self.hop_bands = hop_bands
        if home_hop_distance is not None:
            if is_home_idx is None:
                raise ValueError("is_home_idx is required when home_hop_distance is provided.")
            # Register as a non-persistent buffer: moves to device but not saved in checkpoints
            self.register_buffer(
                "home_hop_distance", torch.as_tensor(home_hop_distance, dtype=torch.float), persistent=False
            )
            # One MetricCollection per hop band containing recall@k and ndcg@k
            self.hop_band_test_metrics = build_hop_band_metrics(hop_bands, k)
            # Per-band weighted BCE (node-pooled) and visited-node count, for context alongside the ranking bands.
            self.hop_band_bce = torch.nn.ModuleDict({label: MeanMetric() for label, _, _ in hop_bands})
            self.hop_band_pos = torch.nn.ModuleDict({label: SumMetric() for label, _, _ in hop_bands})
            # Tracks which hop bands actually received positive examples during the test epoch
            self._seen_hop_bands: set[str] = set()
        else:
            # Hop-band metrics disabled
            self.home_hop_distance = None
            self.hop_band_test_metrics = None
            self.hop_band_bce = None
            self.hop_band_pos = None

        # Metrics: Capture predicted and true expected |RG_i| sizes
        # test_pred_size: running mean of sum(sigmoid(logits)) per user (expected visits)
        self.test_pred_size = MeanMetric()
        # test_true_size: running mean of sum(labels) per user (actual visits)
        self.test_true_size = MeanMetric()

        # Metrics: evaluate actual Poisson-sampled sets
        # test_sampled_recall: mean recall of sets sampled via Poisson draws
        self.test_sampled_recall = MeanMetric()
        # test_sampled_size: mean number of nodes selected in each Poisson sample
        self.test_sampled_size = MeanMetric()

        # Training: Population logits injection
        if pop_mode not in ("none", "offset", "feature"):
            raise ValueError(f"pop_mode must be none|offset|feature, got {pop_mode}")
        if pop_mode != "none" and pop_logit is None:
            raise ValueError(f"pop_mode must be none|offset, got {pop_mode}")

        # Store the popularity injection strategy
        self.pop_mode = pop_mode
        if pop_logit is not None:
            # Register as non-persistent buffer (moves to device, not saved in checkpoints)
            self.register_buffer("pop_logit", pop_logit, persistent=False)
        else:
            self.pop_logit = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
        batch: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Description: Delegate to the wrapped GNN model's forward pass.

        Input:
          - x (torch.Tensor): Node feature matrix, shape [num_nodes, in_channels].
          - edge_index (torch.Tensor): Graph connectivity, shape [2, num_edges].
          - edge_attr (torch.Tensor | None): Edge feature matrix.
          - batch (torch.Tensor | None): Batch assignment vector.

        Output:
          - (torch.Tensor): Raw logits (before sigmoid), shape [num_nodes, 1].
        """
        return self.model(x, edge_index, edge_attr, batch)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """
        Description: Compute the BCE loss (+ optional L1 regularisation) for
        one mini-batch and log ``train_loss``.

        Input:
          - batch (pyg.data.Batch): A batched collection of user graphs.
          - batch_idx (int): Index of this mini-batch within the epoch
            (used by Lightning; not used here).

        Output:
          - (torch.Tensor): Scalar loss tensor that Lightning backpropagates.
        """
        # Forward pass: assemble features and compute raw logits
        out = self.compute_logits(batch)
        # Standard (unweighted) BCE loss between logits and binary labels
        loss = F.binary_cross_entropy_with_logits(out, batch.y.float())

        # Optional L1 regularisation: add sum of absolute model parameter values
        if self.reg == "l1":
            l1_norm = sum(p.abs().sum() for p in self.model.parameters())
            loss = loss + self.lambda_reg * l1_norm

        # Log the training loss (aggregated over the whole epoch, not per step)
        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=batch.num_nodes)
        return loss

    def _common_val_test_step(self, batch) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Description: Shared logic for validation and test steps.

        Computes the logits, two BCE variants, sigmoid probabilities, flat
        binary targets, and per-node user IDs.

        Input:
          - batch (pyg.data.Batch): A batched collection of user graphs.

        Output:
          - (tuple): Six tensors:
            - ``out``:         raw logits [num_nodes, 1]
            - ``bce``:         unweighted BCE scalar
            - ``bce_weighted``: weighted BCE scalar
            - ``probs``:       sigmoid probabilities [num_nodes]
            - ``target``:      binary labels [num_nodes] (squeezed to 1D)
            - ``users``:       user-ID per node [num_nodes] (for per-user grouping)
        """
        # Forward pass: get raw logits
        out = self.compute_logits(batch)
        # Unweighted BCE (gives equal weight to positive and negative nodes)
        bce = F.binary_cross_entropy_with_logits(out, batch.y.float())
        # Weighted BCE (emphasises rare positive nodes)
        bce_weighted = F.binary_cross_entropy_with_logits(out, batch.y.float(), pos_weight=self.pos_weight)
        # Convert logits to probabilities in [0, 1]
        probs = out.squeeze(-1).sigmoid()
        # Flatten labels to 1D for metric computation
        target = batch.y.squeeze(-1)
        # Expand user_id from per-graph to per-node using the batch vector
        users = batch.user_id[batch.batch]

        return out, bce, bce_weighted, probs, target, users

    def validation_step(self, batch, batch_idx: int) -> None:
        """
        Description: Compute and accumulate validation metrics for one
        mini-batch.

        Logs BCE loss values and updates ranking/calibration metric states.
        Final metric values are computed and logged in
        ``on_validation_epoch_end``.

        Input:
          - batch (pyg.data.Batch): A batched collection of user graphs.
          - batch_idx (int): Mini-batch index (ignored).

        Output:
          - None (updates metric states and logs loss scalars).
        """
        out, bce, bce_weighted, probs, target, users = self._common_val_test_step(batch)

        # Log both BCE variants (averaged over the epoch, not per step)
        self.log("val_bce", bce, on_step=False, on_epoch=True, batch_size=batch.num_nodes)
        self.log("val_bce_weighted", bce_weighted, on_step=False, on_epoch=True, batch_size=batch.num_nodes)

        # torchmetrics Retrieval* treat preds as probabilities and drop preds <= 0, so feed sigmoid
        # (monotonic, preserves ranking) rather than raw logits.
        self.val_metrics.update(out.squeeze(-1).sigmoid(), target.long(), indexes=users)

        # Update calibration error metric with probabilities and binary labels
        self.val_calibration.update(probs, target.long())

    def on_validation_epoch_end(self) -> None:
        """
        Description: Finalise and log all validation metrics at the end of
        each epoch, then reset metric states for the next epoch.

        Output:
          - None (logs val_* metrics to Lightning logger).
        """
        # Compute and log all ranking metrics (R-Precision, Recall@k, etc.)
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

        # Compute and log the Expected Calibration Error (L1 norm)
        self.log("val_calibration_l1", self.val_calibration.compute())
        self.val_calibration.reset()

    def test_step(self, batch, batch_idx: int) -> None:
        """
        Description: Compute and accumulate all test metrics for one
        mini-batch.

        Extends the validation step with hop-band ranking metrics, predicted
        and true set-size tracking, and Poisson-sampled recall evaluation.
        Final values are logged in ``on_test_epoch_end``.

        Input:
          - batch (pyg.data.Batch): A batched collection of user graphs.
          - batch_idx (int): Mini-batch index (ignored).

        Output:
          - None (updates metric states and logs BCE scalars).
        """
        out, bce, bce_weighted, probs, target, users = self._common_val_test_step(batch)

        # Log BCEs (epoch-level aggregates)
        self.log("test_bce", bce, on_step=False, on_epoch=True, batch_size=batch.num_nodes)
        self.log("test_bce_weighted", bce_weighted, on_step=False, on_epoch=True, batch_size=batch.num_nodes)

        # Update other metrics, feed sigmoid to conform to torchmetrics calling convention (see validation step)
        self.test_metrics.update(out.squeeze(-1).sigmoid(), batch.y.squeeze(-1).long(), indexes=users)

        # Update hop-band metrics (if enabled)
        if self.hop_band_test_metrics is not None:
            self._update_hop_band_metrics(batch, out)

        # Update BCE calibration error
        self.test_calibration.update(probs, target.long())

        # Update true and predicted expected |RG_i| set sizes (per user)
        for idx in torch.unique(batch.user_id):
            m = users == idx                                # mask: nodes belonging to user idx
            self.test_pred_size.update(probs[m].sum())     # sum of predicted probabilities ≈ expected size
            self.test_true_size.update(target[m].sum())    # sum of labels = actual visited count

        # Update sampled set recall and size metrics
        self._update_sampled_sets_metrics(batch, out)

    def _update_hop_band_metrics(self, batch: pyg.data.Batch, out: torch.Tensor) -> None:
        """
        Description: Group nodes by their hop distance from each user's home
        and update the per-hop-band ranking and BCE metrics.

        For each hop band (defined by a label and a [low, high] range of hop
        counts), this method:
          1. Determines each node's hop distance from its user's home node.
          2. Masks nodes that fall within the current band.
          3. Updates the band's ranking metrics (recall, NDCG), BCE, and
             positive-count accumulators.

        Input:
          - batch (pyg.data.Batch): Batched user graphs.
          - out (torch.Tensor): Raw logits [num_nodes, 1] from the model.

        Output:
          - None (updates metric states).
        """
        device = out.device
        # Flatten logits and compute probabilities for ranking
        logits = out.squeeze(-1)
        scores = logits.sigmoid()
        # Binary targets (visited = 1)
        target = batch.y.squeeze(-1).long()

        # Each node's position within its graph is its network node index (graphs share the network order).
        # node_idx[r] = within-graph index of row r in batch.x
        node_idx = torch.arange(batch.num_nodes, device=device) - batch.ptr.to(device)[batch.batch]

        # Identify home nodes via the is_home column
        is_home = batch.x[:, self.is_home_idx] > 0.0

        # For each graph, store which network node index is the home node
        home_idx_per_graph = torch.zeros(batch.num_graphs, dtype=torch.long, device=device)
        home_idx_per_graph[batch.batch[is_home]] = node_idx[is_home]

        # Look up hop distance from home for every node using the all-pairs matrix
        hop = self.home_hop_distance[home_idx_per_graph[batch.batch], node_idx]

        # Per-node user IDs for grouping metrics by user
        user_ids = batch.user_id[batch.batch]

        # Update metrics for each hop band
        for label, low, high in self.hop_bands:
            # Boolean mask: nodes whose hop distance falls within this band
            mask = (hop >= low) & (hop <= high)
            if not mask.any():
                continue

            band_target = target[mask]  # labels for nodes in this band
            # Update ranking metrics (recall@k, NDCG@k) for the band
            self.hop_band_test_metrics[label].update(scores[mask], band_target, indexes=user_ids[mask])

            # Compute per-node BCE for the band (not reduced, for the MeanMetric accumulator)
            band_bce = F.binary_cross_entropy_with_logits(logits[mask], band_target.float(), reduction="none")
            self.hop_band_bce[label].update(band_bce)
            # Count positive examples in this band (to check if the band is non-trivial)
            self.hop_band_pos[label].update(band_target.sum())

            # Record this band as "seen" (has positives) so we can log it at epoch end
            if band_target.sum() > 0:
                self._seen_hop_bands.add(label)

    def _update_sampled_sets_metrics(self, batch, out, n_draws: int = 8):
        """
        Description: Evaluate stochastic Poisson-sampled location sets and
        update recall / size accumulators.

        Draws ``n_draws`` independent Poisson samples from the model's
        predicted probabilities and computes, for each user:
          - ``test_sampled_recall``: fraction of true visits that were
            selected in the sampled set.
          - ``test_sampled_size``: number of nodes in the sampled set.

        Multiple draws reduce variance of the estimate.

        Input:
          - batch: Batched user graphs.
          - out (torch.Tensor): Raw logits [num_nodes, 1].
          - n_draws (int): Number of independent Poisson draws (default 8).

        Output:
          - None (updates metric states).
        """
        logits = out.squeeze(-1)                   # flat logits [num_nodes]
        target = batch.y.squeeze(-1).float()       # binary labels [num_nodes]
        users = batch.user_id[batch.batch]         # user ID per node [num_nodes]
        # Fixed-seed generator for reproducibility of test evaluations
        gen = torch.Generator(device=out.device).manual_seed(42)

        # pos_weight=None under plain-BCE training (the default); pass float(self.pos_weight) only
        # if the model was trained with weighted BCE.
        for _ in range(n_draws):
            # Draw a binary sample: 1 = selected, 0 = not selected (per-node Bernoulli)
            chosen = poisson_sampling(logits, gen, pos_weight=None).squeeze(-1)
            for idx in torch.unique(users):
                m = users == idx          # mask for nodes belonging to user idx
                r = target[m].sum()       # number of true visits for this user
                if r == 0:
                    continue              # skip users who visited no nodes
                # Count how many true visits were included in the sampled set
                hits = (chosen[m] * target[m]).sum()
                self.test_sampled_recall.update(hits / r)          # recall for this draw
                self.test_sampled_size.update(chosen[m].sum())     # size of sampled set

    def on_test_epoch_end(self) -> None:
        """
        Description: Finalise and log all test metrics at the end of the test
        epoch, then reset metric states.

        Logs in order: ranking metrics → hop-band metrics → calibration →
        set-size statistics → sampled-set recall.

        Output:
          - None (logs test_* metrics to Lightning logger).
        """
        # Log global ranking metrics (R-Precision, Recall@k, NDCG@k, Precision@k)
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

        if self.hop_band_test_metrics is not None:
            # Only compute/log hop bands that saw at least one positive (others would have all groups skipped).
            for label in self._seen_hop_bands:
                # Log ranking metrics for this hop band
                self.log_dict(self.hop_band_test_metrics[label].compute())
                # Log mean BCE within this band
                self.log(f"test_hop_{label}_nll", self.hop_band_bce[label].compute())
                # Log total number of positive nodes in this band
                self.log(f"test_hop_{label}_n_pos", self.hop_band_pos[label].compute())
            # Reset all hop-band metric accumulators
            for collection in self.hop_band_test_metrics.values():
                collection.reset()
            for metric in self.hop_band_bce.values():
                metric.reset()
            for metric in self.hop_band_pos.values():
                metric.reset()
            # Clear the set of "seen" bands for the next test run
            self._seen_hop_bands.clear()

        # Log calibration error (L1 Expected Calibration Error)
        self.log("test_calibration_l1", self.test_calibration.compute())
        self.test_calibration.reset()

        # Log mean predicted and true set sizes across all test users
        self.log("test_pred_size", self.test_pred_size.compute())
        self.log("test_true_size", self.test_true_size.compute())
        self.test_pred_size.reset()
        self.test_true_size.reset()

        # Log mean recall and size of Poisson-sampled location sets
        self.log("test_sampled_recall", self.test_sampled_recall.compute())
        self.log("test_sampled_size", self.test_sampled_size.compute())
        self.test_sampled_recall.reset()
        self.test_sampled_size.reset()

    def configure_optimizers(self):
        """
        Description: Set up the AdamW optimiser and optionally a
        ReduceLROnPlateau learning-rate scheduler.

        The scheduler is only added when:
          - ``schedule_lr`` is True, AND
          - ``trainer.overfit_batches == 0`` (not in overfitting-diagnostic mode,
            where validation loss is intentionally not a useful signal).

        Output:
          - If no scheduler: the ``AdamW`` optimiser directly.
          - If scheduler: a dict ``{"optimizer": ..., "lr_scheduler": {...}}``
            as required by Lightning.
        """
        # Build the AdamW optimiser with the configured learning rate and weight decay
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Only add a scheduler if not trying to overfit (i.e. not in diagnostic mode). Validation error is not a useful
        # signal when purposefully overfitting
        if self.trainer.overfit_batches == 0 and self.schedule_lr:
            # Reduce LR by half when val_bce plateaus for 5 consecutive epochs
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_bce"}}

        return optimizer

    def _pop_logits_feature(self) -> torch.Tensor | None:
        """
        Description: Return the popularity logit vector when ``pop_mode ==
        "feature"``, otherwise return ``None``.

        Used by ``compute_logits`` to decide whether to pass pop_logit to
        ``extract_features`` as an additional input column.

        Output:
          - (torch.Tensor | None): ``self.pop_logit`` when mode is "feature",
            else ``None``.
        """
        return self.pop_logit if self.pop_mode == "feature" else None

    def compute_logits(self, batch) -> torch.Tensor:
        """
        Description: Assemble node features and compute the model's output
        logits, optionally adding the popularity logit offset.

        Steps:
          1. Call ``extract_features`` to build the full feature matrix.
          2. Pass through the GNN model to get raw logits.
          3. If ``pop_mode == "offset"``, add the (unscaled) per-node
             popularity logit to the output logits.

        Input:
          - batch (pyg.data.Batch): A batched collection of user graphs.

        Output:
          - (torch.Tensor): Raw logits (before sigmoid), shape [num_nodes, 1].
        """
        # Build the assembled feature matrix (with optional pop-logit feature and demographics)
        x = extract_features(batch, self.full_info, pop_logit=self._pop_logits_feature())
        # Forward pass through the wrapped GNN model
        out = self(x, batch.edge_index, batch.edge_attr, batch.batch)

        # If offset mode: shift every logit by the node's marginal popularity log-odds
        if self.pop_mode == "offset":
            out = out + create_pop_logit_column(self.pop_logit, batch, standardize=False)

        return out
