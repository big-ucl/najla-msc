"""
ActivityDataModule: LightningDataModule wrapping the load_dataset pipeline.

This module bridges the raw data pipeline (``dataset.py``) and the Lightning
training loop (``lightning_module.py``).

``ActivityDataModule`` is the standard way to provide data to a PyTorch
Lightning ``Trainer``.  It encapsulates:
  - Loading / building the ``ActivityDataset``.
  - Splitting into train / validation / test subsets.
  - Fitting and applying ``StandardScaler`` normalisation.
  - Wrapping each subset in a PyG ``DataLoader`` for mini-batch iteration.
  - Computing the positive-class BCE weight and per-node popularity logits
    used by the training loss.
"""

from pathlib import Path

import lightning as L
import torch
import torch_geometric as pyg

from activitygraphs.config import Config
from activitygraphs.ml.dataset import ActivityDataset, FittedScalers, load_dataset


def _compute_training_weights(train_dataset: ActivityDataset) -> torch.Tensor:
    """
    Description: Compute the positive-class weight for Binary Cross-Entropy
    loss from the training split.

    Because only a small fraction of nodes are visited per user (class
    imbalance), BCE with a positive-class weight helps the model pay more
    attention to positive examples.  The weight is defined as:
      ``pos_weight = sqrt(num_negative / num_positive)``

    Using the square root (instead of the raw ratio) gives a softer correction
    that avoids over-emphasising rare positives.

    Input:
      - train_dataset (ActivityDataset): The training split of the dataset.
        Must already have been set up (``setup()`` called).

    Output:
      - (torch.Tensor): A scalar tensor containing the positive-class weight,
        to be passed to ``F.binary_cross_entropy_with_logits``.
    """
    # Get the integer row indices for the training users
    idx = train_dataset.indices()
    # Collect all binary labels for training users: shape [num_train_users, num_nodes, 1]
    y = train_dataset.spatial_labels[idx]
    # Count how many node visits are positive (label == 1)
    num_pos = y.sum()
    # Weight = sqrt(neg / pos); the square root softens the imbalance correction
    return torch.sqrt((y.numel() - num_pos) / num_pos)


def _compute_node_popularity_logit(train_dataset: ActivityDataset) -> torch.Tensor:
    """
    Description: Compute the per-node visit frequency logit (log-odds) over
    the training dataset.

    For each node ``n``, this estimates ``logit(p_n)`` where ``p_n`` is the
    fraction of training users who visited node ``n``.  This is the same
    per-node rate used by ``NodeBaseline`` and can be injected into the model
    as:
      - an *offset* added to the model's output logits, or
      - an *input feature* column prepended to ``x``.

    The idea is to test whether the model learns anything *beyond* the simple
    marginal visit rate of each node.

    Input:
      - train_dataset (ActivityDataset): The training split.

    Output:
      - (torch.Tensor): Shape [num_nodes] — the logit of the visit rate for
        each node.
    """
    # Get the training user indices
    idx = train_dataset.indices()
    # Collect labels for training users: shape [num_train_users, num_nodes, 1]
    y = train_dataset.spatial_labels[idx].float()
    # Average over users (dim 0) → per-node visit rate; squeeze last dim to [num_nodes]
    # Clamp to (0, 1) to avoid log(0)
    p = y.mean(dim=0).squeeze(-1).clamp(1e-6, 1 - 1e-6)

    # Convert visit rate to log-odds: log(p / (1 - p))
    return torch.log(p / (1 - p))


class ActivityDataModule(L.LightningDataModule):
    """
    Description: LightningDataModule that wraps the full data pipeline for
    activity graph visit prediction.

    This class is the standard interface between raw data and the Lightning
    ``Trainer``.  On the first call to ``setup()`` it:
      1. Calls ``load_dataset`` to load/build, split, and scale the
         ``ActivityDataset``.
      2. Computes the positive-class BCE weight (for imbalanced labels).
      3. Computes per-node popularity logits (for the pop-logit injection
         feature / offset).

    It then exposes ``train_dataloader``, ``val_dataloader``, and
    ``test_dataloader`` returning PyG ``DataLoader`` objects that the Trainer
    iterates over during training, validation, and testing respectively.

    All attributes are lazily populated; attempting to access them before
    ``setup()`` raises a ``RuntimeError`` with a clear message.

    Attributes:
      - cfg (Config): Hydra config.
      - val_size (float): Validation fraction of the total dataset.
      - test_size (float): Test fraction of the total dataset.
      - seed (int): Random seed for the split.
      - batch_size (int): Number of user graphs per mini-batch.
      - project_root (Path | None): Optional override for the project root.
      - _train_dataset: Training subset (populated by ``setup()``).
      - _val_dataset: Validation subset.
      - _test_dataset: Test subset.
      - _scalers: Fitted scalers.
      - _pos_weight: Positive-class BCE weight scalar.
      - _pop_logit: Per-node popularity log-odds vector.
    """

    def __init__(
        self,
        cfg: Config,
        val_size: float,
        test_size: float,
        seed: int,
        batch_size: int,
        project_root: Path | None = None,
    ):
        """
        Description: Store configuration; data loading is deferred to ``setup()``.

        Input:
          - cfg (Config): Hydra configuration containing dataset paths.
          - val_size (float): Fraction of total individuals for validation.
          - test_size (float): Fraction of total individuals for testing.
          - seed (int): Random seed for reproducible train/val/test splits.
          - batch_size (int): Number of user graphs per DataLoader mini-batch.
          - project_root (Path | None): Override for project root directory.
            Auto-detected from repo structure if ``None``.

        Output:
          - (ActivityDataModule): Initialised but not yet loaded.
        """
        super().__init__()
        # Store configuration for use in setup()
        self.cfg = cfg
        self.val_size = val_size       # fraction of data for validation
        self.test_size = test_size     # fraction of data for testing
        self.seed = seed               # random seed for the split
        self.batch_size = batch_size   # number of graphs per mini-batch
        self.project_root = project_root  # optional project root override

        # Lazy-loaded data attributes — populated by setup()
        self._train_dataset: ActivityDataset | None = None
        self._val_dataset: ActivityDataset | None = None
        self._test_dataset: ActivityDataset | None = None
        self._scalers: FittedScalers | None = None
        # Positive-class weight for BCE loss (scalar); computed from train labels
        self._pos_weight: torch.Tensor | None = None
        # Per-node visit frequency log-odds; shape [num_nodes]
        self._pop_logit: torch.Tensor | None = None

    def setup(self, stage: str | None = None) -> None:
        """
        Description: Load, split, and scale the dataset.  Idempotent — safe
        to call multiple times (does nothing on subsequent calls).

        This method is called automatically by the Lightning ``Trainer``
        before training, validation, or testing begins.

        Input:
          - stage (str | None): Lightning stage hint (``"fit"``, ``"validate"``,
            ``"test"``, ``"predict"``).  Ignored here as all splits are loaded
            together.

        Output:
          - None (populates ``self._train_dataset``, ``self._val_dataset``, etc.).
        """
        # Guard: skip if setup() has already been called
        if self._train_dataset is not None:
            return

        # Load / build / split / scale the full dataset
        self._train_dataset, self._val_dataset, self._test_dataset, self._scalers = load_dataset(
            self.cfg, self.val_size, self.test_size, self.seed, self.project_root
        )

        assert self._train_dataset is not None

        # Compute the positive-class weight for class-imbalance correction
        self._pos_weight = _compute_training_weights(self._train_dataset)
        # Compute per-node marginal visit log-odds for popularity injection
        self._pop_logit = _compute_node_popularity_logit(self._train_dataset)

    def train_dataloader(self) -> pyg.loader.DataLoader:
        """
        Description: Return a DataLoader that iterates over the training set
        in random order (shuffled).

        Output:
          - (pyg.loader.DataLoader): Shuffled training DataLoader with
            ``batch_size`` graphs per batch and 8 worker processes.
        """
        assert self._train_dataset is not None
        # shuffle=True: randomise the order of users each epoch to improve training
        return pyg.loader.DataLoader(self._train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self) -> pyg.loader.DataLoader:
        """
        Description: Return a DataLoader that iterates over the validation set
        in a fixed order (no shuffling).

        Output:
          - (pyg.loader.DataLoader): Validation DataLoader (not shuffled).
        """
        assert self._val_dataset is not None
        return pyg.loader.DataLoader(self._val_dataset, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self) -> pyg.loader.DataLoader:
        """
        Description: Return a DataLoader that iterates over the test set
        in a fixed order (no shuffling).

        Output:
          - (pyg.loader.DataLoader): Test DataLoader (not shuffled).
        """
        assert self._test_dataset is not None
        return pyg.loader.DataLoader(self._test_dataset, batch_size=self.batch_size, num_workers=8)

    @property
    def train_dataset(self) -> ActivityDataset:
        """
        Description: Expose the training subset.

        Output:
          - (ActivityDataset): Training split (requires ``setup()`` first).
        """
        if self._train_dataset is None:
            raise RuntimeError("Call setup() before accessing train_dataset.")
        return self._train_dataset

    @property
    def val_dataset(self) -> ActivityDataset:
        """
        Description: Expose the validation subset.

        Output:
          - (ActivityDataset): Validation split (requires ``setup()`` first).
        """
        if self._val_dataset is None:
            raise RuntimeError("Call setup() before accessing val_dataset.")
        return self._val_dataset

    @property
    def test_dataset(self) -> ActivityDataset:
        """
        Description: Expose the test subset.

        Output:
          - (ActivityDataset): Test split (requires ``setup()`` first).
        """
        if self._test_dataset is None:
            raise RuntimeError("Call setup() before accessing test_dataset.")
        return self._test_dataset

    @property
    def scalers(self) -> FittedScalers:
        """
        Description: Expose the fitted ``FittedScalers`` container.

        Output:
          - (FittedScalers): Scalers fitted on the training split.
        """
        if self._scalers is None:
            raise RuntimeError("Call setup() before accessing scalers.")
        return self._scalers

    @property
    def pos_weight(self) -> torch.Tensor:
        """
        Description: Expose the positive-class BCE weight.

        Output:
          - (torch.Tensor): Scalar weight to be passed as ``pos_weight`` to
            ``F.binary_cross_entropy_with_logits``.
        """
        if self._pos_weight is None:
            raise RuntimeError("Call setup() before accessing pos_weight.")
        return self._pos_weight

    @property
    def pop_logit(self) -> torch.Tensor:
        """
        Description: Expose the per-node popularity log-odds vector.

        Output:
          - (torch.Tensor): Shape [num_nodes] — logit of the training visit
            rate for each node.
        """
        if self._pop_logit is None:
            raise RuntimeError("Call setup() before accessing pop_logit.")
        return self._pop_logit
