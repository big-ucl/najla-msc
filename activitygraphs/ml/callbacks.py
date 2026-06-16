"""
Lightning callbacks for metric collection and training diagnostics.

``EpochMetricsCollector``
  Accumulates per-epoch training and validation metrics into a list of dicts
  that ``run_experiment`` converts into a Polars DataFrame.  Avoids the
  overhead of writing to CSV and re-parsing.

``OverfitDebugCallback``
  Prints summary statistics about the first training batch and the model's
  final output on the last epoch.  Used to diagnose whether the model is
  learning at all during controlled overfitting experiments.
"""

import lightning as L
import torch
import torch_geometric as pyg

from activitygraphs.ml.lightning_module import ActivityGraphModule, extract_features


class OverfitDebugCallback(L.Callback):
    """
    Description: Debug callback that prints batch and model output statistics
    during controlled overfitting experiments.

    Activated when ``debug=True`` is passed to ``run_experiment``.

    At the **first epoch's first batch** it prints:
      - Batch size (number of nodes and graphs).
      - Label statistics (total positives / total elements).
      - Whether any NaN values are present in ``x``.
      - Min / max / mean of ``x``.
      - The positive-class BCE weight.

    At the **last epoch's first batch** it prints:
      - Standard deviation, min, max, and mean of the model's output logits,
        helping verify that the model is not stuck at a constant output.
    """

    def on_train_batch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule, batch: pyg.data.Batch, batch_idx: int
    ) -> None:
        """
        Description: Hook called at the start of every training batch.
        Only acts on ``batch_idx == 0`` (first batch of the epoch) to avoid
        excessive output.

        Input:
          - trainer (L.Trainer): The Lightning trainer.
          - pl_module (L.LightningModule): The wrapped model (expected to be
            ``ActivityGraphModule``).
          - batch (pyg.data.Batch): The current mini-batch.
          - batch_idx (int): Index of this batch within the epoch.

        Output:
          - None (prints diagnostic information to stdout).
        """
        # Only inspect the first batch of each epoch to avoid flooding output
        if batch_idx != 0:
            return

        assert isinstance(pl_module, ActivityGraphModule)

        if trainer.current_epoch == 0:
            # Compute feature statistics to check scaling and presence of NaNs
            b_min, b_max, b_mean = batch.x.min().item(), batch.x.max().item(), batch.x.mean().item()

            print(f"[overfit-debug] num_nodes={batch.num_nodes} num_graphs={batch.num_graphs}")
            print(f"[overfit-debug] y.sum()={batch.y.sum().item()} y.numel()={batch.y.numel()}")
            print(f"[overfit-debug] x.isnan().any()={torch.isnan(batch.x).any().item()}")
            print(f"[overfit-debug] x.min/max/mean={b_min:.4f}/{b_max:.4f}/{b_mean:.4f}")
            print(f"[overfit-debug] pos_weight={pl_module.pos_weight.item()}")

        # On the final epoch, print model output statistics
        if trainer.max_epochs is not None and trainer.current_epoch == trainer.max_epochs - 1:
            with torch.no_grad():
                # Compute logits without storing gradients (diagnostic only)
                x = extract_features(batch, pl_module.full_info)
                out = pl_module.compute_logits(x, batch.edge_index, batch.edge_attr, batch.batch)

            # Print spread and location of output logits; near-zero std suggests a collapsed model
            print(
                f"[overfit-debug-final] out std={out.std().item():.4f} "
                f"min={out.min().item():.4f} max={out.max().item():.4f} mean={out.mean().item():.4f}"
            )


class EpochMetricsCollector(L.Callback):
    """
    Description: Lightning callback that accumulates per-epoch metrics into a
    list of dicts for later conversion to a Polars DataFrame.

    Instead of writing metrics to a CSV file and re-parsing it (the standard
    Lightning CSV logging flow), this callback reads metrics directly from
    ``trainer.callback_metrics`` at the appropriate hook, avoiding file I/O
    and string parsing overhead.

    Layout of ``self.rows``:
      - One ``stage="fit"`` dict per training epoch containing ``train_*``
        and ``val_*`` metrics logged during that epoch.
      - One ``stage="test"`` dict (appended once) containing all ``test_*``
        metrics.  ``epoch`` is set to ``None`` for the test row.

    Attributes:
      - rows (list[dict]): Accumulated metric rows.  Each dict is one row
        of the eventual results DataFrame.
    """

    def __init__(self) -> None:
        """
        Description: Initialise the collector with an empty row list.

        Output:
          - (EpochMetricsCollector): Empty collector, ready to attach to a Trainer.
        """
        # This list collects one dict per epoch during fit, plus one for the test stage
        self.rows: list[dict] = []

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """
        Description: Collect training and validation metrics at the end of
        each training epoch.

        Skipped during Lightning's sanity-check validation to avoid
        including preliminary (incomplete) metrics.

        Input:
          - trainer (L.Trainer): The Lightning trainer.
          - pl_module (L.LightningModule): The Lightning module (not used directly).

        Output:
          - None (appends one dict to ``self.rows`` if metrics were logged).
        """
        # Skip the initial sanity-check pass (partial data, not a real epoch)
        if trainer.sanity_checking:
            return

        # Collect all metrics whose keys start with "train_" or "val_"
        metrics = {
            key: value.item() for key, value in trainer.callback_metrics.items() if key.startswith(("train_", "val_"))
        }
        # Only append if at least one metric was logged this epoch
        if metrics:
            self.rows.append({"stage": "fit", "epoch": trainer.current_epoch, **metrics})

    def on_test_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """
        Description: Collect all test metrics after the test loop completes.

        Important timing note: this hook fires *after* ``on_test_epoch_end``
        in the LightningModule (where the Retrieval* ranking metrics are
        computed and logged).  Collecting here ensures all test metrics —
        including the ranking ones — are present in ``callback_metrics``.

        Input:
          - trainer (L.Trainer): The Lightning trainer.
          - pl_module (L.LightningModule): The Lightning module.

        Output:
          - None (appends one dict with ``stage="test"`` to ``self.rows``).
        """
        # Collect on `on_test_end`, not `on_test_epoch_end`: callback hooks run before the
        # LightningModule's `on_test_epoch_end`, which is where the Retrieval* ranking metrics are
        # logged. Reading at epoch-end would capture only the per-batch test_bce/test_bce_weighted.
        # Collect all metrics whose keys start with "test_"
        metrics = {key: value.item() for key, value in trainer.callback_metrics.items() if key.startswith("test_")}
        # epoch=None for the test row (there is no epoch number associated with testing)
        self.rows.append({"stage": "test", "epoch": None, **metrics})
