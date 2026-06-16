"""
Training orchestration: run_experiment, evaluate_baseline, and helpers.

This module provides two top-level functions for running experiments:

``run_experiment``
  Full training loop for a GNN model.  Sets up the Lightning Trainer,
  runs ``trainer.fit()``, optionally runs ``trainer.test()``, and returns
  all per-epoch metrics as a Polars DataFrame.

``evaluate_baseline``
  Evaluates a non-learning baseline model (e.g. ``NodeBaseline``) using
  the same metric infrastructure as the GNN experiments, returning a
  compatible Polars DataFrame.

Both functions accept a ``WandBParams`` dataclass for optional experiment
tracking via Weights & Biases.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import lightning as L
import polars as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from activitygraphs.ml.callbacks import EpochMetricsCollector, OverfitDebugCallback
from activitygraphs.ml.datamodule import ActivityDataModule
from activitygraphs.ml.lightning_module import ActivityGraphModule


@dataclass
class WandBParams:
    """
    Description: Configuration container for optional Weights & Biases
    experiment tracking.

    Pass an instance of this class to ``run_experiment`` or
    ``evaluate_baseline`` to enable logging to W&B.

    Attributes:
      - use_wandb (bool): Master switch — if False, W&B is disabled and a
        local CSV logger is used instead.
      - project (str | None): W&B project name.
      - entity (str | None): W&B entity (team or username).
      - group (str | None): W&B run group (for grouping related runs).
      - dataset_name (str | None): Added as a W&B tag to identify the dataset.
    """

    use_wandb: bool           # whether to use W&B at all
    project: str | None = None    # W&B project name
    entity: str | None = None     # W&B entity / team name
    group: str | None = None      # W&B run group
    dataset_name: str | None = None  # dataset tag for W&B


def evaluate_baseline(
    baseline: torch.nn.Module,
    datamodule: ActivityDataModule,
    name: str,
    k: int = 5,
    wandb_params: WandBParams | None = None,
) -> pl.DataFrame:
    """
    Description: Evaluate a non-learning baseline model using the same
    metric infrastructure as GNN experiments.

    Wraps the baseline in an ``ActivityGraphModule`` (with ``lr=0.0`` since
    no training occurs), runs the validation and test loops, and returns the
    results in the same two-row Polars DataFrame format as ``run_experiment``.

    The returned DataFrame has:
      - One ``stage="fit"`` row with ``val_*`` metrics and ``train_loss=0``.
      - One ``stage="test"`` row with ``test_*`` metrics.

    Input:
      - baseline (torch.nn.Module): A fitted frequency-based baseline model
        (e.g. ``NodeBaseline`` after calling ``.fit()``).
      - datamodule (ActivityDataModule): Must have been set up (``setup()``
        called).
      - name (str): Display name for the baseline (appears in the ``name``
        column of the DataFrame).
      - k (int): Rank cutoff for @k metrics (default 5).
      - wandb_params (WandBParams | None): Optional W&B logging config.

    Output:
      - (pl.DataFrame): Two-row results DataFrame.
    """
    # Wrap the baseline in ActivityGraphModule for metric computation
    # lr=0.0 since no gradient updates are needed
    baseline_module = ActivityGraphModule(
        model=baseline,
        lr=0.0,
        pos_weight=datamodule.pos_weight,
        k=k,
        home_hop_distance=datamodule.train_dataset.home_hop_distance,
        is_home_idx=datamodule.train_dataset.is_home_col_idx,
    )

    # Set up logger: use W&B if configured, else disable logging entirely
    if wandb_params and wandb_params.use_wandb:
        logger: WandbLogger | bool = WandbLogger(
            project=wandb_params.project,
            entity=wandb_params.entity,
            name=name,
            group=wandb_params.group,
            tags=[t for t in [wandb_params.dataset_name, "baseline"] if t],
        )
    else:
        logger = False  # disables Lightning logger

    # Create a minimal Trainer (no training, just validate/test)
    trainer = L.Trainer(logger=logger, enable_progress_bar=False)

    try:
        # Run validation and test loops to compute metrics
        (val_results,) = trainer.validate(baseline_module, datamodule=datamodule)
        (test_results,) = trainer.test(baseline_module, datamodule=datamodule)
    finally:
        # Always close the W&B run even if an exception occurred
        if wandb_params and wandb_params.use_wandb:
            import wandb

            wandb.finish()

    # Build two-row result DataFrame matching the run_experiment format
    fit_row = {"name": name, "stage": "fit", "epoch": 0, "train_loss": 0.0, **val_results}
    test_row = {"name": name, "stage": "test", "epoch": None, **test_results}
    return pl.DataFrame([fit_row, test_row])


def run_experiment(
    model: torch.nn.Module,
    datamodule: ActivityDataModule,
    num_epochs: int = 10,
    verbose: int = 1,
    name: str | None = None,
    lr: float = 0.01,
    reg: str | None = None,
    full_info: bool = False,
    model_save_dir: Path | None = None,
    fast_dev_run: bool = False,
    overfit_batches: int = 0,
    weight_decay: float = 1e-4,
    schedule_lr: bool = False,
    pop_mode: Literal["none", "offset", "feature"] = "none",
    wandb_params: WandBParams | None = None,
    debug: bool = False,
) -> pl.DataFrame:
    """
    Description: Train a GNN model for the specified number of epochs and
    return per-epoch training and evaluation metrics as a Polars DataFrame.

    Steps:
      1. Call ``datamodule.setup()`` (idempotent).
      2. Wrap the model in an ``ActivityGraphModule`` with the correct loss
         weights, metrics, and optimiser config.
      3. Build Lightning callbacks: ``EpochMetricsCollector``,
         ``LearningRateMonitor``, optional ``ModelCheckpoint``,
         optional ``OverfitDebugCallback``.
      4. Configure a logger (W&B or CSV).
      5. Run ``trainer.fit()``.
      6. Optionally run ``trainer.test(ckpt_path="best")`` if a save dir was
         provided (loads best checkpoint by ``val_bce``).
      7. Collect and return all metrics as a long-format DataFrame.

    Input:
      - model (torch.nn.Module): Model to train.  Must implement
        ``forward(x, edge_index, edge_attr, batch) → logits``.
      - datamodule (ActivityDataModule): The data module.  ``setup()`` is
        called internally if not already done.
      - num_epochs (int): Number of full training epochs (default 10).
      - verbose (int): Non-zero enables the Lightning progress bar (default 1).
      - name (str | None): Experiment name for logging and checkpoint filename.
        Defaults to the model's class name.
      - lr (float): Initial AdamW learning rate (default 0.01).
      - reg (str | None): Regularisation type; ``"l1"`` adds L1 weight
        penalty to the training loss.
      - full_info (bool): If True, include the distance-from-home column in
        node features (only for the distance-augmented MLP baseline).
      - model_save_dir (Path | None): If provided, the best checkpoint
        (by ``val_bce``) is saved here and a test run is performed on it.
        Pass ``None`` to skip saving and testing.
      - fast_dev_run (bool): If True, run only 1 training and 1 validation
        batch, then exit immediately.  Returns an empty DataFrame.  Useful
        for checking that the pipeline does not crash.
      - overfit_batches (int): If > 0, overfit on this many fixed batches
        (diagnostic mode).  Learning-rate scheduling is disabled in this mode.
      - weight_decay (float): AdamW weight-decay coefficient (default 1e-4).
      - schedule_lr (bool): Enable ReduceLROnPlateau on ``val_bce``
        (default False).
      - pop_mode (str): Popularity injection mode:
        ``"none"`` — no injection;
        ``"offset"`` — add pop-logit to model output;
        ``"feature"`` — prepend z-scored pop-logit as input feature.
      - wandb_params (WandBParams | None): W&B logging configuration.
        Pass ``None`` or ``use_wandb=False`` for CSV-only logging.
      - debug (bool): If True, attach ``OverfitDebugCallback`` to print
        batch statistics at the start and end of training.

    Output:
      - (pl.DataFrame): Long-format results DataFrame.
        Columns: ``name``, ``stage``, ``epoch``, ``train_loss``,
        ``val_*`` metrics, ``test_*`` metrics.
        Rows: one ``stage="fit"`` row per epoch (with ``val_*`` filled) and
        one ``stage="test"`` row (with ``test_*`` filled); missing columns
        are null.  Empty DataFrame when ``fast_dev_run`` is True.
    """
    # Default name: use the model class name if none given
    name = name or model.__class__.__name__

    # Ensure data is loaded (idempotent)
    datamodule.setup()

    # Retrieve per-node popularity logit only if it will be used
    pop_logit = datamodule.pop_logit if pop_mode != "none" else None

    # Wrap the GNN model in the Lightning module
    lit_model = ActivityGraphModule(
        model=model,
        lr=lr,
        pos_weight=datamodule.pos_weight,
        reg=reg,
        full_info=full_info,
        # Set k to the dataset's median realised size for scale-adaptive evaluation
        k=datamodule.train_dataset.median_realised_size,
        weight_decay=weight_decay,
        schedule_lr=schedule_lr,
        home_hop_distance=datamodule.train_dataset.home_hop_distance,
        is_home_idx=datamodule.train_dataset.is_home_col_idx,
        pop_logit=pop_logit,
        pop_mode=pop_mode,
    )

    # --- Build the callbacks ---

    # collector: gathers per-epoch train_* and val_* metrics into a list
    collector = EpochMetricsCollector()
    # List of callbacks; LearningRateMonitor logs LR each epoch
    callbacks: list[L.Callback] = [collector, LearningRateMonitor(logging_interval="epoch")]

    # ModelCheckpoint: saves the checkpoint with the lowest val_bce
    if model_save_dir is not None:
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(model_save_dir), filename=name, save_last=False, save_top_k=1, monitor="val_bce"
            )
        )

    # Optional diagnostic callback
    if debug:
        callbacks.append(OverfitDebugCallback())

    # Directory for log files (defaults to current directory)
    log_dir = str(model_save_dir) if model_save_dir is not None else "."

    # --- Configure logging ---

    if wandb_params and wandb_params.use_wandb:
        # W&B logger: logs metrics and hyperparameters to the cloud
        logger = WandbLogger(
            project=wandb_params.project,
            entity=wandb_params.entity,
            name=name,
            group=wandb_params.group,
            save_dir=log_dir,
            tags=[wandb_params.dataset_name] if wandb_params.dataset_name else None,
        )
        # Log all hyperparameters for this run to W&B
        logger.log_hyperparams({
            "model": name,
            "dataset": wandb_params.dataset_name,
            "lr": lr,
            "weight_decay": weight_decay,
            "num_epochs": num_epochs,
            "reg": reg,
            "full_info": full_info,
            "schedule_lr": schedule_lr,
            "overfit_batches": overfit_batches,
            "pop_mode": pop_mode,
        })
    else:
        # Fallback: local CSV logger saves metrics to log_dir/name/version_*/metrics.csv
        logger = CSVLogger(save_dir=log_dir, name=name)

    # --- Create trainer and run training ---

    trainer = L.Trainer(
        max_epochs=num_epochs,
        logger=logger,
        callbacks=callbacks,
        accelerator="auto",       # auto-selects GPU/CPU/TPU
        gradient_clip_val=1.0,    # clip gradients to prevent exploding gradients
        enable_progress_bar=bool(verbose),
        enable_model_summary=False,
        fast_dev_run=fast_dev_run,
        overfit_batches=overfit_batches,
    )

    # Run the full training loop (calls setup, train, and validation hooks)
    trainer.fit(lit_model, datamodule=datamodule)

    # --- Process and return results ---

    try:
        # fast_dev_run is only used for pipeline checks; return empty DataFrame
        if fast_dev_run:
            return pl.DataFrame()

        # If a save directory was provided, run the test loop on the best checkpoint
        if model_save_dir is not None:
            trainer.test(lit_model, datamodule=datamodule, ckpt_path="best")

        # Collect all per-epoch rows from the EpochMetricsCollector callback
        results = pl.DataFrame(collector.rows).with_columns(pl.lit(name).alias("name"))

        # Reorder columns: name, stage, epoch first, then all metric columns
        first_cols = ["name", "stage", "epoch"]
        other_cols = [c for c in results.columns if c not in first_cols]
        results = results.select(first_cols + other_cols)

        return results
    finally:
        # Always close the W&B run, even if an exception occurred
        if wandb_params and wandb_params.use_wandb:
            import wandb

            wandb.finish()
