"""
Module: test/test_callbacks.py

Description:
    Unit tests for the EpochMetricsCollector Lightning callback, which records
    per-epoch training/validation metrics and per-test-run test metrics from the
    trainer's callback_metrics dictionary.

    Tests verify:
      - That only train_* and val_* keys are recorded during training epochs.
      - That only test_* keys are recorded during test runs.
      - That the epoch is set correctly (integer during fit, None during test).
      - That sanity-check epochs are not recorded.
      - That epochs with no relevant metric keys are skipped.
"""

import types

import pytest
import torch

from activitygraphs.ml.callbacks import EpochMetricsCollector


def fake_trainer(callback_metrics: dict, sanity_checking: bool = False, current_epoch: int = 0):
    """
    Description:
        Creates a minimal stand-in trainer namespace that exposes only the attributes
        that EpochMetricsCollector reads.  This avoids the overhead of a full PyTorch
        Lightning Trainer in unit tests.

    Input:
      - callback_metrics (dict): dictionary mapping metric name strings to scalar float
            values.  These are converted to scalar torch tensors.
      - sanity_checking (bool): whether to simulate a sanity-check run (epochs during
            sanity checking should be skipped by the collector).  Defaults to False.
      - current_epoch (int): the current trainer epoch number.  Defaults to 0.

    Output:
      - (types.SimpleNamespace): a lightweight object with .callback_metrics,
            .sanity_checking, and .current_epoch attributes.
    """
    return types.SimpleNamespace(
        callback_metrics={key: torch.tensor(val) for key, val in callback_metrics.items()},
        sanity_checking=sanity_checking,
        current_epoch=current_epoch,
    )


class TestEpochMetricsCollector:
    """
    Description:
        Tests for EpochMetricsCollector, a PyTorch Lightning callback that accumulates
        per-epoch training/validation metrics (during fit) and test metrics (at the end
        of a test run) into a list of row dictionaries for later export to a DataFrame.
    """

    def test_fit_row_keeps_train_and_val_only(self):
        """
        Description:
            Verifies that on_train_epoch_end records a row with stage="fit", the
            correct epoch number, and only the train_* and val_* metrics — test_*
            metrics present in callback_metrics at epoch end must be excluded.

        Input:
          Setup: a fake trainer with train_loss, val_bce, val_precision@5, and test_bce
          at epoch 3.

        Assertions:
          - collector.rows has exactly 1 row.
          - row["stage"] == "fit".
          - row["epoch"] == 3.
          - train_loss and val_bce values match.
          - "test_bce" is NOT in the row.
        """
        collector = EpochMetricsCollector()
        trainer = fake_trainer(
            {"train_loss": 0.5, "val_bce": 0.4, "val_precision@5": 0.2, "test_bce": 0.9}, current_epoch=3
        )

        collector.on_train_epoch_end(trainer, None)

        (row,) = collector.rows
        assert row["stage"] == "fit"
        assert row["epoch"] == 3
        assert row["train_loss"] == pytest.approx(0.5)
        assert row["val_bce"] == pytest.approx(0.4)
        assert "test_bce" not in row

    def test_test_row_keeps_test_only_with_null_epoch(self):
        """
        Description:
            Verifies that on_test_end records a row with stage="test", epoch=None
            (test runs have no epoch number), and only the test_* metrics — val_*
            metrics present in callback_metrics must be excluded.

        Input:
          Setup: a fake trainer with val_bce, test_bce, test_precision@5, and
          test_r_precision in callback_metrics.
          Note: ranking metrics like test_r_precision are only populated by
          on_test_epoch_end, so the collector must read them in on_test_end, not earlier.

        Assertions:
          - collector.rows has exactly 1 row.
          - row["stage"] == "test", row["epoch"] is None.
          - test_bce and test_r_precision values match.
          - "val_bce" is NOT in the row.
        """
        collector = EpochMetricsCollector()
        # Ranking metrics (test_r_precision etc.) are only in callback_metrics by on_test_end, after the
        # module's on_test_epoch_end runs; the collector must read them there, not at epoch end.
        trainer = fake_trainer({"val_bce": 0.4, "test_bce": 0.3, "test_precision@5": 0.25, "test_r_precision": 0.5})

        collector.on_test_end(trainer, None)

        (row,) = collector.rows
        assert row["stage"] == "test"
        assert row["epoch"] is None
        assert row["test_bce"] == pytest.approx(0.3)
        assert row["test_r_precision"] == pytest.approx(0.5)
        assert "val_bce" not in row

    def test_sanity_check_epoch_skipped(self):
        """
        Description:
            Verifies that on_train_epoch_end does NOT record any row when the trainer
            is in sanity-checking mode.  The sanity check runs a few validation batches
            before training starts; recording it would corrupt the epoch metrics.

        Input:
          Setup: a fake trainer with sanity_checking=True.

        Assertion:
          - collector.rows remains empty after the call.
        """
        collector = EpochMetricsCollector()
        trainer = fake_trainer({"val_bce": 0.4}, sanity_checking=True)

        collector.on_train_epoch_end(trainer, None)

        assert collector.rows == []

    def test_fit_row_skipped_when_no_train_or_val_metrics(self):
        """
        Description:
            Verifies that on_train_epoch_end does NOT record a row when the callback
            metrics contain no train_* or val_* keys (e.g. only "lr" from the scheduler).
            This avoids polluting the metrics history with empty rows.

        Input:
          Setup: a fake trainer with only {"lr": 0.01} in callback_metrics.

        Assertion:
          - collector.rows remains empty.
        """
        collector = EpochMetricsCollector()
        trainer = fake_trainer({"lr": 0.01})  # no train_/val_ keys

        collector.on_train_epoch_end(trainer, None)

        assert collector.rows == []
