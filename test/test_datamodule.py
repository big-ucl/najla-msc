"""
Module: test/test_datamodule.py

Description:
    Integration smoke tests for ActivityDataModule, which splits the PyG dataset
    (ActivityDataset) into train/val/test subsets and provides DataLoaders for each.

    Tests verify:
      - All individuals are covered exactly once across the three splits (no leakage,
        no omissions).
      - The three splits are pairwise disjoint (no individual appears in two splits).
      - The positive class weight (pos_weight) is positive and is computed only from
        the training split (no data leakage from val/test).

    Marked ``integration`` and skipped automatically when the processed Toronto dataset
    or config is not available in the current environment.
"""

import pytest
import torch

from activitygraphs.config import load_config
from activitygraphs.ml.datamodule import ActivityDataModule
from activitygraphs.utils import get_project_root

# Apply the "integration" mark to all tests in this module so they can be skipped
# in CI environments without the real dataset by running: pytest -m "not integration"
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def datamodule() -> ActivityDataModule:
    """
    Description:
        Module-scoped pytest fixture that loads the Toronto configuration, builds
        the ActivityDataModule and calls setup() to create the three data splits.
        If the Toronto dataset or config is not available (e.g. in CI), the test is
        automatically skipped rather than failing.

    Output:
      - (ActivityDataModule): a configured and set-up ActivityDataModule ready for use.
    """
    root = get_project_root()
    try:
        # Load the toronto config; verbose=False suppresses setup log output in tests
        cfg = load_config(root, verbose=False, data="toronto")
        # val_size=0.2 and test_size=0.1 give a 70/20/10 train/val/test split
        dm = ActivityDataModule(cfg, val_size=0.2, test_size=0.1, seed=42, batch_size=64)
        dm.setup()  # triggers the actual split computation
    except Exception as exc:  # config or processed data not present in this environment
        pytest.skip(f"toronto dataset/config unavailable: {exc}")
    return dm


class TestActivityDataModule:
    """
    Description:
        Integration tests for ActivityDataModule.  All tests in this class use the
        module-scoped `datamodule` fixture, which loads the real Toronto processed
        dataset.  The tests check that:
          (1) Every individual in the dataset is assigned to exactly one split.
          (2) The three splits are pairwise disjoint.
          (3) The positive class weight is correct and computed from training data only.
    """
    def test_split_covers_all_individuals(self, datamodule):
        """
        Description:
            Checks that the total number of individuals across train+val+test equals
            the full dataset size.  Since subsets share the spatial_labels tensor,
            shape[0] gives the total individual count regardless of the split.
        """
        # spatial_labels.shape[0] is the total number of individuals in the full dataset
        total = datamodule.train_dataset.spatial_labels.shape[0]
        # Sum of individuals across all three splits must equal the total
        n_split = len(datamodule.train_dataset) + len(datamodule.val_dataset) + len(datamodule.test_dataset)
        assert n_split == total

    def test_splits_are_disjoint(self, datamodule):
        """
        Description:
            Regression guard that verifies the three splits are pairwise disjoint —
            no individual index should appear in more than one split.  This guards
            against data leakage bugs in the split logic.
        """
        # Retrieve the integer indices belonging to each split
        train = set(datamodule.train_dataset.indices())
        val = set(datamodule.val_dataset.indices())
        test = set(datamodule.test_dataset.indices())
        # All three pairs must have empty intersection
        assert train.isdisjoint(val)
        assert train.isdisjoint(test)
        assert val.isdisjoint(test)

    def test_pos_weight_positive_and_train_only(self, datamodule):
        """
        Description:
            Verifies that (a) the positive-class weight is strictly positive (required for
            BCEWithLogitsLoss) and (b) the weight is computed using only the training split,
            not the val or test data.

            The expected formula is: pos_weight = sqrt((num_neg) / (num_pos)).
        """
        assert datamodule.pos_weight > 0  # must be positive to up-weight rare visits

        # Recompute pos_weight independently from the train split indices
        idx = datamodule.train_dataset.indices()  # integer indices of training samples
        y = datamodule.train_dataset.spatial_labels[idx]  # training labels, shape (n_train, N)
        num_pos = y.sum()                                  # total number of positive (visited) cells
        # The weight formula balances the class imbalance: sqrt(neg/pos)
        expected = torch.sqrt((y.numel() - num_pos) / num_pos)
        torch.testing.assert_close(datamodule.pos_weight, expected)
