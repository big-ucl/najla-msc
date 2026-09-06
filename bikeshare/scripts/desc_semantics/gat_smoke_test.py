"""
Smoke test for desc_semantics/gat.py.

Runs all four GAT feature/dataframe settings on one spatial fold and one
hyperparameter setting. Semantic modes use PCA32 as a separate late-fusion
branch, not as GAT node features. This is for pipeline validation only, not
final thesis results.
"""

import gc
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch

# Keep the smoke run fast and aligned with the optimized GAT code.
os.environ.setdefault("GAT_DAY_BATCH", "30")
os.environ.setdefault("GAT_DIAGNOSTIC", "0")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import gat

SMOKE_RESULTS_DIR = gat.RESULTS_DIR / "smoke_test"
SMOKE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SMOKE_FEATURE_SET_MODES = [
    "graph_baseline",
    "graph_minilm",
    "graph_gpt",
    "graph_gpt_small",
]

SMOKE_FOLD = gat.active_folds[0]


def get_smoke_params():
    """Return one small GAT hyperparameter setting for a quick full-pipeline run."""

    params = dict(gat.get_gat_smoke_grid()[0])
    params["max_epochs"] = int(os.environ.get("GAT_SMOKE_MAX_EPOCHS", "3"))
    params["early_stopping_patience"] = int(os.environ.get("GAT_SMOKE_PATIENCE", "2"))
    params["trial_id"] = 0
    params["trial_group"] = "smoke_test_one_fold"
    return params


def batch_count(tensors, split_name):
    """Count non-empty day batches for one split."""

    return sum(1 for count in tensors.batch_counts[split_name] if count > 0)


def run_one_feature_smoke(feature_set_mode, params):
    """Run one feature setting on the first spatial fold."""

    start_time = time.time()
    print("=" * 72)
    print("Smoke feature setting:", feature_set_mode)
    print("Fold id:", SMOKE_FOLD["fold_id"])
    print("Hyperparameters:", params)

    feature_data = gat.build_feature_set_objects(feature_set_mode)
    gat_fold = gat.prepare_hierarchical_gat_fold(feature_data, SMOKE_FOLD)
    tensors = gat.build_fold_tensors(gat_fold)

    print("CT input dim:", gat_fold["ct_input_dim"])
    print("Station input dim:", gat_fold["station_input_dim"])
    print("Semantic input dim:", gat_fold["semantic_input_dim"])
    print("Semantic hidden dim:", gat.SEMANTIC_HIDDEN_DIM if gat_fold["semantic_input_dim"] > 0 else 0)
    print("Text PCA components:", gat.N_TEXT_PCA_COMPONENTS)
    print("Days:", tensors.num_days)
    print("Station rows:", len(tensors.meta_date))
    print("Day batch size:", tensors.day_batch_size)
    print("Static CT features:", tensors.static_ct)
    print("Train batches:", batch_count(tensors, "train"))
    print("Val batches:", batch_count(tensors, "val"))
    print("Test batches:", batch_count(tensors, "test"))

    fold_result, prediction_df, epoch_df = gat.train_one_gat_fold_fast(
        gat_fold=gat_fold,
        tensors=tensors,
        params=params,
        device=gat.device,
        metric_split="test",
        return_predictions=True,
        return_epoch_history=True,
    )

    fold_result["phase"] = "smoke_test"
    fold_result["seconds"] = time.time() - start_time
    fold_result["train_batches_per_epoch"] = batch_count(tensors, "train")
    fold_result["val_batches_per_epoch"] = batch_count(tensors, "val")
    fold_result["test_batches_per_epoch"] = batch_count(tensors, "test")

    prediction_df["phase"] = "smoke_test"
    epoch_df["phase"] = "smoke_test"

    print("Finished:", feature_set_mode)
    print("Best epoch:", fold_result["best_epoch"])
    print("Test_Avg_MSE:", fold_result.get("Test_Avg_MSE"))
    print("Seconds:", round(fold_result["seconds"], 2))

    del feature_data, gat_fold, tensors
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return fold_result, prediction_df, epoch_df


def main():
    """Run the one-fold smoke test for all four GAT feature settings."""

    torch.set_num_threads(2)
    params = get_smoke_params()

    print("Starting desc_semantics GAT smoke test")
    print("Results directory:", SMOKE_RESULTS_DIR)
    print("Feature settings:", SMOKE_FEATURE_SET_MODES)
    print("Fold id:", SMOKE_FOLD["fold_id"])
    print("Text PCA components:", gat.N_TEXT_PCA_COMPONENTS)
    print("Semantic hidden dim:", gat.SEMANTIC_HIDDEN_DIM)
    print("Device:", gat.device)

    fold_rows = []
    prediction_dfs = []
    epoch_dfs = []

    for feature_set_mode in SMOKE_FEATURE_SET_MODES:
        fold_result, prediction_df, epoch_df = run_one_feature_smoke(
            feature_set_mode=feature_set_mode,
            params=params,
        )
        fold_rows.append(fold_result)
        prediction_dfs.append(prediction_df)
        epoch_dfs.append(epoch_df)

        pd.DataFrame(fold_rows).to_csv(
            SMOKE_RESULTS_DIR / "gat_smoke_test_fold_partial.csv",
            index=False,
        )

    fold_df = pd.DataFrame(fold_rows)
    predictions_df = pd.concat(prediction_dfs, axis=0, ignore_index=True)
    epochs_df = pd.concat(epoch_dfs, axis=0, ignore_index=True)

    fold_df.to_csv(SMOKE_RESULTS_DIR / "gat_smoke_test_fold.csv", index=False)
    predictions_df.to_csv(SMOKE_RESULTS_DIR / "gat_smoke_test_predictions.csv", index=False)
    epochs_df.to_csv(SMOKE_RESULTS_DIR / "gat_smoke_test_epochs.csv", index=False)

    print("=" * 72)
    print("Smoke test complete")
    print("Saved:", SMOKE_RESULTS_DIR / "gat_smoke_test_fold.csv")
    print("Saved:", SMOKE_RESULTS_DIR / "gat_smoke_test_predictions.csv")
    print("Saved:", SMOKE_RESULTS_DIR / "gat_smoke_test_epochs.csv")
    print(fold_df[["feature_set_mode", "fold_id", "semantic_input_dim", "semantic_hidden_dim", "best_epoch", "Test_Avg_MSE", "seconds"]])

    return fold_df, predictions_df, epochs_df


if __name__ == "__main__":
    smoke_fold_df, smoke_predictions_df, smoke_epochs_df = main()
