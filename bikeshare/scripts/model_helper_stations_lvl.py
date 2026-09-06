"""Shared station-level tools for CatBoost, MLP, and GAT.

MiniLM embedding creation is intentionally kept in the MLP notebook.

Target order:
    column 0 = inflow_count
    column 1 = outflow_count

The station identifier is station_name.
"""

# Import from __future__.
from __future__ import annotations

# Import random.
import random
# Import from pathlib.
from pathlib import Path
# Import from typing.
from typing import Iterable, Sequence

# Import numpy.
import numpy as np
# Import pandas.
import pandas as pd
# Import torch.
import torch
# Import torch.nn.
import torch.nn as nn
# Import from sklearn.compose.
from sklearn.compose import ColumnTransformer
# Import from sklearn.metrics.
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    r2_score,
)
# Import from sklearn.preprocessing.
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Define regression metrics.
REGRESSION_METRICS = (
    "MAE",
    "MSE",
    "RMSE",
    "R2",
    "RMSLE",
)

# Define ranking metrics.
RANKING_METRICS = (
    "Recall@10",
    "NDCG@10",
    "Recall@20",
    "NDCG@20",
)

# Combine all metrics.
TWO_TARGET_METRICS = REGRESSION_METRICS + RANKING_METRICS


# Define BikeShareModelHelper.
class BikeShareModelHelper:
    """Shared model functions."""

    # Set TARGET COLS.
    TARGET_COLS = ("inflow_count", "outflow_count")
    # Set TARGET LABELS.
    TARGET_LABELS = ("Inflow", "Outflow")
    # Set METRIC NAMES.
    METRIC_NAMES = TWO_TARGET_METRICS

    # Define load station level datasets.
    @staticmethod
    def load_station_level_datasets(
        model_df_file: str | Path,
        fold_assignments_file: str | Path,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load the model data and saved folds."""

        # Load the model data.
        df = pd.read_parquet(model_df_file)

        # Load the fold table.
        fold_assignments_df = pd.read_parquet(fold_assignments_file)

        # Return the result.
        return df, fold_assignments_df

    # Define prepare spatial folds from assignments.
    @staticmethod
    def prepare_spatial_folds_from_assignments(
        df: pd.DataFrame,
        fold_assignments_df: pd.DataFrame,
        *,
        group_col: str = "spatial_group",
        fold_col: str = "fold_id",
        split_col: str = "split",
        include_dataframes: bool = False,
    ) -> list[dict]:
        """Build the saved spatial folds."""

        # Store the folds.
        prepared_folds = []

        # Build each fold.
        for fold_id in sorted(fold_assignments_df[fold_col].unique()):
            # Select one fold.
            fold_table = fold_assignments_df[
                fold_assignments_df[fold_col] == fold_id
            ]

            # Get the groups for each split.
            groups_by_split = {
                split: tuple(
                    sorted(
                        fold_table.loc[
                            fold_table[split_col] == split,
                            group_col,
                        ].tolist()
                    )
                )
                for split in ("train", "val", "test")
            }

            # Store the fold details.
            fold = {
                "fold_id": int(fold_id),
                "train_groups": groups_by_split["train"],
                "val_groups": groups_by_split["val"],
                "test_groups": groups_by_split["test"],
            }

            # Get row indices for each split.
            for split in ("train", "val", "test"):
                # Set indices.
                indices = np.flatnonzero(
                    df[group_col].isin(groups_by_split[split]).to_numpy()
                )

                # Set fold.
                fold[f"{split}_idx"] = indices

                # Check the condition.
                if include_dataframes:
                    # Set fold.
                    fold[f"{split}_df"] = df.iloc[indices].copy()

            # Add the result to the list.
            prepared_folds.append(fold)

        # Return the result.
        return prepared_folds

    # Define target transform.
    @staticmethod
    def target_transform(values, *, transform: str = "log1p") -> np.ndarray:
        """Transform the target."""

        # Convert to an array.
        values = np.asarray(values, dtype=float)

        # Apply log1p.
        if transform == "log1p":
            # Return the result.
            return np.log1p(values)

        # Keep the original values.
        return values.copy()

    # Define inverse target transform.
    @staticmethod
    def inverse_target_transform(
        values,
        *,
        transform: str = "log1p",
    ) -> np.ndarray:
        """Return predictions to the original scale."""

        # Convert to an array.
        values = np.asarray(values, dtype=float)

        # Reverse log1p.
        if transform == "log1p":
            # Return the result.
            return np.clip(np.expm1(values), 0, None)

        # Keep the original values.
        return values.copy()

    @staticmethod
    def stack_unscaled_features(X_scaled, df, unscaled_cols):
        """Append unscaled columns to a transformed feature array."""

        if not unscaled_cols:
            return X_scaled

        X_unscaled = df[unscaled_cols].to_numpy(dtype=np.float32)
        return np.hstack([X_scaled, X_unscaled])

    @staticmethod
    def bounded_inverse_target_transform(
        y_pred_transformed,
        y_train_transformed,
        transform="log1p",
    ):
        """Clip transformed predictions to training bounds before inversion."""

        max_train_limit = y_train_transformed.max()
        y_pred_clipped = np.clip(
            y_pred_transformed,
            a_min=0.0,
            a_max=max_train_limit,
        )

        if transform == "log1p":
            return np.expm1(y_pred_clipped)

        raise ValueError(f"Unknown transform: {transform}")


    @staticmethod
    def bounded_inverse_poisson_log_rate(
        y_pred_log_rate,
        y_train_counts,
        safety_multiplier=3.0,
    ):
        """Convert bounded Poisson log-rate predictions back to count scale."""

        max_count = float(np.max(y_train_counts))
        max_log_rate = np.log(max(safety_multiplier * max_count, 1.0))
        y_pred_clipped = np.clip(
            y_pred_log_rate,
            a_min=np.log(1e-6),
            a_max=max_log_rate,
        )
        return np.exp(y_pred_clipped)

    # Define evaluate regression.
    @staticmethod
    def evaluate_regression(y_true, y_pred) -> dict[str, float]:
        """Calculate regression metrics."""

        # Prepare the true values.
        y_true = np.asarray(y_true, dtype=float).reshape(-1)

        # Prepare the predictions.
        y_pred = np.clip(
            np.asarray(y_pred, dtype=float).reshape(-1),
            0,
            None,
        )

        # Calculate MSE.
        mse = mean_squared_error(y_true, y_pred)

        # Return the result.
        return {
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "MSE": float(mse),
            "RMSE": float(np.sqrt(mse)),
            "R2": float(r2_score(y_true, y_pred)),
            "RMSLE": float(
                np.sqrt(
                    mean_squared_log_error(
                        np.clip(y_true, 0, None),
                        y_pred,
                    )
                )
            ),
        }

    # Define ranking at k by date.
    @staticmethod
    def ranking_at_k_by_date(
        df: pd.DataFrame,
        *,
        date_col: str,
        id_col: str,
        true_col: str,
        pred_col: str,
        k_values: Iterable[int] = (10, 20),
    ) -> dict[str, float]:
        """Calculate Recall@K and NDCG@K by date."""

        # Keep the ranking columns.
        ranking_df = df[
            [date_col, id_col, true_col, pred_col]
        ].copy()

        # Convert the values to numbers.
        ranking_df[true_col] = pd.to_numeric(
            ranking_df[true_col],
            errors="coerce",
        )
        # Set ranking df.
        ranking_df[pred_col] = pd.to_numeric(
            ranking_df[pred_col],
            errors="coerce",
        )

        # Remove missing rows.
        ranking_df = ranking_df.dropna(
            subset=[date_col, id_col, true_col, pred_col]
        )

        # Keep one station per date.
        ranking_df = (
            ranking_df.groupby([date_col, id_col], as_index=False)
            .agg(
                **{
                    true_col: (true_col, "sum"),
                    pred_col: (pred_col, "sum"),
                }
            )
        )

        # Store the metrics.
        results = {}

        # Calculate each K value.
        for k in k_values:
            # Set recalls.
            recalls = []
            # Set ndcgs.
            ndcgs = []

            # Rank stations for each date.
            for _, date_df in ranking_df.groupby(date_col, sort=False):
                # Set effective k.
                effective_k = min(int(k), len(date_df))

                # Set true sorted.
                true_sorted = date_df.sort_values(
                    true_col,
                    ascending=False,
                    kind="mergesort",
                )
                # Set pred sorted.
                pred_sorted = date_df.sort_values(
                    pred_col,
                    ascending=False,
                    kind="mergesort",
                )

                # Set true top ids.
                true_top_ids = set(
                    true_sorted.head(effective_k)[id_col]
                )
                # Set pred top ids.
                pred_top_ids = set(
                    pred_sorted.head(effective_k)[id_col]
                )

                # Add the result to the list.
                recalls.append(
                    len(true_top_ids.intersection(pred_top_ids))
                    / effective_k
                )

                # Set pred relevance.
                pred_relevance = np.clip(
                    pred_sorted.head(effective_k)[true_col].to_numpy(float),
                    0,
                    None,
                )
                # Set ideal relevance.
                ideal_relevance = np.clip(
                    true_sorted.head(effective_k)[true_col].to_numpy(float),
                    0,
                    None,
                )

                # Set discounts.
                discounts = np.log2(
                    np.arange(2, effective_k + 2)
                )
                # Set dcg.
                dcg = np.sum(pred_relevance / discounts)
                # Set idcg.
                idcg = np.sum(ideal_relevance / discounts)

                # Add the result to the list.
                ndcgs.append(
                    float(dcg / idcg) if idcg > 0 else 0.0
                )

            # Set results.
            results[f"Recall@{k}"] = float(np.mean(recalls))
            # Set results.
            results[f"NDCG@{k}"] = float(np.mean(ndcgs))

        # Return the result.
        return results

    # Define evaluate two targets.
    @staticmethod
    def evaluate_two_targets(
        metadata_df: pd.DataFrame,
        y_true,
        y_pred,
        *,
        date_col: str = "date",
        id_col: str = "station_name",
        target_labels: Sequence[str] = ("Inflow", "Outflow"),
        k_values: Sequence[int] = (10, 20),
    ) -> dict[str, float]:
        """Evaluate inflow and outflow."""

        # Prepare the true values.
        y_true = np.asarray(y_true, dtype=float)

        # Prepare the predictions.
        y_pred = np.clip(
            np.asarray(y_pred, dtype=float),
            0,
            None,
        )

        # Store each target result.
        per_target_results = []

        # Evaluate inflow and outflow.
        for target_index in range(2):
            # Set true values.
            true_values = y_true[:, target_index]
            # Set predicted values.
            predicted_values = y_pred[:, target_index]

            # Set regression metrics.
            regression_metrics = (
                BikeShareModelHelper.evaluate_regression(
                    y_true=true_values,
                    y_pred=predicted_values,
                )
            )

            # Set ranking df.
            ranking_df = metadata_df[
                [date_col, id_col]
            ].copy()
            # Set ranking df.
            ranking_df["target_value"] = true_values
            # Set ranking df.
            ranking_df["prediction"] = predicted_values

            # Set ranking metrics.
            ranking_metrics = (
                BikeShareModelHelper.ranking_at_k_by_date(
                    ranking_df,
                    date_col=date_col,
                    id_col=id_col,
                    true_col="target_value",
                    pred_col="prediction",
                    k_values=k_values,
                )
            )

            # Add the result to the list.
            per_target_results.append(
                {
                    **regression_metrics,
                    **ranking_metrics,
                }
            )

        # Store all metrics.
        results = {}

        # List the metric names.
        metric_names = REGRESSION_METRICS + tuple(
            metric
            for k in k_values
            for metric in (f"Recall@{k}", f"NDCG@{k}")
        )

        # Add inflow, outflow, and average values.
        for metric in metric_names:
            # Set target values.
            target_values = []

            # Loop through the values.
            for target_label, target_metrics in zip(
                target_labels,
                per_target_results,
            ):
                # Set metric value.
                metric_value = float(target_metrics[metric])
                # Set results.
                results[f"{target_label}_{metric}"] = metric_value
                # Add the result to the list.
                target_values.append(metric_value)

            # Set results.
            results[f"Avg_{metric}"] = float(
                np.mean(target_values)
            )

        # Return the result.
        return results

    # Define aggregate fold metrics.
    @staticmethod
    def aggregate_fold_metrics(
        fold_results_df: pd.DataFrame,
        *,
        metric_cols: Sequence[str],
    ) -> dict[str, float]:
        """Calculate fold means and standard deviations."""

        # Store the summary.
        summary = {}

        # Summarise each metric.
        for metric in metric_cols:
            # Set summary.
            summary[f"{metric}_mean"] = float(
                fold_results_df[metric].mean()
            )
            # Set summary.
            summary[f"{metric}_std"] = float(
                fold_results_df[metric].std()
            )

        # Return the result.
        return summary

    # Define set random seed.
    @staticmethod
    def set_random_seed(seed: int = 42) -> None:
        """Set random seeds."""

        # Set Python's seed.
        random.seed(seed)

        # Set NumPy's seed.
        np.random.seed(seed)

        # Set PyTorch's seed.
        torch.manual_seed(seed)
        # Run torch.cuda.manual_seed.
        torch.cuda.manual_seed(seed)
        # Run torch.cuda.manual_seed_all.
        torch.cuda.manual_seed_all(seed)

        # Make CUDA results reproducible.
        torch.backends.cudnn.deterministic = True
        # Set benchmark.
        torch.backends.cudnn.benchmark = False

    # Define build feature preprocessor.
    @staticmethod
    def build_feature_preprocessor(
        numerical_cols: Sequence[str],
        categorical_cols: Sequence[str],
    ) -> ColumnTransformer:
        """Scale numbers and encode categories."""

        # Return the result.
        return ColumnTransformer(
            transformers=[
                (
                    "numerical",
                    StandardScaler(),
                    list(numerical_cols),
                ),
                (
                    "categorical",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                    list(categorical_cols),
                ),
            ],
            remainder="drop",
        )



# Define get activation.
def get_activation(activation: str) -> nn.Module:
    """Return the selected activation."""

    # Set activations.
    activations = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "leaky_relu": nn.LeakyReLU(negative_slope=0.01),
    }

    # Return the result.
    return activations[activation]


# Define maybe layer norm.
def maybe_layer_norm(
    hidden_dim: int,
    use_layer_norm: bool,
) -> nn.Module:
    """Use LayerNorm when selected."""

    # Return the result.
    return (
        nn.LayerNorm(hidden_dim)
        if use_layer_norm
        else nn.Identity()
    )


# Define FeatureEncoder.
class FeatureEncoder(nn.Module):
    """Encode input features."""

    # Define   init  .
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        activation: str,
        use_layer_norm: bool,
        dropout: float,
    ) -> None:
        # Run the function.__init__.
        super().__init__()

        # Build the encoder.
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            get_activation(activation),
            maybe_layer_norm(hidden_dim, use_layer_norm),
            nn.Dropout(dropout),
        )

    # Define forward.
    def forward(self, x):
        """Run the encoder."""

        # Return the result.
        return self.layers(x)


# Define RegressionHead.
class RegressionHead(nn.Module):
    """Create regression outputs."""

    # Define   init  .
    def __init__(
        self,
        hidden_dim: int,
        output_dim: int = 2,
    ) -> None:
        # Run the function.__init__.
        super().__init__()

        # Store the output size.
        self.output_dim = output_dim

        # Build the output layer.
        self.output_layer = nn.Linear(
            hidden_dim,
            output_dim,
        )

    # Define forward.
    def forward(self, x):
        """Create predictions."""

        # Set output.
        output = self.output_layer(x)

        # Return the result.
        return (
            output.squeeze(-1)
            if self.output_dim == 1
            else output
        )


# List the public objects.
__all__ = [
    "BikeShareModelHelper",
    "FeatureEncoder",
    "RegressionHead",
    "get_activation",
    "maybe_layer_norm",
    "REGRESSION_METRICS",
    "RANKING_METRICS",
    "TWO_TARGET_METRICS",
]
