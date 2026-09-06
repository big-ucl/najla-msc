"""
Station-level MLP multi-target regression with semantic ablation variants.

"""

# =========================================================
# 1. Imports and project paths
# =========================================================

# Import garbage collection so finished models can be released.
import gc

# Import os so the run mode can be selected from the terminal.
import os

# Import sys so the project root can be added to Python's module path.
import sys


# Import Path for Linux/WSL-safe file paths.
from pathlib import Path

# Import NumPy for arrays and numeric operations.
import numpy as np

# Import pandas for parquet reads and CSV writes.
import pandas as pd

# Import PyTorch for model training.
import torch

# Import torch.nn for layers and losses.
import torch.nn as nn


# Import PCA for reducing MiniLM embeddings inside each fold.
from sklearn.decomposition import PCA

# Import ParameterGrid for the MLP grid.
from sklearn.model_selection import ParameterGrid

# Set the project root.
PROJECT_ROOT = Path("/home/najla/dev/najla-msc/bikeshare")

# Add the project root to Python path.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Import shared station-level helper functions and model blocks.
from scripts.model_helper_stations_lvl import (
    BikeShareModelHelper as mh,
    FeatureEncoder,
    RegressionHead,
    get_activation,
    maybe_layer_norm,
)

# Print progress immediately in the terminal.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


# =========================================================
# 2. Input and output paths
# =========================================================

# Set the model dataframe file.
MODEL_DF_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_df"
    / "df_2targets_txtTokens_graphFeat.parquet"
)

# Set the saved spatial fold file.
FOLD_ASSIGNMENTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_df"
    / "folds_of_2targets_txtTokens_graphFeat.parquet"
)

# Set the saved MiniLM embedding file.
EMBEDDING_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "graph"
    / "cat_all_wikidata_minilm_embeddings.parquet"
)

# Set the semantic-enhancement MLP results folder.
RESULTS_DIR = PROJECT_ROOT / "results" / "cat_semantics" / "mlp"

# Create the results folder.
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Base names are built per dataframe variant inside main().

# Set the metric used to sort MLP configurations.
COMPARISON_METRIC = "Avg_MSE_mean"

# Mark stations whose Wikidata text is real text, not the placeholder.
TEXT_COLUMN = "wiki_items_text"
TEXT_HAS_WIKI_COLUMN = "_has_wiki_text"
PCA_LOC_ID_COLUMN = "_loc_id_for_pca"
NO_WIKI_TEXT_VALUES = {"", "no_wiki_data", "no_wikidata"}


# =========================================================
# 3. Shared columns
# =========================================================

# Set the target order: inflow first, outflow second.
target_cols = [
    "inflow_count",
    "outflow_count",
]

# Set the date column for ranking metrics.
date_col = "date"

# Set the station ID column for ranking metrics.
id_col = "station_name"

# Set the spatial group column for SPCV folds.
group_col = "spatial_group"

# Set categorical columns for one-hot encoding.
categorical_cols = [
    "day_type",
    "public_holiday",
    "Main_Weather_Category",
    "season",
]


# =========================================================
# 4. Fold preparation
# =========================================================

# Define prepare_mlp_fold.
def prepare_mlp_fold(
    df,
    fold,
    numerical_cols,
    categorical_cols,
    wiki_cols,
    *,
    wiki_embedding_mode="pca16",
    n_wiki_components=16,
):
    """Prepare one saved SPCV fold for multi-output MLP training."""

    # Select train, validation, and test rows.
    train_df = df.iloc[fold["train_idx"]].copy()
    val_df = df.iloc[fold["val_idx"]].copy()
    test_df = df.iloc[fold["test_idx"]].copy()

    # Use only MiniLM columns that still exist in this dataframe variant.
    wiki_cols = [col for col in wiki_cols if col in train_df.columns]

    # Raw mode keeps the original 384 MiniLM dimensions.
    # Stations with no Wikidata row have no embedding row, so zeros mean no semantic signal.
    if wiki_cols and wiki_embedding_mode == "raw":
        pca = None
        wiki_model_cols = wiki_cols
        train_df[wiki_model_cols] = train_df[wiki_model_cols].fillna(0)
        val_df[wiki_model_cols] = val_df[wiki_model_cols].fillna(0)
        test_df[wiki_model_cols] = test_df[wiki_model_cols].fillna(0)

    # PCA mode compresses MiniLM dimensions inside each fold.
    # Fit PCA only on training CTs that have real Wikidata text.
    # Then give every no-text node a zero PCA vector.
    elif wiki_cols:
        train_fit_df = (
            train_df.loc[
                train_df[TEXT_HAS_WIKI_COLUMN].astype(bool),
                [PCA_LOC_ID_COLUMN, *wiki_cols],
            ]
            .drop_duplicates(subset=PCA_LOC_ID_COLUMN)
        )

        # PCA cannot learn more components than real-text training CTs.
        n_real_wiki_locations = len(train_fit_df)
        if n_real_wiki_locations < n_wiki_components:
            raise ValueError(
                f"Fold has only {n_real_wiki_locations} real-Wikidata locations "
                f"but PCA requests {n_wiki_components} components."
            )

        pca = PCA(n_components=n_wiki_components, random_state=42)
        pca.fit(train_fit_df[wiki_cols].fillna(0))

        def transform_wiki_columns(source_df):
            pca_values = pca.transform(source_df[wiki_cols].fillna(0))
            no_text_mask = ~source_df[TEXT_HAS_WIKI_COLUMN].astype(bool).to_numpy()
            pca_values[no_text_mask] = 0
            return pca_values

        wiki_train = transform_wiki_columns(train_df)
        wiki_val = transform_wiki_columns(val_df)
        wiki_test = transform_wiki_columns(test_df)
        wiki_pca_cols = [f"wiki_pca_{i}" for i in range(n_wiki_components)]
        train_df[wiki_pca_cols] = wiki_train
        val_df[wiki_pca_cols] = wiki_val
        test_df[wiki_pca_cols] = wiki_test
        wiki_model_cols = wiki_pca_cols

    else:
        # No semantic embeddings in this variant, so there is no PCA block.
        pca = None
        wiki_model_cols = []

    # Add either raw wiki columns, PCA wiki columns, or no wiki columns.
    fold_numerical_cols = [*numerical_cols, *wiki_model_cols]

    # Build the scaler and one-hot encoder.
    preprocessor = mh.build_feature_preprocessor(
        numerical_cols=fold_numerical_cols,
        categorical_cols=categorical_cols,
    )

    # Fit preprocessing on train rows and transform all splits.
    X_train = np.asarray(preprocessor.fit_transform(train_df), dtype=np.float32)
    X_val = np.asarray(preprocessor.transform(val_df), dtype=np.float32)
    X_test = np.asarray(preprocessor.transform(test_df), dtype=np.float32)

    # Store transformed feature names for MLP feature-importance output.
    feature_names = list(preprocessor.get_feature_names_out())

    # Keep raw targets for metric calculation.
    y_train_raw = train_df[target_cols].to_numpy(dtype=float)
    y_val_raw = val_df[target_cols].to_numpy(dtype=float)
    y_test_raw = test_df[target_cols].to_numpy(dtype=float)

    # Transform targets for model training.
    y_train = mh.target_transform(y_train_raw, transform="log1p").astype(np.float32)
    y_val = mh.target_transform(y_val_raw, transform="log1p").astype(np.float32)
    y_test = mh.target_transform(y_test_raw, transform="log1p").astype(np.float32)

    # Keep validation and test metadata for ranking metrics.
    val_metadata = val_df[["loc_id", date_col, id_col, group_col, *target_cols]].copy().reset_index(drop=True)
    test_metadata = test_df[["loc_id", date_col, id_col, group_col, *target_cols]].copy().reset_index(drop=True)

    # Return all arrays, metadata, feature names, and fold labels.
    return {
        "fold_id": fold["fold_id"],
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "y_train_raw": y_train_raw,
        "y_val_raw": y_val_raw,
        "y_test_raw": y_test_raw,
        "input_dim": X_train.shape[1],
        "output_dim": len(target_cols),
        "feature_names": feature_names,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "val_metadata": val_metadata,
        "test_metadata": test_metadata,
        "preprocessor": preprocessor,
        "pca": pca,
        "train_groups": fold["train_groups"],
        "val_groups": fold["val_groups"],
        "test_groups": fold["test_groups"],
    }

# =========================================================
# 5. Model
# =========================================================

# Define MLPRegressor.
class MLPRegressor(nn.Module):
    """Jointly predict log-inflow and log-outflow."""

    # Define the model layers.
    def __init__(
        self,
        input_dim,
        hidden_dims,
        activation,
        use_layer_norm,
        dropout,
        output_dim=2,
    ):
        # Initialise nn.Module.
        super().__init__()

        # Create the first encoder block.
        # create the 1st hidden layer
        self.encoder = FeatureEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dims[0],
            activation=activation,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
        )

        # Store additional hidden layers.
        layers = []

        # Add one block for each remaining hidden layer.
        # build the remaining hidden layer as list of layer objects
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            layers.extend(
                [
                    nn.Linear(in_dim, out_dim),
                    get_activation(activation),
                    maybe_layer_norm(out_dim, use_layer_norm),
                    nn.Dropout(dropout),
                ]
            )

        # Create the hidden backbone.
        # turn the list into executable PyTorch block
        self.backbone = nn.Sequential(*layers)

        # Create the two-target regression head.
        self.head = RegressionHead(
            hidden_dim=hidden_dims[-1],
            output_dim=output_dim,
        )

    # Define forward, how the code navigate through the layers
    def forward(self, x):
        # Encode the feature vector.
        x = self.encoder(x)

        # Pass features through the backbone.
        x = self.backbone(x)

        # Return inflow and outflow predictions.
        return self.head(x)


# =========================================================
# 6. Feature importance helper
# =========================================================

def collapse_transformed_feature_name(
    transformed_name,
    categorical_cols,
):
    """Map transformed sklearn feature names back to model column names."""

    if transformed_name.startswith("numerical__"):
        raw_name = transformed_name.replace("numerical__", "", 1)
        if raw_name.startswith("wiki_pca_"):
            return "wiki_pca"
        return raw_name

    if transformed_name.startswith("categorical__"):
        raw_name = transformed_name.replace("categorical__", "", 1)
        for categorical_col in categorical_cols:
            if raw_name == categorical_col or raw_name.startswith(f"{categorical_col}_"):
                return categorical_col
        return raw_name

    return transformed_name


def get_mlp_feature_importance(
    model,
    feature_names,
    categorical_cols,
):
    """Create column-level MLP importance from first-layer absolute weights."""

    # The first Linear layer connects every input feature to the first hidden layer.
    first_layer = model.encoder.layers[0]

    # Average absolute weights over hidden units to get one score per transformed input.
    importance_raw = (
        first_layer.weight.detach().abs().mean(dim=0).cpu().numpy()
    )

    # Keep transformed names and collapsed original column names.
    encoded_df = pd.DataFrame(
        {
            "encoded_feature": feature_names,
            "feature": [
                collapse_transformed_feature_name(name, categorical_cols)
                for name in feature_names
            ],
            "importance_raw": importance_raw,
        }
    )

    # Average encoded columns so high-cardinality features are not inflated.
    feature_importance_df = (
        encoded_df.groupby("feature", as_index=False)
        .agg(
            encoded_features_count=("encoded_feature", "nunique"),
            importance_raw=("importance_raw", "mean"),
        )
    )

    # Convert the averaged feature scores to percentages.
    feature_importance_df["importance_pct"] = (
        100.0
        * feature_importance_df["importance_raw"]
        / feature_importance_df["importance_raw"].sum()
    )

    feature_importance_df = (
        feature_importance_df
        .sort_values("importance_pct", ascending=False)
        .reset_index(drop=True)
    )

    return feature_importance_df

# =========================================================
# 6. Hyperparameter grid
# =========================================================

# Define get_mlp_grid_exp.
def get_mlp_grid_exp():
    """Return the MLP hyperparameter grid from the notebook."""

    # Define AdamW configurations.
    adamw_grid = {
        "hidden_dims": [
            (128, 64),
            (128, 64, 32),
        ],
        "activation": [
            "relu",
            "gelu",
        ],
        "use_layer_norm": [True],
        "dropout": [
            0.10,
            0.30,
        ],
        "optimizer_name": ["AdamW"],
        "learning_rate": [
            0.003,
            0.001,
        ],
        "weight_decay": [
            0.001,
        ],
        "loss_name": ["MSE"],
        "batch_size": [4096],
        "max_epochs": [150],
        "early_stopping_patience": [10],
        "scheduler_name": [
            "ReduceLROnPlateau"
        ],
        "scheduler_patience": [4],
        "scheduler_factor": [0.50],
        "random_seed": [42],
    }

    # Return the AdamW grid.
    return adamw_grid


def get_mlp_smoke_test():
    """
    Return one shared best MLP configuration for a quick one-fold test.

    This uses one fixed selection for df0, df1, and df2. It is not a new grid
    search; it only checks the full training, validation, test, and CSV path.
    """

    return [
        {
            "hidden_dims": (128, 64),
            "activation": "relu",
            "use_layer_norm": True,
            "dropout": 0.10,
            "optimizer_name": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "loss_name": "MSE",
            "batch_size": 256,
            "max_epochs": 100,
            "early_stopping_patience": 10,
            "scheduler_name": "ReduceLROnPlateau",
            "scheduler_patience": 4,
            "scheduler_factor": 0.50,
            "random_seed": 42,
        }
    ]


# =========================================================
# 7. One-fold model fitting and evaluation
# =========================================================

# Define fit_one_mlp_fold.
def fit_one_mlp_fold(
    fold,
    params,
    *,
    device,
):
    """Fit one MLP using training rows and validation-only early stopping."""

    # Set the fold-specific random seed.
    mh.set_random_seed(params["random_seed"] + fold["fold_id"])

    # Reuse the fold tensors already stored on the GPU.
    X_train = fold["X_train_gpu"]
    y_train = fold["y_train_gpu"]
    n_train = X_train.shape[0]
    batch_size = params["batch_size"]

    # Create the MLP model with the selected grid parameters.
    model = MLPRegressor(
        input_dim=fold["input_dim"],
        hidden_dims=params["hidden_dims"],
        activation=params["activation"],
        use_layer_norm=params["use_layer_norm"],
        dropout=params["dropout"],
        output_dim=fold["output_dim"],
    ).to(device)

    # Create exactly the same MSE training loss.
    criterion = {
        "MSE": nn.MSELoss(),
    }[params["loss_name"]]

    # Create exactly the same AdamW optimizer.
    optimizer_class = {
        "AdamW": torch.optim.AdamW,
    }[params["optimizer_name"]]
    optimizer = optimizer_class(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )

    # Create exactly the same validation-controlled scheduler.
    scheduler_class = {
        "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    }[params["scheduler_name"]]
    scheduler = scheduler_class(
        optimizer,
        mode="min",
        factor=params["scheduler_factor"],
        patience=params["scheduler_patience"],
    )

    # Store the best validation state and the complete training curve.
    best_val_loss = float("inf")
    best_epoch = 0
    best_model_state = None
    epochs_without_improvement = 0
    epoch_history = []

    # Train until max_epochs or validation early stopping.
    for epoch in range(1, params["max_epochs"] + 1):
        model.train()
        train_loss_sum = torch.zeros((), device=device)
        permutation = torch.randperm(n_train).to(device)

        # Train with the same shuffled GPU index slices.
        for batch_start in range(0, n_train, batch_size):
            batch_indices = permutation[batch_start : batch_start + batch_size]
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]

            # Clear gradients, calculate predictions and MSE, then update weights.
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Accumulate loss on the GPU and synchronize once per epoch.
            train_loss_sum += loss.detach() * X_batch.shape[0]

        # Calculate mean training MSE on the log1p target scale.
        train_loss = (train_loss_sum / n_train).item()

        # Calculate validation MSE in one GPU pass.
        model.eval()
        with torch.no_grad():
            val_loss = criterion(
                model(fold["X_val_gpu"]),
                fold["y_val_gpu"],
            ).item()

        # Update the learning-rate scheduler using validation loss only.
        scheduler.step(val_loss)

        # Save the model state only when validation loss improves.
        validation_improved = val_loss < best_val_loss
        if validation_improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Save one row of the training and validation curve.
        epoch_history.append(
            {
                "epoch": epoch,
                "train_MSE_log1p": train_loss,
                "val_MSE_log1p": val_loss,
                "best_val_MSE_log1p": best_val_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "validation_improved": validation_improved,
                "epochs_without_improvement": epochs_without_improvement,
            }
        )

        # Stop after the same number of non-improving validation epochs.
        if epochs_without_improvement >= params["early_stopping_patience"]:
            break

    # Restore the model state from the best validation epoch.
    model.load_state_dict(best_model_state)
    model.eval()

    # Return the restored model and its validation-controlled training history.
    return {
        "model": model,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "epoch_history": epoch_history,
    }


# Define train_one_mlp_fold.
def train_one_mlp_fold(
    fold,
    params,
    *,
    device,
):
    """Train one grid-search model and evaluate validation rows only."""

    # Fit the model using training rows and validation-only early stopping.
    fit_output = fit_one_mlp_fold(
        fold=fold,
        params=params,
        device=device,
    )

    # Read the restored best-validation model.
    model = fit_output.pop("model")

    # Calculate feature importance from the restored validation model.
    feature_importance_df = get_mlp_feature_importance(
        model=model,
        feature_names=fold["feature_names"],
        categorical_cols=categorical_cols,
    )

    # Predict validation rows only during hyperparameter search.
    with torch.no_grad():
        val_pred_log = model(fold["X_val_gpu"]).cpu().numpy()

    # Convert validation predictions back to the original count scale.
    val_predictions = mh.inverse_target_transform(
        val_pred_log,
        transform="log1p",
    )

    # Calculate exactly the same validation metrics used for selection.
    validation_metrics = mh.evaluate_two_targets(
        metadata_df=fold["val_metadata"],
        y_true=fold["y_val_raw"],
        y_pred=val_predictions,
        date_col=date_col,
        id_col=id_col,
        target_labels=("Inflow", "Outflow"),
        k_values=(10, 20),
    )

    # Release this grid-search model before the next trial.
    del model
    gc.collect()

    # Return validation outputs only; no test prediction is made here.
    return {
        "best_epoch": fit_output["best_epoch"],
        "best_val_loss": fit_output["best_val_loss"],
        "epoch_history": fit_output["epoch_history"],
        "validation_metrics": validation_metrics,
        "feature_importance_df": feature_importance_df,
    }


# Define test_best_mlp_fold.
def test_best_mlp_fold(
    fold,
    params,
    *,
    device,
):
    """Retrain the validation-selected MLP and evaluate its test rows."""

    # Retrain the selected configuration from the beginning on this fold.
    # Training still uses only training rows, while validation rows still
    # control the scheduler, early stopping, and restored model state.
    fit_output = fit_one_mlp_fold(
        fold=fold,
        params=params,
        device=device,
    )

    # Read the newly retrained best-validation model.
    model = fit_output.pop("model")

    # Calculate feature importance from this final selected model.
    feature_importance_df = get_mlp_feature_importance(
        model=model,
        feature_names=fold["feature_names"],
        categorical_cols=categorical_cols,
    )

    # Predict test rows only after validation has selected the winner.
    with torch.no_grad():
        test_pred_log = model(fold["X_test_gpu"]).cpu().numpy()

    # Convert test predictions back to the original count scale.
    test_predictions = mh.inverse_target_transform(
        test_pred_log,
        transform="log1p",
    )

    # Store row-level test predictions for the selected winning model.
    test_prediction_df = fold["test_metadata"][[date_col, "loc_id", id_col, *target_cols]].copy()
    test_prediction_df = test_prediction_df.rename(
        columns={
            date_col: "date",
            id_col: "station_name",
            "inflow_count": "actual_inflow",
            "outflow_count": "actual_outflow",
        }
    )
    test_prediction_df["fold_id"] = fold["fold_id"]
    test_prediction_df["predicted_inflow"] = test_predictions[:, 0]
    test_prediction_df["predicted_outflow"] = test_predictions[:, 1]
    test_prediction_df = test_prediction_df[
        [
            "fold_id",
            "date",
            "loc_id",
            "station_name",
            "actual_inflow",
            "predicted_inflow",
            "actual_outflow",
            "predicted_outflow",
        ]
    ]

    # Calculate exactly the same two-target test metrics used previously.
    test_metrics = mh.evaluate_two_targets(
        metadata_df=fold["test_metadata"],
        y_true=fold["y_test_raw"],
        y_pred=test_predictions,
        date_col=date_col,
        id_col=id_col,
        target_labels=("Inflow", "Outflow"),
        k_values=(10, 20),
    )

    # Release this final selected model before the next test fold.
    del model
    gc.collect()

    # Return final test outputs and the winner-retraining records.
    return {
        "best_epoch": fit_output["best_epoch"],
        "best_val_loss": fit_output["best_val_loss"],
        "epoch_history": fit_output["epoch_history"],
        "test_metrics": test_metrics,
        "test_prediction_df": test_prediction_df,
        "feature_importance_df": feature_importance_df,
    }

# =========================================================
# 8. Main experiment
# =========================================================

# Define main.
def main():
    """Run the MLP semantic variants with validation-only model selection."""

    # Set the device and shared experiment seed.
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mh.set_random_seed(42)
    print(f"DEVICE | {device}", flush=True)

    # Select the search size from the terminal.
    # MLP_SEARCH_NAME=smoke_test runs one shared config on one fold.
    # MLP_SEARCH_NAME=full_grid runs the complete grid on all folds.
    search_name = os.environ.get("MLP_SEARCH_NAME", "full_grid")

    # Select how MiniLM wiki embeddings enter the model.
    # raw keeps the original 384 wiki_emb_* columns.
    # pca16, pca32, and pca64 fit PCA inside each fold.
    wiki_embedding_mode = os.environ.get("MLP_WIKI_MODE", "pca16")

    if wiki_embedding_mode == "raw":
        n_wiki_components = 384
        wiki_mode_label = "raw384"
    elif wiki_embedding_mode == "pca64":
        n_wiki_components = 64
        wiki_mode_label = "pca64_0no_text"
    elif wiki_embedding_mode == "pca32":
        n_wiki_components = 32
        wiki_mode_label = "pca32"
    elif wiki_embedding_mode == "pca16":
        n_wiki_components = 16
        wiki_mode_label = "pca16"
    else:
        raise ValueError(f"Unknown MLP_WIKI_MODE: {wiki_embedding_mode}")

    # Load the shared model dataframe, frozen folds, and MiniLM embeddings.
    base_df = pd.read_parquet(MODEL_DF_FILE)
    fold_assignments_df = pd.read_parquet(FOLD_ASSIGNMENTS_FILE)
    wiki_embedding_df = pd.read_parquet(EMBEDDING_FILE)

    # Keep a simple marker for stations with real Wikidata text.
    wiki_text = (
        base_df[TEXT_COLUMN]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    base_df[TEXT_HAS_WIKI_COLUMN] = (~wiki_text.isin(NO_WIKI_TEXT_VALUES)).astype(np.int8)
    base_df[PCA_LOC_ID_COLUMN] = base_df["loc_id"].astype(str)

    # Identify every MiniLM embedding column before joining the tables.
    full_wiki_cols = [
        col
        for col in wiki_embedding_df.columns
        if col.startswith("wiki_emb_")
    ]

    # Left join semantic embeddings using the location identifier.
    full_df = base_df.merge(
        wiki_embedding_df,
        on="loc_id",
        how="left",
    )

    # Remove the original Wikidata text after its embeddings are joined.
    full_df = full_df.drop(columns=["wiki_items_text"], errors="ignore")

    # Identify every Node2Vec column for the graph-feature ablations.
    node2vec_cols = [
        col
        for col in full_df.columns
        if col.startswith("node2vec_")
    ]

    # Define columns that are metadata, targets, or otherwise not model inputs.
    drop_feature_cols = [
        "loc_id",
        "loc_name",
        "loc_id_key",
        "lon",
        "lat",
        "spatial_group",
        "date",
        "inflow_count",
        "outflow_count",
        "end_station_name",
        "start_station_name",
        "station_name",
        "start_station_count",
        "end_station_count",
        "start_stations_count",
        "end_stations_count",
        "start_capacity_avg",
        "wiki_items_text",
        TEXT_HAS_WIKI_COLUMN,
        PCA_LOC_ID_COLUMN,
    ]

    # Define the unchanged validation metrics aggregated across spatial folds.
    metric_cols = [
        f"{prefix}_{metric}"
        for prefix in ("Inflow", "Outflow", "Avg")
        for metric in mh.METRIC_NAMES
    ]

    # Store identifiers shared by every result row.
    split_type = "SPCV"
    target_transform = "log1p"
    model_name = "MLPRegressorMultiOutput"

    # Run one arm when MLP_DF_INDX is set, otherwise run all three.
    df_indx_env = os.environ.get("MLP_DF_INDX")
    selected_df_indices = (
        [int(df_indx_env)]
        if df_indx_env is not None
        else [0, 1, 2]
    )

    for df_indx in selected_df_indices:

        # Variant 0 adds semantic MiniLM embeddings without Node2Vec features.
        if df_indx == 0:
            df = full_df.drop(columns=node2vec_cols)
            wiki_cols = full_wiki_cols
            feature_set_name = "df0_baseline_semantic_no_node2vec"
            feature_set_code = "DF0_BASELINE_TEXT_EMB"

        # Variant 1 adds Node2Vec graph features without semantic embeddings.
        elif df_indx == 1:
            df = full_df.drop(columns=full_wiki_cols)
            wiki_cols = []
            feature_set_name = "df1_baseline_node2vec_no_semantic"
            feature_set_code = "DF1_BASELINE_NODE2VEC"

        # Variant 2 is the baseline without semantic embeddings or Node2Vec features.
        else:
            df = full_df.drop(columns=[*full_wiki_cols, *node2vec_cols])
            wiki_cols = []
            feature_set_name = "df2_baseline_no_semantic_no_node2vec"
            feature_set_code = "DF2_BASELINE"

        # Keep numeric model columns after metadata, categorical, and raw
        # MiniLM columns have been assigned to their own processing blocks.
        numerical_cols = [
            col
            for col in df.columns
            if (
                col not in drop_feature_cols
                and col not in categorical_cols
                and col not in wiki_cols
            )
        ]

        # Rebuild the same saved SPCV row indices for this dataframe variant.
        spcv_folds = mh.prepare_spatial_folds_from_assignments(
            df=df,
            fold_assignments_df=fold_assignments_df,
            group_col=group_col,
            include_dataframes=False,
        )

        # Pick either the complete grid or the one-config smoke test.
        if search_name == "full_grid":
            parameter_combinations = list(ParameterGrid(get_mlp_grid_exp()))
            active_folds = spcv_folds
        elif search_name == "smoke_test":
            parameter_combinations = get_mlp_smoke_test()
            active_folds = spcv_folds[:1]
        else:
            raise ValueError(f"Unknown MLP_SEARCH_NAME: {search_name}")

        # Keep simple metadata that explains the active semantic feature shape.
        if wiki_cols and wiki_embedding_mode == "raw":
            wiki_model_dim = len(wiki_cols)
        elif wiki_cols:
            wiki_model_dim = n_wiki_components
        else:
            wiki_model_dim = 0

        run_metadata = {
            "search_name": search_name,
            "wiki_embedding_mode": wiki_embedding_mode,
            "wiki_input_dim": len(wiki_cols),
            "wiki_model_dim": wiki_model_dim,
        }

        # Use CatBoost-style filenames: full_grid is the default filename,
        # and smoke_test is only added when explicitly selected from terminal.
        result_prefix = (
            f"multiReg_mlp_12pm_5yrs_"
            f"{wiki_mode_label}_"
            f"{feature_set_name}"
        )
        if search_name != "full_grid":
            result_prefix = f"{result_prefix}_{search_name}"
        grid_results_file = RESULTS_DIR / f"{result_prefix}_grid.csv"
        fold_results_file = RESULTS_DIR / f"{result_prefix}_fold.csv"
        feature_importance_file = RESULTS_DIR / f"{result_prefix}_feature_importance.csv"

        # Keep the MLP-specific parameter lookup and epoch-curve files.
        trials_file = RESULTS_DIR / f"{result_prefix}_trials.csv"
        epochs_file = RESULTS_DIR / f"{result_prefix}_epochs.csv"
        predictions_file = RESULTS_DIR / f"{result_prefix}_predictions.csv"

        # Save the trial-number-to-parameters lookup before model fitting.
        trials_df = pd.DataFrame(
            [
                {
                    "df_indx": df_indx,
                    "feature_set_name": feature_set_name,
                    "feature_set_code": feature_set_code,
                    "trial_number": trial_number,
                    **run_metadata,
                    **params,
                }
                for trial_number, params in enumerate(
                    parameter_combinations,
                    start=1,
                )
            ]
        )
        trials_df.to_csv(trials_file, index=False)

        # Print the existing compact variant-level progress line.
        print(
            f"VARIANT | df_indx={df_indx} | "
            f"trials={len(parameter_combinations)} | folds={len(active_folds)}",
            flush=True,
        )

        # Store compact fold rows in memory.
        # Validation rows are added first and selected-winner test rows later.
        all_fold_results = []

        # =================================================
        # Validation grid: train and validate every trial
        # =================================================

        # Prepare one fold, run every trial on validation, save, then release it.
        for fold_position, fold_definition in enumerate(active_folds, start=1):

            # Fit fold-specific preprocessing using training rows only.
            fold = prepare_mlp_fold(
                df=df,
                fold=fold_definition,
                numerical_cols=numerical_cols,
                categorical_cols=categorical_cols,
                wiki_cols=wiki_cols,
                wiki_embedding_mode=wiki_embedding_mode,
                n_wiki_components=n_wiki_components,
            )

            # Move only training and validation tensors to the GPU during search.
            # Test tensors remain unused until the winning trial is known.
            for key in (
                "X_train",
                "y_train",
                "X_val",
                "y_val",
            ):
                fold[f"{key}_gpu"] = torch.from_numpy(fold[key]).to(device)

            # Calculate fold metadata once and reuse it for every trial.
            fold_result_static = {
                "fold_id": fold["fold_id"],
                "n_train_rows": len(fold["train_df"]),
                "n_val_rows": len(fold["val_df"]),
                "n_test_rows": len(fold["test_df"]),
                "n_train_cts": fold["train_df"]["loc_id"].nunique(),
                "n_val_cts": fold["val_df"]["loc_id"].nunique(),
                "n_test_cts": fold["test_df"]["loc_id"].nunique(),
                "train_groups": list(fold["train_groups"]),
                "val_groups": list(fold["val_groups"]),
                "test_groups": list(fold["test_groups"]),
            }

            # Train every hyperparameter configuration on this fold.
            for trial_number, params in enumerate(
                parameter_combinations,
                start=1,
            ):

                # Preserve the existing MLP run-name format.
                run_name = (
                    f"multiReg_mlp_12pm_5yrs_"
                    f"{feature_set_code}_"
                    f"trial_{trial_number:03d}"
                )

                # Train and evaluate validation only.
                fold_output = train_one_mlp_fold(
                    fold=fold,
                    params=params,
                    device=device,
                )

                # Store one CatBoost-style validation fold row.
                fold_result = {
                    "df_indx": df_indx,
                    "feature_set_name": feature_set_name,
                    "feature_set_code": feature_set_code,
                    "result_stage": "validation",
                    "trial_number": trial_number,
                    "run_name": run_name,
                    "split_type": split_type,
                    "target_transform": target_transform,
                    "model_name": model_name,
                    **run_metadata,
                    **fold_result_static,
                    "best_epoch": fold_output["best_epoch"],
                    "best_val_loss": fold_output["best_val_loss"],
                    **params,
                    **fold_output["validation_metrics"],
                }
                all_fold_results.append(fold_result)

                # Build this validation model's epoch-curve rows.
                epoch_rows_df = pd.DataFrame(
                    [
                        {
                            "df_indx": df_indx,
                            "feature_set_name": feature_set_name,
                            "feature_set_code": feature_set_code,
                            "result_stage": "validation",
                            "trial_number": trial_number,
                            "run_name": run_name,
                            "fold_id": fold["fold_id"],
                            **run_metadata,
                            **epoch_row,
                        }
                        for epoch_row in fold_output["epoch_history"]
                    ]
                )

                # Build this validation model's first-layer importance rows.
                importance_rows_df = pd.DataFrame(
                    [
                        {
                            "df_indx": df_indx,
                            "feature_set_name": feature_set_name,
                            "feature_set_code": feature_set_code,
                            "result_stage": "validation",
                            "trial_number": trial_number,
                            "run_name": run_name,
                            "fold_id": fold["fold_id"],
                            **run_metadata,
                            "feature": importance_row["feature"],
                            "encoded_features_count": importance_row["encoded_features_count"],
                            "importance_raw": importance_row["importance_raw"],
                            "importance_pct": importance_row["importance_pct"],
                        }
                        for _, importance_row in fold_output[
                            "feature_importance_df"
                        ].iterrows()
                    ]
                )

                # Start each streamed file on the first validation fit.
                write_header = fold_position == 1 and trial_number == 1
                write_mode = "w" if write_header else "a"

                # Write the training curve immediately instead of accumulating it.
                epoch_rows_df.to_csv(
                    epochs_file,
                    mode=write_mode,
                    header=write_header,
                    index=False,
                )

                # Write feature importance immediately instead of accumulating it.
                importance_rows_df.to_csv(
                    feature_importance_file,
                    mode=write_mode,
                    header=write_header,
                    index=False,
                )

                # Release trial-sized tables and model outputs immediately.
                del epoch_rows_df, importance_rows_df, fold_output
                gc.collect()

            # Checkpoint all completed validation rows after this fold.
            pd.DataFrame(all_fold_results).to_csv(
                fold_results_file,
                index=False,
            )

            # Print the existing compact fold-level progress line.
            print(
                f"FOLD | df_indx={df_indx} | "
                f"fold={fold['fold_id']:02d} | "
                f"{fold_position}/{len(active_folds)} complete",
                flush=True,
            )

            # Release this prepared fold before preparing the next one.
            del fold, fold_result_static
            torch.cuda.empty_cache()
            gc.collect()

        # =================================================
        # Validation aggregation and winner selection
        # =================================================

        # Convert the completed validation rows to a dataframe.
        fold_results_df = pd.DataFrame(all_fold_results)

        # Store one aggregate validation row per trial.
        all_grid_results = []

        # Aggregate every unchanged metric across the saved SPCV folds.
        for trial_number, params in enumerate(
            parameter_combinations,
            start=1,
        ):

            # Select only this trial's validation fold rows.
            trial_validation_df = fold_results_df[
                (
                    fold_results_df["trial_number"] == trial_number
                )
                & (
                    fold_results_df["result_stage"] == "validation"
                )
            ]

            # Preserve the existing MLP run-name format.
            run_name = (
                f"multiReg_mlp_12pm_5yrs_"
                f"{feature_set_code}_"
                f"trial_{trial_number:03d}"
            )

            # Add one aggregated validation result with unchanged metrics.
            all_grid_results.append(
                {
                    "df_indx": df_indx,
                    "feature_set_name": feature_set_name,
                    "feature_set_code": feature_set_code,
                    "trial_number": trial_number,
                    "run_name": run_name,
                    "split_type": split_type,
                    "target_transform": target_transform,
                    "model_name": model_name,
                    "n_spcv_folds": len(active_folds),
                    **run_metadata,
                    **params,
                    **mh.aggregate_fold_metrics(
                        trial_validation_df,
                        metric_cols=metric_cols,
                    ),
                    "best_epoch_mean": trial_validation_df["best_epoch"].mean(),
                    "best_epoch_std": trial_validation_df["best_epoch"].std(),
                    "best_val_loss_mean": trial_validation_df[
                        "best_val_loss"
                    ].mean(),
                    "best_val_loss_std": trial_validation_df[
                        "best_val_loss"
                    ].std(),
                }
            )

            # Keep the grid file sorted by the unchanged MLP comparison metric.
            grid_results_df = (
                pd.DataFrame(all_grid_results)
                .sort_values(COMPARISON_METRIC)
                .reset_index(drop=True)
            )

            # Checkpoint the aggregate grid after every completed trial row.
            grid_results_df.to_csv(
                grid_results_file,
                index=False,
            )

        # Select the configuration with the lowest validation Avg_MSE_mean.
        best_trial_number = int(
            grid_results_df.loc[
                0,
                "trial_number",
            ]
        )

        # Recover the selected parameter dictionary from the trial lookup.
        best_params = parameter_combinations[
            best_trial_number - 1
        ]

        # Preserve the selected run name in final test rows.
        best_run_name = grid_results_df.loc[
            0,
            "run_name",
        ]

        # Print the existing compact validation-selected winner line.
        print(
            f"BEST | df_indx={df_indx} | "
            f"trial={best_trial_number:03d} | "
            f"validation_MSE={grid_results_df.loc[0, COMPARISON_METRIC]:.4f}",
            flush=True,
        )

        # =================================================
        # Final test: retrain and test only the winner
        # =================================================

        # Prepare every fold again so the selected model is retrained cleanly.
        for final_fold_position, fold_definition in enumerate(active_folds, start=1):

            # Refit the same train-only preprocessing for this final fold model.
            fold = prepare_mlp_fold(
                df=df,
                fold=fold_definition,
                numerical_cols=numerical_cols,
                categorical_cols=categorical_cols,
                wiki_cols=wiki_cols,
                wiki_embedding_mode=wiki_embedding_mode,
                n_wiki_components=n_wiki_components,
            )

            # Move train, validation, and test tensors to the GPU.
            # Test tensors enter the model only in this winner-only stage.
            for key in (
                "X_train",
                "y_train",
                "X_val",
                "y_val",
                "X_test",
                "y_test",
            ):
                fold[f"{key}_gpu"] = torch.from_numpy(fold[key]).to(device)

            # Retrain the selected configuration and evaluate this test fold.
            fold_output = test_best_mlp_fold(
                fold=fold,
                params=best_params,
                device=device,
            )

            # Store one CatBoost-style selected-model test row.
            fold_result = {
                "df_indx": df_indx,
                "feature_set_name": feature_set_name,
                "feature_set_code": feature_set_code,
                "result_stage": "test_best",
                "trial_number": best_trial_number,
                "run_name": best_run_name,
                "split_type": split_type,
                "target_transform": target_transform,
                "model_name": model_name,
                **run_metadata,
                "fold_id": fold["fold_id"],
                "n_train_rows": len(fold["train_df"]),
                "n_val_rows": len(fold["val_df"]),
                "n_test_rows": len(fold["test_df"]),
                "n_train_cts": fold["train_df"]["loc_id"].nunique(),
                "n_val_cts": fold["val_df"]["loc_id"].nunique(),
                "n_test_cts": fold["test_df"]["loc_id"].nunique(),
                "train_groups": list(fold["train_groups"]),
                "val_groups": list(fold["val_groups"]),
                "test_groups": list(fold["test_groups"]),
                "best_epoch": fold_output["best_epoch"],
                "best_val_loss": fold_output["best_val_loss"],
                **best_params,
                **fold_output["test_metrics"],
            }
            all_fold_results.append(fold_result)

            # Build the selected model's retraining-curve rows.
            epoch_rows_df = pd.DataFrame(
                [
                    {
                        "df_indx": df_indx,
                        "feature_set_name": feature_set_name,
                        "feature_set_code": feature_set_code,
                        "result_stage": "test_best",
                        "trial_number": best_trial_number,
                        "run_name": best_run_name,
                        "fold_id": fold["fold_id"],
                        **run_metadata,
                        **epoch_row,
                    }
                    for epoch_row in fold_output["epoch_history"]
                ]
            )

            # Build the selected model's final feature-importance rows.
            importance_rows_df = pd.DataFrame(
                [
                    {
                        "df_indx": df_indx,
                        "feature_set_name": feature_set_name,
                        "feature_set_code": feature_set_code,
                        "result_stage": "test_best",
                        "trial_number": best_trial_number,
                        "run_name": best_run_name,
                        "fold_id": fold["fold_id"],
                        **run_metadata,
                        "feature": importance_row["feature"],
                        "encoded_features_count": importance_row["encoded_features_count"],
                        "importance_raw": importance_row["importance_raw"],
                        "importance_pct": importance_row["importance_pct"],
                    }
                    for _, importance_row in fold_output[
                        "feature_importance_df"
                    ].iterrows()
                ]
            )

            # Append winner-retraining epochs to the existing MLP curve file.
            epoch_rows_df.to_csv(
                epochs_file,
                mode="a",
                header=False,
                index=False,
            )

            # Append winner feature importance to the CatBoost-style file.
            importance_rows_df.to_csv(
                feature_importance_file,
                mode="a",
                header=False,
                index=False,
            )

            # Save row-level predictions only for the selected winning model.
            prediction_rows_df = fold_output["test_prediction_df"]
            prediction_write_header = final_fold_position == 1
            prediction_rows_df.to_csv(
                predictions_file,
                mode="w" if prediction_write_header else "a",
                header=prediction_write_header,
                index=False,
            )

            # Checkpoint validation rows plus completed winner test rows.
            pd.DataFrame(all_fold_results).to_csv(
                fold_results_file,
                index=False,
            )

            # Release this final fold before retraining the next one.
            del fold, fold_result
            del epoch_rows_df, importance_rows_df, prediction_rows_df, fold_output
            torch.cuda.empty_cache()
            gc.collect()

        # Release this variant before starting the next dataframe experiment.
        del trials_df, fold_results_df, grid_results_df
        del all_fold_results, all_grid_results
        gc.collect()

# =========================================================
# 9. Script entry point
# =========================================================

# Run main only when the file is executed directly.
if __name__ == "__main__":
    main()