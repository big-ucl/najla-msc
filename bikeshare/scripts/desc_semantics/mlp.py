"""
Station-level MLP multi-target regression with semantic ablation variants.

"""

# =========================================================
# 1. Imports and project paths
# =========================================================

# Import garbage collection so finished models can be released.
import gc

# Import multiprocessing so fold-level work can run in separate processes.
import multiprocessing as mp

# Import a process pool for optional fold-level parallel execution.
from concurrent.futures import ProcessPoolExecutor

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
    / "desc_semantics"
    / "desc_df_2targets_txtTokens_graphFeat.parquet"
)

# Set the saved spatial fold file.
FOLD_ASSIGNMENTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_df"
    / "desc_semantics"
    / "desc_folds_of_2targets_txtTokens_graphFeat.parquet"
)

# Set the saved MiniLM embedding file.
EMBEDDING_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "graph"
    / "desc_semantic"
    / "desc_all_wikidata_minilm_embeddings.parquet"
)

# Set the description-semantic MLP results folder.
RESULTS_DIR = PROJECT_ROOT / "results" / "desc_semantics" / "mlp"

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
    wiki_embedding_mode="pca32",
    n_wiki_components=32,
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

    # Build the scaler and one-hot encoder without the wiki columns.
    preprocessor = mh.build_feature_preprocessor(
        numerical_cols=numerical_cols,
        categorical_cols=categorical_cols,
    )

    # Fit preprocessing on train rows and transform all splits.
    X_train_base = np.asarray(
        preprocessor.fit_transform(train_df),
        dtype=np.float32,
    )
    X_val_base = np.asarray(preprocessor.transform(val_df), dtype=np.float32)
    X_test_base = np.asarray(preprocessor.transform(test_df), dtype=np.float32)

    # Append raw or PCA wiki columns after scaled baseline features.
    X_train = mh.stack_unscaled_features(X_train_base, train_df, wiki_model_cols)
    X_val = mh.stack_unscaled_features(X_val_base, val_df, wiki_model_cols)
    X_test = mh.stack_unscaled_features(X_test_base, test_df, wiki_model_cols)

    # Store transformed feature names with appended unscaled wiki columns.
    feature_names = list(preprocessor.get_feature_names_out()) + wiki_model_cols

    # Keep the transformed semantic-column positions for branch models.
    semantic_feature_indices = np.asarray(
        [
            index
            for index, feature_name in enumerate(feature_names)
            if (
                feature_name.startswith("numerical__wiki_pca_")
                or feature_name.startswith("numerical__wiki_emb_")
                or feature_name.startswith("wiki_pca_")
                or feature_name.startswith("wiki_emb_")
            )
        ],
        dtype=np.int64,
    )
    semantic_feature_index_set = set(semantic_feature_indices.tolist())
    base_feature_indices = np.asarray(
        [
            index
            for index in range(len(feature_names))
            if index not in semantic_feature_index_set
        ],
        dtype=np.int64,
    )

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
        "semantic_feature_indices": semantic_feature_indices,
        "base_feature_indices": base_feature_indices,
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

    def __init__(
        self,
        input_dim,
        hidden_dims,
        activation,
        use_layer_norm,
        dropout,
        output_dim=2,
        *,
        architecture="concat",
        semantic_hidden_dim=0,
        semantic_dropout=None,
        semantic_use_layer_norm=None,
        semantic_feature_indices=None,
        base_feature_indices=None,
    ):
        super().__init__()

        semantic_feature_indices = np.asarray(
            [] if semantic_feature_indices is None else semantic_feature_indices,
            dtype=np.int64,
        )
        base_feature_indices = np.asarray(
            (
                np.arange(input_dim)
                if base_feature_indices is None
                else base_feature_indices
            ),
            dtype=np.int64,
        )

        self.input_dim = input_dim
        self.architecture = architecture
        semantic_dropout = dropout if semantic_dropout is None else semantic_dropout
        semantic_use_layer_norm = (
            use_layer_norm
            if semantic_use_layer_norm is None
            else semantic_use_layer_norm
        )
        self.uses_semantic_branch = (
            architecture == "semantic_branch"
            and len(semantic_feature_indices) > 0
            and semantic_hidden_dim > 0
        )

        self.register_buffer(
            "semantic_feature_indices",
            torch.as_tensor(semantic_feature_indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "base_feature_indices",
            torch.as_tensor(base_feature_indices, dtype=torch.long),
            persistent=False,
        )

        if self.uses_semantic_branch:
            base_input_dim = len(base_feature_indices)
            semantic_input_dim = len(semantic_feature_indices)

            self.encoder = FeatureEncoder(
                input_dim=base_input_dim,
                hidden_dim=hidden_dims[0],
                activation=activation,
                use_layer_norm=use_layer_norm,
                dropout=dropout,
            )
            self.backbone = self._build_backbone(
                hidden_dims=hidden_dims,
                activation=activation,
                use_layer_norm=use_layer_norm,
                dropout=dropout,
            )
            self.semantic_encoder = nn.Sequential(
                nn.Linear(semantic_input_dim, semantic_hidden_dim),
                get_activation(activation),
                maybe_layer_norm(semantic_hidden_dim, semantic_use_layer_norm),
                nn.Dropout(semantic_dropout),
            )
            self.semantic_gate = nn.Sequential(
                nn.Linear(semantic_hidden_dim, semantic_hidden_dim),
                nn.Sigmoid(),
            )
            self.head = RegressionHead(
                hidden_dim=hidden_dims[-1] + semantic_hidden_dim,
                output_dim=output_dim,
            )
        else:
            self.encoder = FeatureEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dims[0],
                activation=activation,
                use_layer_norm=use_layer_norm,
                dropout=dropout,
            )
            self.backbone = self._build_backbone(
                hidden_dims=hidden_dims,
                activation=activation,
                use_layer_norm=use_layer_norm,
                dropout=dropout,
            )
            self.head = RegressionHead(
                hidden_dim=hidden_dims[-1],
                output_dim=output_dim,
            )


    @staticmethod
    def _build_backbone(
        hidden_dims,
        activation,
        use_layer_norm,
        dropout,
    ):
        layers = []
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            layers.extend(
                [
                    nn.Linear(in_dim, out_dim),
                    get_activation(activation),
                    maybe_layer_norm(out_dim, use_layer_norm),
                    nn.Dropout(dropout),
                ]
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.uses_semantic_branch:
            base_x = x.index_select(1, self.base_feature_indices)
            semantic_x = x.index_select(1, self.semantic_feature_indices)
            base_x = self.backbone(self.encoder(base_x))
            semantic_x = self.semantic_encoder(semantic_x)
            semantic_x = semantic_x * self.semantic_gate(semantic_x)
            x = torch.cat([base_x, semantic_x], dim=1)
            return self.head(x)

        x = self.encoder(x)
        x = self.backbone(x)
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

    if getattr(model, "uses_semantic_branch", False):
        importance_raw = np.zeros(len(feature_names), dtype=float)
        base_layer = model.encoder.layers[0]
        semantic_layer = model.semantic_encoder[0]
        base_indices = model.base_feature_indices.detach().cpu().numpy()
        semantic_indices = model.semantic_feature_indices.detach().cpu().numpy()
        importance_raw[base_indices] = (
            base_layer.weight.detach().abs().mean(dim=0).cpu().numpy()
        )
        importance_raw[semantic_indices] = (
            semantic_layer.weight.detach().abs().mean(dim=0).cpu().numpy()
        )
    else:
        first_layer = model.encoder.layers[0]
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
def get_mlp_grid_exp(
    *,
    has_semantic_features=True,
):
    """Return the full grid with the smoke-test architecture assignment."""

    grid = {
        "hidden_dims": [
            (128, 64),
            (64, 32),
        ],
        "activation": ["relu"],
        "use_layer_norm": [True],
        "dropout": [
            0.10,
            0.20,
            0.30,
        ],
        "optimizer_name": ["AdamW"],
        "learning_rate": [
            0.001,
            0.0005,
        ],
        "weight_decay": [
            0.001,
            0.01,
        ],
        "loss_name": ["MSE"],
        "batch_size": [
            512,
            1024,
        ],
        "max_epochs": [100],
        "early_stopping_patience": [10],
        "scheduler_name": ["ReduceLROnPlateau"],
        "scheduler_patience": [4],
        "scheduler_factor": [0.50],
        "random_seed": [42],
    }

    if has_semantic_features:
        grid["architecture"] = ["semantic_branch"]
        grid["semantic_hidden_dim"] = [8]
    else:
        grid["architecture"] = ["concat"]
        grid["semantic_hidden_dim"] = [0]

    return grid


# Define get_mlp_smoke_test.
def get_mlp_smoke_test(*, has_semantic_features):
    """Return the one comparable smoke-test config for DF0, DF1, and DF2."""

    params = {
        "hidden_dims": (64, 32),
        "activation": "relu",
        "use_layer_norm": True,
        "dropout": 0.3,
        "optimizer_name": "AdamW",
        "learning_rate": 0.0001,
        "weight_decay": 0.1,
        "loss_name": "MSE",
        "batch_size": 1024,
        "max_epochs": 100,
        "early_stopping_patience": 10,
        "scheduler_name": "ReduceLROnPlateau",
        "scheduler_patience": 4,
        "scheduler_factor": 0.50,
        "random_seed": 42,
    }

    if has_semantic_features:
        params["architecture"] = "semantic_branch"
        params["semantic_hidden_dim"] = 8
    else:
        params["architecture"] = "concat"
        params["semantic_hidden_dim"] = 0

    return [params]


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
        architecture=params.get("architecture", "concat"),
        semantic_hidden_dim=params.get("semantic_hidden_dim", 0),
        semantic_dropout=params.get("semantic_dropout"),
        semantic_use_layer_norm=params.get("semantic_use_layer_norm"),
        semantic_feature_indices=fold["semantic_feature_indices"],
        base_feature_indices=fold["base_feature_indices"],
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

    # Convert validation predictions back with bounded log1p inversion.
    val_predictions = mh.bounded_inverse_target_transform(
        y_pred_transformed=val_pred_log,
        y_train_transformed=fold["y_train"],
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

    # Convert test predictions back with bounded log1p inversion.
    test_predictions = mh.bounded_inverse_target_transform(
        y_pred_transformed=test_pred_log,
        y_train_transformed=fold["y_train"],
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
# 8. Optional parallel fold workers
# =========================================================

_MLP_PARALLEL_CONTEXT = {}


def _init_mlp_parallel_worker(context):
    global _MLP_PARALLEL_CONTEXT
    _MLP_PARALLEL_CONTEXT = context
    torch.set_num_threads(context["torch_num_threads"])


def _prepare_parallel_mlp_fold(fold_definition, tensor_keys):
    context = _MLP_PARALLEL_CONTEXT
    device = torch.device(context["device_type"])
    fold = prepare_mlp_fold(
        df=context["df"],
        fold=fold_definition,
        numerical_cols=context["numerical_cols"],
        categorical_cols=context["categorical_cols"],
        wiki_cols=context["wiki_cols"],
        wiki_embedding_mode=context["wiki_embedding_mode"],
        n_wiki_components=context["n_wiki_components"],
    )
    for key in tensor_keys:
        fold[f"{key}_gpu"] = torch.from_numpy(fold[key]).to(device)
    return fold, device


def _get_fold_result_static(fold):
    return {
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


def _get_mlp_run_name(feature_set_code, trial_number):
    return (
        f"multiReg_mlp_12pm_5yrs_"
        f"{feature_set_code}_"
        f"trial_{trial_number:03d}"
    )


def _run_validation_fold_worker(task):
    context = _MLP_PARALLEL_CONTEXT
    fold_position = task["fold_position"]
    fold, device = _prepare_parallel_mlp_fold(
        fold_definition=task["fold_definition"],
        tensor_keys=(
            "X_train",
            "y_train",
            "X_val",
            "y_val",
        ),
    )
    fold_result_static = _get_fold_result_static(fold)
    fold_results = []
    epoch_rows = []
    importance_rows = []
    trial_logs = []

    for trial_number, params in enumerate(
        context["parameter_combinations"],
        start=1,
    ):
        run_name = _get_mlp_run_name(
            feature_set_code=context["feature_set_code"],
            trial_number=trial_number,
        )
        fold_output = train_one_mlp_fold(
            fold=fold,
            params=params,
            device=device,
        )
        fold_result = {
            "df_indx": context["df_indx"],
            "feature_set_name": context["feature_set_name"],
            "feature_set_code": context["feature_set_code"],
            "result_stage": "validation",
            "trial_number": trial_number,
            "run_name": run_name,
            "split_type": context["split_type"],
            "target_transform": context["target_transform"],
            "model_name": context["model_name"],
            **context["run_metadata"],
            **fold_result_static,
            "best_epoch": fold_output["best_epoch"],
            "best_val_loss": fold_output["best_val_loss"],
            **params,
            **fold_output["validation_metrics"],
        }
        fold_results.append(fold_result)
        trial_logs.append(
            f"TRIAL | profile={context['smoke_profile']} | "
            f"df_indx={context['df_indx']} | fold={fold['fold_id']:02d} | "
            f"trial={trial_number:03d}/{len(context['parameter_combinations'])} | "
            f"validation_MSE={fold_result['Avg_MSE']:.4f}"
        )
        epoch_rows.extend(
            [
                {
                    "df_indx": context["df_indx"],
                    "feature_set_name": context["feature_set_name"],
                    "feature_set_code": context["feature_set_code"],
                    "result_stage": "validation",
                    "trial_number": trial_number,
                    "run_name": run_name,
                    "fold_id": fold["fold_id"],
                    **context["run_metadata"],
                    **epoch_row,
                }
                for epoch_row in fold_output["epoch_history"]
            ]
        )
        importance_rows.extend(
            [
                {
                    "df_indx": context["df_indx"],
                    "feature_set_name": context["feature_set_name"],
                    "feature_set_code": context["feature_set_code"],
                    "result_stage": "validation",
                    "trial_number": trial_number,
                    "run_name": run_name,
                    "fold_id": fold["fold_id"],
                    **context["run_metadata"],
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
        del fold_output
        gc.collect()

    fold_id = fold["fold_id"]
    del fold, fold_result_static
    torch.cuda.empty_cache()
    gc.collect()
    return {
        "fold_position": fold_position,
        "fold_id": fold_id,
        "fold_results": fold_results,
        "epoch_rows": epoch_rows,
        "importance_rows": importance_rows,
        "trial_logs": trial_logs,
    }


def _run_test_fold_worker(task):
    context = _MLP_PARALLEL_CONTEXT
    fold_position = task["fold_position"]
    fold, device = _prepare_parallel_mlp_fold(
        fold_definition=task["fold_definition"],
        tensor_keys=(
            "X_train",
            "y_train",
            "X_val",
            "y_val",
            "X_test",
            "y_test",
        ),
    )
    fold_output = test_best_mlp_fold(
        fold=fold,
        params=context["best_params"],
        device=device,
    )
    fold_result = {
        "df_indx": context["df_indx"],
        "feature_set_name": context["feature_set_name"],
        "feature_set_code": context["feature_set_code"],
        "result_stage": "test_best",
        "trial_number": context["best_trial_number"],
        "run_name": context["best_run_name"],
        "split_type": context["split_type"],
        "target_transform": context["target_transform"],
        "model_name": context["model_name"],
        **context["run_metadata"],
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
        **context["best_params"],
        **fold_output["test_metrics"],
    }
    epoch_rows = [
        {
            "df_indx": context["df_indx"],
            "feature_set_name": context["feature_set_name"],
            "feature_set_code": context["feature_set_code"],
            "result_stage": "test_best",
            "trial_number": context["best_trial_number"],
            "run_name": context["best_run_name"],
            "fold_id": fold["fold_id"],
            **context["run_metadata"],
            **epoch_row,
        }
        for epoch_row in fold_output["epoch_history"]
    ]
    importance_rows = [
        {
            "df_indx": context["df_indx"],
            "feature_set_name": context["feature_set_name"],
            "feature_set_code": context["feature_set_code"],
            "result_stage": "test_best",
            "trial_number": context["best_trial_number"],
            "run_name": context["best_run_name"],
            "fold_id": fold["fold_id"],
            **context["run_metadata"],
            "feature": importance_row["feature"],
            "encoded_features_count": importance_row["encoded_features_count"],
            "importance_raw": importance_row["importance_raw"],
            "importance_pct": importance_row["importance_pct"],
        }
        for _, importance_row in fold_output[
            "feature_importance_df"
        ].iterrows()
    ]
    prediction_rows = fold_output["test_prediction_df"].to_dict("records")
    fold_id = fold["fold_id"]
    del fold, fold_output
    torch.cuda.empty_cache()
    gc.collect()
    return {
        "fold_position": fold_position,
        "fold_id": fold_id,
        "fold_result": fold_result,
        "epoch_rows": epoch_rows,
        "importance_rows": importance_rows,
        "prediction_rows": prediction_rows,
    }

# =========================================================
# 8. Main experiment
# =========================================================

# Define main.
def main():
    """Run the MLP semantic variants with validation-only model selection."""

    # Set the device and shared experiment seed.
    torch_num_threads = int(os.environ.get("MLP_TORCH_THREADS", "2"))
    torch.set_num_threads(torch_num_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parallel_jobs = int(os.environ.get("MLP_N_JOBS", "1"))
    if parallel_jobs < 1:
        raise ValueError("MLP_N_JOBS must be >= 1")
    parallel_start_method = os.environ.get("MLP_PARALLEL_START_METHOD", "spawn")
    mh.set_random_seed(42)
    print(f"DEVICE | {device}", flush=True)
    if parallel_jobs > 1:
        print(
            f"PARALLEL | jobs={parallel_jobs} | "
            f"start_method={parallel_start_method} | "
            f"torch_threads={torch_num_threads}",
            flush=True,
        )

    # Select the search size from the terminal.
    # MLP_SEARCH_NAME=smoke_test runs the one-trial smoke profile on one fold.
    # MLP_SEARCH_NAME=full_grid runs the complete grid on all folds.
    search_name = os.environ.get("MLP_SEARCH_NAME", "full_grid")
    smoke_profile = "semantic_branch"

    # Select how MiniLM wiki embeddings enter the model.
    # raw keeps the original 384 wiki_emb_* columns.
    # pca16, pca32, and pca64 fit PCA inside each fold.
    wiki_embedding_mode = os.environ.get("MLP_WIKI_MODE", "pca32")

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

    if wiki_embedding_mode.startswith("pca"):
        active_results_dir = RESULTS_DIR / f"pca{n_wiki_components}"
    else:
        active_results_dir = RESULTS_DIR / wiki_mode_label
    active_results_dir.mkdir(parents=True, exist_ok=True)

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

    existing_desc_cols = [
        col
        for col in base_df.columns
        if col.startswith("wiki_emb_")
    ]
    existing_desc_cols.extend(
        [
            col
            for col in [
                "attraction_count",
                "attraction_missing_flag",
            ]
            if col in base_df.columns
        ]
    )
    if existing_desc_cols:
        base_df = base_df.drop(columns=existing_desc_cols)

    # Identify every MiniLM embedding column before joining the tables.
    full_wiki_cols = [
        col
        for col in wiki_embedding_df.columns
        if col.startswith("wiki_emb_")
    ]

    desc_extra_cols = [
        "attraction_count",
        "attraction_missing_flag",
    ]
    missing_desc_extra_cols = [
        col
        for col in desc_extra_cols
        if col not in wiki_embedding_df.columns
    ]
    if missing_desc_extra_cols:
        raise ValueError(
            "Missing required desc baseline columns in embedding file: "
            f"{missing_desc_extra_cols}"
        )

    # Left join semantic embeddings using the location identifier.
    base_df["loc_id"] = base_df["loc_id"].astype(str)
    wiki_embedding_df["loc_id"] = wiki_embedding_df["loc_id"].astype(str)
    full_df = base_df.merge(
        wiki_embedding_df[["loc_id", *full_wiki_cols, *desc_extra_cols]],
        on="loc_id",
        how="left",
    )

    if "attraction_missing_flag" in full_df.columns:
        full_df[TEXT_HAS_WIKI_COLUMN] = (
            ~full_df["attraction_missing_flag"].astype(bool)
        ).astype(np.int8)

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
        elif df_indx == 2:
            df = full_df.drop(columns=[*full_wiki_cols, *node2vec_cols])
            wiki_cols = []
            feature_set_name = "df2_baseline_no_semantic_no_node2vec"
            feature_set_code = "DF2_BASELINE"

        else:
            raise ValueError(f"Unknown MLP_DF_INDX: {df_indx}")

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
            parameter_combinations = list(
                ParameterGrid(
                    get_mlp_grid_exp(
                        has_semantic_features=bool(wiki_cols),
                    )
                )
            )
            active_folds = spcv_folds
        elif search_name == "smoke_test":
            parameter_combinations = get_mlp_smoke_test(
                has_semantic_features=bool(wiki_cols),
            )
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
            "smoke_profile": smoke_profile,
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
        if search_name == "smoke_test":
            result_prefix = f"{result_prefix}_{search_name}_{smoke_profile}"
        grid_results_file = active_results_dir / f"{result_prefix}_grid.csv"
        fold_results_file = active_results_dir / f"{result_prefix}_fold.csv"
        feature_importance_file = active_results_dir / f"{result_prefix}_feature_importance.csv"

        # Keep the MLP-specific parameter lookup and epoch-curve files.
        trials_file = active_results_dir / f"{result_prefix}_trials.csv"
        epochs_file = active_results_dir / f"{result_prefix}_epochs.csv"
        predictions_file = active_results_dir / f"{result_prefix}_predictions.csv"

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
            f"VARIANT | profile={smoke_profile} | df_indx={df_indx} | "
            f"trials={len(parameter_combinations)} | folds={len(active_folds)}",
            flush=True,
        )

        # Store compact fold rows in memory.
        # Validation rows are added first and selected-winner test rows later.
        all_fold_results = []

        # =================================================
        # Validation grid: train and validate every trial
        # =================================================

        run_parallel_folds = parallel_jobs > 1 and len(active_folds) > 1

        if run_parallel_folds:
            parallel_context = {
                "df": df,
                "df_indx": df_indx,
                "feature_set_name": feature_set_name,
                "feature_set_code": feature_set_code,
                "numerical_cols": numerical_cols,
                "categorical_cols": categorical_cols,
                "wiki_cols": wiki_cols,
                "wiki_embedding_mode": wiki_embedding_mode,
                "n_wiki_components": n_wiki_components,
                "parameter_combinations": parameter_combinations,
                "run_metadata": run_metadata,
                "split_type": split_type,
                "target_transform": target_transform,
                "model_name": model_name,
                "smoke_profile": smoke_profile,
                "device_type": device.type,
                "torch_num_threads": torch_num_threads,
            }
            fold_tasks = [
                {
                    "fold_position": fold_position,
                    "fold_definition": fold_definition,
                }
                for fold_position, fold_definition in enumerate(
                    active_folds,
                    start=1,
                )
            ]
            process_workers = min(parallel_jobs, len(fold_tasks))
            process_context = mp.get_context(parallel_start_method)
            first_validation_write = True

            print(
                f"PARALLEL | stage=validation | "
                f"df_indx={df_indx} | workers={process_workers}",
                flush=True,
            )

            with ProcessPoolExecutor(
                max_workers=process_workers,
                mp_context=process_context,
                initializer=_init_mlp_parallel_worker,
                initargs=(parallel_context,),
            ) as executor:
                validation_outputs = executor.map(
                    _run_validation_fold_worker,
                    fold_tasks,
                    chunksize=1,
                )

                for validation_output in validation_outputs:
                    all_fold_results.extend(validation_output["fold_results"])
                    for trial_log in validation_output["trial_logs"]:
                        print(trial_log, flush=True)

                    epoch_rows_df = pd.DataFrame(
                        validation_output["epoch_rows"]
                    )
                    importance_rows_df = pd.DataFrame(
                        validation_output["importance_rows"]
                    )
                    write_mode = "w" if first_validation_write else "a"
                    epoch_rows_df.to_csv(
                        epochs_file,
                        mode=write_mode,
                        header=first_validation_write,
                        index=False,
                    )
                    importance_rows_df.to_csv(
                        feature_importance_file,
                        mode=write_mode,
                        header=first_validation_write,
                        index=False,
                    )
                    first_validation_write = False

                    pd.DataFrame(all_fold_results).to_csv(
                        fold_results_file,
                        index=False,
                    )
                    print(
                        f"FOLD | df_indx={df_indx} | "
                        f"fold={validation_output['fold_id']:02d} | "
                        f"{validation_output['fold_position']}/{len(active_folds)} complete",
                        flush=True,
                    )
                    del epoch_rows_df, importance_rows_df, validation_output
                    gc.collect()

            torch.cuda.empty_cache()
            gc.collect()

        else:
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

                    print(
                        f"TRIAL | profile={smoke_profile} | "
                        f"df_indx={df_indx} | fold={fold['fold_id']:02d} | "
                        f"trial={trial_number:03d}/{len(parameter_combinations)} | "
                        f"validation_MSE={fold_result['Avg_MSE']:.4f}",
                        flush=True,
                    )

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

        if run_parallel_folds:
            parallel_context = {
                "df": df,
                "df_indx": df_indx,
                "feature_set_name": feature_set_name,
                "feature_set_code": feature_set_code,
                "numerical_cols": numerical_cols,
                "categorical_cols": categorical_cols,
                "wiki_cols": wiki_cols,
                "wiki_embedding_mode": wiki_embedding_mode,
                "n_wiki_components": n_wiki_components,
                "run_metadata": run_metadata,
                "split_type": split_type,
                "target_transform": target_transform,
                "model_name": model_name,
                "device_type": device.type,
                "torch_num_threads": torch_num_threads,
                "best_params": best_params,
                "best_trial_number": best_trial_number,
                "best_run_name": best_run_name,
            }
            fold_tasks = [
                {
                    "fold_position": fold_position,
                    "fold_definition": fold_definition,
                }
                for fold_position, fold_definition in enumerate(
                    active_folds,
                    start=1,
                )
            ]
            process_workers = min(parallel_jobs, len(fold_tasks))
            process_context = mp.get_context(parallel_start_method)
            prediction_write_header = True

            print(
                f"PARALLEL | stage=test_best | "
                f"df_indx={df_indx} | workers={process_workers}",
                flush=True,
            )

            with ProcessPoolExecutor(
                max_workers=process_workers,
                mp_context=process_context,
                initializer=_init_mlp_parallel_worker,
                initargs=(parallel_context,),
            ) as executor:
                test_outputs = executor.map(
                    _run_test_fold_worker,
                    fold_tasks,
                    chunksize=1,
                )

                for test_output in test_outputs:
                    all_fold_results.append(test_output["fold_result"])
                    epoch_rows_df = pd.DataFrame(test_output["epoch_rows"])
                    importance_rows_df = pd.DataFrame(
                        test_output["importance_rows"]
                    )
                    prediction_rows_df = pd.DataFrame(
                        test_output["prediction_rows"]
                    )
                    epoch_rows_df.to_csv(
                        epochs_file,
                        mode="a",
                        header=False,
                        index=False,
                    )
                    importance_rows_df.to_csv(
                        feature_importance_file,
                        mode="a",
                        header=False,
                        index=False,
                    )
                    prediction_rows_df.to_csv(
                        predictions_file,
                        mode="w" if prediction_write_header else "a",
                        header=prediction_write_header,
                        index=False,
                    )
                    prediction_write_header = False
                    pd.DataFrame(all_fold_results).to_csv(
                        fold_results_file,
                        index=False,
                    )
                    del epoch_rows_df, importance_rows_df, prediction_rows_df
                    del test_output
                    gc.collect()

            torch.cuda.empty_cache()
            gc.collect()

        else:
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