

# =========================================================
# 1. Imports and project paths
# =========================================================

# Import garbage collection so completed model objects can be released.
import gc

# Import sys so the project root can be added to Python's module path.
import sys

# Import os so the feature set and search mode can be selected from the terminal.
import os

# Import Path for Linux/WSL-safe file paths.
from pathlib import Path

# Remove this script folder so catboost.py does not shadow CatBoost's package.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))

# Import NumPy for arrays, target transformation, indexing, and averages.
import numpy as np

# Import pandas for parquet input and CSV results.
import pandas as pd

# Import CatBoostRegressor for the two independent target models.
from catboost import CatBoostRegressor

# Import the CatBoost GPU-count utility for a clear startup check.
from catboost.utils import get_gpu_device_count

# Import PCA so MiniLM embeddings can be reduced inside each fold.
from sklearn.decomposition import PCA

# Import ParameterGrid for the complete exhaustive Experiment 2 search.
from sklearn.model_selection import ParameterGrid

# Import ParameterSampler for the reduced random search.
from sklearn.model_selection import ParameterSampler

# Define the root of the bikeshare project inside WSL.
PROJECT_ROOT = Path(
    "/home/najla/dev/najla-msc/bikeshare"
)

# Add the project root to Python's import path when it is not already present.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(
        str(PROJECT_ROOT)
    )

# Import the station-level evaluation helper used by the station-level notebook.
from scripts.model_helper_stations_lvl import BikeShareModelHelper as mh

# Make standard output line-buffered so every progress line appears immediately.
if hasattr(
    sys.stdout,
    "reconfigure",
):
    sys.stdout.reconfigure(
        line_buffering=True
    )


# =========================================================
# 2. Input and output locations
# =========================================================

# Define the final model dataframe saved by the notebook.
MODEL_DF_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_df"
    / "df_2targets_txtTokens_graphFeat.parquet"
)

# Define the frozen spatial fold assignments saved by the notebook.
FOLD_ASSIGNMENTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_df"
    / "folds_of_2targets_txtTokens_graphFeat.parquet"
)

# Define the saved MiniLM embedding file used by the MLP script.
EMBEDDING_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "graph"
    / "cat_all_wikidata_minilm_embeddings.parquet"
)

# Name the raw wiki text column so it can be dropped consistently.
TEXT_COLUMN = "wiki_items_text"
TEXT_HAS_WIKI_COLUMN = "_has_wiki_text"
PCA_LOC_ID_COLUMN = "_loc_id_for_pca"
NO_WIKI_TEXT_VALUES = {"", "no_wiki_data", "no_wikidata"}

# Default PCA dimensionality when CatBoost uses compressed MiniLM features.
WIKI_PCA_COMPONENTS = 32

# Define the CatBoost semantic-enhancement result directory used by this script.
RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "cat_semantics"
    / "catboost"
)

# Create the result directory if it does not already exist.
RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# Preserve the notebook's complete-grid result filename.
GRID_RESULTS_FILE = (
    RESULTS_DIR
    / "catboost_12pm_5yrs_grid.csv"
)

# Preserve the notebook's complete-grid fold result filename.
FOLD_RESULTS_FILE = (
    RESULTS_DIR
    / "catboost_12pm_5yrs_fold.csv"
)

METRIC_NAMES = mh.METRIC_NAMES


def get_wiki_embedding_config():
    """Resolve whether CatBoost uses raw MiniLM columns or fold-level PCA."""

    wiki_mode = os.environ.get(
        "CATBOOST_WIKI_MODE",
        f"pca{WIKI_PCA_COMPONENTS}",
    ).strip().lower()

    if wiki_mode == "raw":
        return "raw", None, "raw384"

    if wiki_mode == "pca":
        n_wiki_components = int(
            os.environ.get(
                "CATBOOST_WIKI_PCA_COMPONENTS",
                str(WIKI_PCA_COMPONENTS),
            )
        )
    elif wiki_mode.startswith("pca"):
        n_wiki_components = int(
            wiki_mode.removeprefix("pca")
        )
    else:
        n_wiki_components = WIKI_PCA_COMPONENTS

    return "pca", n_wiki_components, f"pca{n_wiki_components}_0no_text"


# =========================================================
# 3. Inflow, outflow, and average evaluation
# =========================================================

def evaluate_two_targets(
    metadata_df,
    y_true,
    y_pred,
):
    """Evaluate both targets with the shared station-level helper."""

    return mh.evaluate_two_targets(
        metadata_df=metadata_df,
        y_true=y_true,
        y_pred=y_pred,
        date_col="date",
        id_col="station_key",
        target_labels=("Inflow", "Outflow"),
        k_values=(10, 20),
    )

# =========================================================
# 4. Three hyperparameter-selection options
# =========================================================

def smoke_test():
    """
    Return one small GPU configuration for a quick end-to-end pipeline test.

    The smoke test intentionally uses only 300 maximum iterations and is run
    on only the first spatial fold. It verifies data loading, GPU training,
    prediction, evaluation, terminal printing, and CSV writing. Its metrics
    are not the final dissertation results.
    """

    # Return a list because the model loop expects multiple configurations.
    return [
        {
            # Use CatBoost native multi-target regression for both targets.
            "loss_function": "MultiRMSE",

            # Use the same native multi-target metric for validation and early stopping.
            "eval_metric": "MultiRMSE",

            # Use the notebook's central learning-rate value.
            "learning_rate": 0.05,

            # Use fewer trees only for the technical smoke test.
            "iterations": 300,

            # MultiRMSE on GPU does not support Lossguide.
            # Use CatBoost's GPU-supported symmetric tree growth instead.
            "grow_policy": "SymmetricTree",

            # SymmetricTree controls tree complexity with depth rather than max_leaves.
            "depth": 6,

            # Preserve the notebook's central L2 regularisation value.
            "l2_leaf_reg": 5,

            # Preserve Bernoulli row sampling.
            "bootstrap_type": "Bernoulli",

            # Preserve the notebook's central subsample value.
            "subsample": 0.8,

            # Preserve the notebook's lower random-strength value.
            "random_strength": 1,

            # Select GPU training.
            "task_type": "GPU",

            # Select the first and only available GPU.
            "devices": "0",

            # Leave part of the 8 GB GPU memory unused for stability.
            "gpu_ram_part": 0.25,

            # Evaluate metrics every 20 boosting iterations on GPU.
            "metric_period": 20,

            # Use the faster GPU border-count option for the smoke test.
            "border_count": 128,

            # Preserve the notebook's one-hot threshold.
            "one_hot_max_size": 2,

            # Preserve the notebook's early-stopping patience.
            "early_stopping_rounds": 50,

            # Keep the best validation iteration.
            "use_best_model": True,

            # Preserve the notebook's random seed.
            "random_seed": 42,
        }
    ]



def get_catboost_grid_exp2():
    """
    Return the complete Experiment 2 CatBoost hyperparameter grid.

    This preserves the notebook grid except for the required GPU changes:
        - rsm is removed for GPU regression compatibility.
        - task_type and devices force GPU 0.
        - gpu_ram_part leaves memory headroom.
        - metric_period reduces frequent GPU metric calculation.
    """

# Define the reduced grid based on the best previous trials.

    catboost_grid_stage2 = {
        # Use CatBoost native multi-target regression for both targets.
        "loss_function": ["MultiRMSE"],

        # Use validation MultiRMSE for early stopping and model selection.
        "eval_metric": ["MultiRMSE"],

        # Both previous best configurations used 0.1.
        "learning_rate": [0.1],

        # Allow early stopping within one large tree budget.
        "iterations": [3000],

        # Use CatBoost's GPU-supported symmetric tree growth.
        "grow_policy": ["SymmetricTree"],

        # Both previous best configurations used depth 4.
        "depth": [4],

        # Keep all three regularisation values because
        # the previous winners used 3 and 10.
        "l2_leaf_reg": [3, 5, 10],

        # Preserve Bernoulli bootstrap sampling.
        "bootstrap_type": ["Bernoulli"],

        # Keep all three because the previous winners
        # used both 0.7 and 1.0.
        "subsample": [0.7, 0.8, 1.0],

        # Keep both values to test random regularisation.
        "random_strength": [1, 5],

        # Force every complete-grid trial to use GPU.
        "task_type": ["GPU"],

        # Use GPU 0.
        "devices": ["0"],

        # Leave GPU-memory headroom.
        "gpu_ram_part": [0.25],

        # Calculate validation metrics every 20 iterations.
        "metric_period": [20],

        # Keep both because the previous best configurations
        # used different border counts.
        "border_count": [128, 255],

        # Preserve the categorical one-hot threshold.
        "one_hot_max_size": [2],

        # Stop when validation MultiRMSE does not improve for 50 checks.
        "early_stopping_rounds": [50],

        # Retain the best validation iteration.
        "use_best_model": [True],

        # Preserve reproducibility.
        "random_seed": [42],
    }

    # Return the complete grid.
    return catboost_grid_stage2



def parameter_sampler(
    n_iter=40,
    random_state=42,
):
    """
    Randomly sample a reduced number of configurations from the full grid.

    Parameters
    ----------
    n_iter:
        Number of unique hyperparameter configurations to sample.
    random_state:
        Seed controlling which configurations are selected.

    Returns
    -------
    sampled_parameters:
        List of scalar CatBoost parameter dictionaries.
    """

    # Create and materialise the sampled configurations as a list.
    sampled_parameters = list(
        ParameterSampler(
            get_catboost_grid_exp2(),
            n_iter=n_iter,
            random_state=random_state,
        )
    )

    # Return the reduced random search.
    return sampled_parameters


# =========================================================
# 5. Model feature and target preparation
# =========================================================

def load_model_dataframe():
    """Load the base dataframe and join MiniLM embeddings once."""

    print(
        f"Loading model dataframe: {MODEL_DF_FILE}",
        flush=True,
    )
    base_df = pd.read_parquet(
        MODEL_DF_FILE
    )
    print(
        f"Loaded dataframe shape: {base_df.shape}",
        flush=True,
    )

    if TEXT_COLUMN in base_df.columns:
        wiki_text = (
            base_df[TEXT_COLUMN]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )
        base_df[TEXT_HAS_WIKI_COLUMN] = (
            ~wiki_text.isin(NO_WIKI_TEXT_VALUES)
        ).astype(np.int8)
        base_df = base_df.drop(
            columns=[TEXT_COLUMN]
        )
    else:
        base_df[TEXT_HAS_WIKI_COLUMN] = 0

    base_df[PCA_LOC_ID_COLUMN] = base_df["loc_id"].astype(str)

    wiki_embedding_df = pd.read_parquet(
        EMBEDDING_FILE
    )
    full_wiki_cols = [
        col
        for col in wiki_embedding_df.columns
        if col.startswith("wiki_emb_")
    ]

    base_df["loc_id"] = base_df["loc_id"].astype(str)
    wiki_embedding_df["loc_id"] = wiki_embedding_df["loc_id"].astype(str)

    full_df = base_df.merge(
        wiki_embedding_df[["loc_id", *full_wiki_cols]],
        on="loc_id",
        how="left",
    )

    node2vec_cols = [
        col
        for col in full_df.columns
        if col.startswith("node2vec_")
    ]

    return full_df, full_wiki_cols, node2vec_cols

def prepare_model_data(
    df,
    wiki_cols,
):
    """Prepare targets, base features, categorical columns, and PCA source columns."""

    target_cols = [
        "inflow_count",
        "outflow_count",
    ]

    wiki_cols = [
        col
        for col in wiki_cols
        if col in df.columns
    ]

    drop_cols = [
        "loc_id",
        "loc_name",
        "loc_id_key",
        "lon",
        "lat",
        "spatial_group",
        "date",
        "outflow_count",
        "inflow_count",
        "end_station_name",
        "start_station_name",
        "station_name",
        "station_key",
        TEXT_COLUMN,
        "start_station_count",
        "end_station_count",
        "start_stations_count",
        "end_stations_count",
        "start_capacity_avg",
    ]

    drop_cols = [
        col
        for col in drop_cols
        if col in df.columns
    ]

    # Source features include raw wiki_emb_* only so fold-level PCA can read them.
    source_feature_cols = [
        col
        for col in df.columns
        if col not in drop_cols
    ]

    # Model features exclude raw wiki_emb_* and internal PCA metadata.
    feature_cols = [
        col
        for col in source_feature_cols
        if (
            col not in wiki_cols
            and col not in (TEXT_HAS_WIKI_COLUMN, PCA_LOC_ID_COLUMN)
        )
    ]

    cat_cols = [
        "day_type",
        "public_holiday",
        "Main_Weather_Category",
        "season",
    ]
    cat_cols = [
        col
        for col in cat_cols
        if col in feature_cols
    ]

    # Raw text is removed; semantic text enters only as numeric PCA columns.
    text_cols = []

    X = df[source_feature_cols]

    y_raw = df[
        target_cols
    ].to_numpy(
        dtype=float,
    )

    y_log = np.log1p(
        y_raw
    )

    return (
        target_cols,
        feature_cols,
        wiki_cols,
        cat_cols,
        text_cols,
        X,
        y_raw,
        y_log,
    )
# =========================================================
# 6. Frozen fold-index preparation
# =========================================================

def prepare_spatial_fold_indices(
    df,
    fold_assignments_df,
):
    """
    Convert the saved group-level fold-assignment table to dataframe row indices.

    The notebook already created and saved the spatial fold design. This
    function does not generate new folds. It only reads each saved fold and
    identifies the corresponding train, validation, and test rows in df.

    Storing integer row positions avoids creating ten complete copies of the
    model dataframe and therefore reduces system-memory usage.
    """

    # Create the output list that will hold all prepared fold dictionaries.
    prepared_folds = []

    # Read the saved fold identifiers in ascending order.
    fold_ids = sorted(
        fold_assignments_df["fold_id"].unique()
    )

    # Process one saved spatial fold at a time.
    for fold_id in fold_ids:

        # Select only the assignment rows belonging to the current fold.
        fold_table = fold_assignments_df.loc[
            fold_assignments_df["fold_id"] == fold_id
        ]

        # Read the spatial groups assigned to training in this fold.
        train_groups = tuple(
            sorted(
                fold_table.loc[
                    fold_table["split"] == "train",
                    "spatial_group",
                ].tolist()
            )
        )

        # Read the spatial groups assigned to validation in this fold.
        val_groups = tuple(
            sorted(
                fold_table.loc[
                    fold_table["split"] == "val",
                    "spatial_group",
                ].tolist()
            )
        )

        # Read the spatial groups assigned to testing in this fold.
        test_groups = tuple(
            sorted(
                fold_table.loc[
                    fold_table["split"] == "test",
                    "spatial_group",
                ].tolist()
            )
        )

        # Mark every dataframe row whose spatial group belongs to training.
        train_mask = df["spatial_group"].isin(
            train_groups
        )

        # Mark every dataframe row whose spatial group belongs to validation.
        val_mask = df["spatial_group"].isin(
            val_groups
        )

        # Mark every dataframe row whose spatial group belongs to testing.
        test_mask = df["spatial_group"].isin(
            test_groups
        )

        # Convert the Boolean training mask to integer row positions.
        train_idx = np.flatnonzero(
            train_mask.to_numpy()
        )

        # Convert the Boolean validation mask to integer row positions.
        val_idx = np.flatnonzero(
            val_mask.to_numpy()
        )

        # Convert the Boolean test mask to integer row positions.
        test_idx = np.flatnonzero(
            test_mask.to_numpy()
        )

        # Store this fold's row positions and original group assignments.
        prepared_folds.append(
            {
                "fold_id": int(fold_id),
                "train_idx": train_idx,
                "val_idx": val_idx,
                "test_idx": test_idx,
                "train_groups": train_groups,
                "val_groups": val_groups,
                "test_groups": test_groups,
            }
        )

    # Return every saved fold in the format expected by the model loops.
    return prepared_folds

def build_fold_feature_matrices(
    X,
    train_idx,
    eval_idx,
    predict_idx,
    feature_cols,
    wiki_cols,
    wiki_embedding_mode,
    n_wiki_components,
):
    """Build fold matrices using raw MiniLM columns or train-only PCA."""

    if not wiki_cols:
        X_train = X.iloc[
            train_idx
        ][feature_cols]
        X_eval = X.iloc[
            eval_idx
        ][feature_cols]
        X_predict = X.iloc[
            predict_idx
        ][feature_cols]
        return X_train, X_eval, X_predict

    train_df = X.iloc[
        train_idx
    ].copy()
    eval_df = X.iloc[
        eval_idx
    ].copy()
    predict_df = X.iloc[
        predict_idx
    ].copy()

    if wiki_embedding_mode == "raw":
        wiki_model_cols = wiki_cols
        fold_feature_cols = [
            *feature_cols,
            *wiki_model_cols,
        ]

        def attach_raw_columns(
            source_df,
        ):
            output_df = source_df[
                fold_feature_cols
            ].copy()
            output_df[wiki_model_cols] = output_df[
                wiki_model_cols
            ].fillna(
                0
            ).astype(
                np.float32
            )
            return output_df

        X_train = attach_raw_columns(
            train_df,
        )
        X_eval = attach_raw_columns(
            eval_df,
        )
        X_predict = attach_raw_columns(
            predict_df,
        )
        return X_train, X_eval, X_predict

    train_fit_df = (
        train_df.loc[
            train_df[TEXT_HAS_WIKI_COLUMN].astype(bool),
            [PCA_LOC_ID_COLUMN, *wiki_cols],
        ]
        .drop_duplicates(
            subset=PCA_LOC_ID_COLUMN
        )
    )
    n_real_wiki_locations = len(train_fit_df)
    if n_real_wiki_locations < n_wiki_components:
        raise ValueError(
            f"Fold has only {n_real_wiki_locations} real-Wikidata locations "
            f"but PCA requests {n_wiki_components} components."
        )

    pca = PCA(
        n_components=n_wiki_components,
        random_state=42,
    )
    pca.fit(
        train_fit_df[wiki_cols].fillna(0)
    )

    def transform_wiki_columns(
        source_df,
    ):
        pca_values = pca.transform(
            source_df[wiki_cols].fillna(0)
        )
        no_text_mask = ~source_df[TEXT_HAS_WIKI_COLUMN].astype(
            bool
        ).to_numpy()
        pca_values[no_text_mask] = 0
        return pca_values

    wiki_train = transform_wiki_columns(
        train_df
    )
    wiki_eval = transform_wiki_columns(
        eval_df
    )
    wiki_predict = transform_wiki_columns(
        predict_df
    )

    wiki_pca_cols = [
        f"wiki_pca_{i}"
        for i in range(n_wiki_components)
    ]
    fold_feature_cols = [
        *feature_cols,
        *wiki_pca_cols,
    ]

    def attach_pca_columns(
        source_df,
        pca_values,
    ):
        pca_df = pd.DataFrame(
            pca_values.astype(np.float32),
            columns=wiki_pca_cols,
            index=source_df.index,
        )
        return pd.concat(
            [
                source_df.drop(columns=wiki_cols),
                pca_df,
            ],
            axis=1,
        )[fold_feature_cols]

    X_train = attach_pca_columns(
        train_df,
        wiki_train,
    )
    X_eval = attach_pca_columns(
        eval_df,
        wiki_eval,
    )
    X_predict = attach_pca_columns(
        predict_df,
        wiki_predict,
    )

    return X_train, X_eval, X_predict
# =========================================================
# 7. One-model multi-target GPU fit and prediction
# =========================================================
def fit_and_predict_multi_target(
    params,
    X_train,
    y_train,
    X_eval,
    y_eval,
    X_predict,
    cat_cols,
    text_cols,
):
    """Fit one MultiRMSE CatBoost model and return predictions plus importances."""

    model = CatBoostRegressor(
        **params
    )

    model.fit(
        X_train,
        y_train,
        eval_set=(
            X_eval,
            y_eval,
        ),
        cat_features=cat_cols,
        text_features=text_cols,
        verbose=False,
    )

    evals_result = model.get_evals_result()

    prediction_log = model.predict(
        X_predict
    )

    best_iteration = model.get_best_iteration()

    feature_importance = np.asarray(
        model.get_feature_importance(),
        dtype=float,
    )

    if feature_importance.ndim > 1:
        feature_importance = feature_importance.mean(
            axis=0
        )

    feature_importance_df = pd.DataFrame(
        {
            "feature": list(X_train.columns),
            "importance": feature_importance,
        }
    )

    del model
    gc.collect()

    return prediction_log, best_iteration, feature_importance_df, evals_result


def build_iteration_history_rows(
    evals_result,
    params,
    df_indx,
    feature_set_name,
    feature_set_code,
    result_stage,
    trial_number,
    run_name,
    fold_id,
    best_iteration,
):
    """Build CatBoost boosting-iteration history rows."""

    learn_metric = evals_result["learn"]["MultiRMSE"]
    validation_metric = evals_result["validation"]["MultiRMSE"]

    metric_period = int(
        params.get(
            "metric_period",
            1,
        )
    )

    # With early stopping / overfitting detection active,
    # CatBoost evaluates the validation metric every iteration.
    validation_iterations = list(
        range(len(validation_metric))
    )

    # The final iteration actually executed by CatBoost.
    final_iteration = validation_iterations[-1]

    # Learn metrics are recorded according to metric_period.
    learn_iterations = list(
        range(
            0,
            final_iteration + 1,
            metric_period,
        )
    )

    # CatBoost also records the final learn metric when the final
    # iteration is not an exact metric_period point.
    if learn_iterations[-1] != final_iteration:
        learn_iterations.append(
            final_iteration
        )

    learn_by_iteration = dict(
        zip(
            learn_iterations,
            learn_metric,
        )
    )

    best_iteration_is_sampled = (
        best_iteration is not None
        and best_iteration >= 0
        and best_iteration < len(validation_metric)
    )

    iteration_rows = []
    best_validation_score = float("inf")
    best_recorded_iteration = 0

    for metric_index, validation_score in enumerate(
        validation_metric
    ):
        iteration = validation_iterations[
            metric_index
        ]

        learn_score = learn_by_iteration.get(
            iteration,
            np.nan,
        )

        validation_improved = (
            validation_score < best_validation_score
        )

        if validation_improved:
            best_validation_score = validation_score
            best_recorded_iteration = iteration

        iteration_rows.append(
            {
                "df_indx": df_indx,
                "feature_set_name": feature_set_name,
                "feature_set_code": feature_set_code,
                "result_stage": result_stage,
                "trial_number": trial_number,
                "run_name": run_name,
                "fold_id": fold_id,
                "metric_index": metric_index,
                "iteration": iteration,
                "learn_MultiRMSE": learn_score,
                "validation_MultiRMSE": validation_score,
                "best_validation_MultiRMSE": best_validation_score,
                "learning_rate": params["learning_rate"],
                "validation_improved": validation_improved,
                "iterations_without_improvement": (
                    iteration - best_recorded_iteration
                ),
                "best_iteration": best_iteration,
                "best_iteration_is_sampled": (
                    best_iteration_is_sampled
                ),
            }
        )

    return iteration_rows
# =========================================================
# 8. Main experiment
# =========================================================

def run_one_dataframe_variant(
    df,
    wiki_cols,
    df_indx,
    feature_set_name,
    feature_set_code,
    fold_assignments_df,
    wiki_embedding_mode,
    n_wiki_components,
    wiki_mode_label,
):
    """Run one CatBoost dataframe variant."""

    # Recreate the station-key alias required by the notebook ranking function.
    # The saved parquet contains station_name but does not contain station_key.
    df["station_key"] = df["station_name"]

    # Prepare the model features and both target matrices.
    (
        target_cols,
        feature_cols,
        wiki_cols,
        cat_cols,
        text_cols,
        X,
        y_raw,
        y_log,
    ) = prepare_model_data(
        df,
        wiki_cols,
    )

    # Convert the saved group-level fold table to low-memory row indices.
    prepared_folds = prepare_spatial_fold_indices(
        df,
        fold_assignments_df,
    )

    # Print the number of frozen folds prepared for model training.
    print(
        f"Prepared folds: {len(prepared_folds)}",
        flush=True,
    )

    # Build all result-column names produced by evaluate_two_targets.
    metric_columns = [
        f"{target_name}_{metric_name}"
        for target_name in [
            "Inflow",
            "Outflow",
            "Avg",
        ]
        for metric_name in METRIC_NAMES
    ]

    # =====================================================
    # ACTIVE SEARCH CHOICE
    # =====================================================

    # Select the search mode from the terminal.
    # CATBOOST_SEARCH_NAME=smoke_test runs one trial on one fold.
    # CATBOOST_SEARCH_NAME=full_grid runs get_catboost_grid_exp2() on all folds.
    # CATBOOST_SEARCH_NAME=parameter_sampler runs a reduced random search.
    search_name = os.environ.get(
        "CATBOOST_SEARCH_NAME",
        "full_grid",
    )

    if search_name == "full_grid":
        parameter_combinations = list(
            ParameterGrid(
                get_catboost_grid_exp2()
            )
        )
        active_folds = prepared_folds
    elif search_name == "smoke_test":
        parameter_combinations = smoke_test()
        active_folds = prepared_folds[:1]
    elif search_name == "parameter_sampler":
        parameter_combinations = parameter_sampler(
            n_iter=int(
                os.environ.get(
                    "CATBOOST_N_ITER",
                    "40",
                )
            ),
            random_state=42,
        )
        active_folds = prepared_folds
    else:
        raise ValueError(
            f"Unknown CATBOOST_SEARCH_NAME: {search_name}"
        )

    if wiki_cols and wiki_embedding_mode == "raw":
        wiki_model_dim = len(wiki_cols)
    elif wiki_cols:
        wiki_model_dim = n_wiki_components
    else:
        wiki_model_dim = 0

    run_metadata = {
        "wiki_embedding_mode": wiki_embedding_mode,
        "wiki_mode_label": wiki_mode_label,
        "wiki_input_dim": len(wiki_cols),
        "wiki_model_dim": wiki_model_dim,
    }

    # Use variant-specific filenames so dataframe experiments never overwrite each other.
    result_prefix = (
        f"multiReg_catboost_12pm_5yrs_{wiki_mode_label}_"
        f"{feature_set_name}"
    )
    if search_name == "full_grid":
        grid_results_file = (
            RESULTS_DIR
            / f"{result_prefix}_grid.csv"
        )
        fold_results_file = (
            RESULTS_DIR
            / f"{result_prefix}_fold.csv"
        )
        feature_importance_file = (
            RESULTS_DIR
            / f"{result_prefix}_feature_importance.csv"
        )
        trials_file = (
            RESULTS_DIR
            / f"{result_prefix}_trials.csv"
        )
        iterations_file = (
            RESULTS_DIR
            / f"{result_prefix}_iterations.csv"
        )
        predictions_file = (
            RESULTS_DIR
            / f"{result_prefix}_predictions.csv"
        )
    else:
        grid_results_file = (
            RESULTS_DIR
            / f"{result_prefix}_{search_name}_grid.csv"
        )
        fold_results_file = (
            RESULTS_DIR
            / f"{result_prefix}_{search_name}_fold.csv"
        )
        feature_importance_file = (
            RESULTS_DIR
            / f"{result_prefix}_{search_name}_feature_importance.csv"
        )
        trials_file = (
            RESULTS_DIR
            / f"{result_prefix}_{search_name}_trials.csv"
        )
        iterations_file = (
            RESULTS_DIR
            / f"{result_prefix}_{search_name}_iterations.csv"
        )
        predictions_file = (
            RESULTS_DIR
            / f"{result_prefix}_{search_name}_predictions.csv"
        )

    trials_df = pd.DataFrame(
        [
            {
                "df_indx": df_indx,
                "feature_set_name": feature_set_name,
                "feature_set_code": feature_set_code,
                "trial_number": trial_number,
                "run_name": (
                    f"multiReg_catboost_12pm_5yrs_"
                    f"{feature_set_code}_"
                    f"trial_{trial_number:03d}"
                ),
                **run_metadata,
                **params,
            }
            for trial_number, params in enumerate(
                parameter_combinations,
                start=1,
            )
        ]
    )
    trials_df.to_csv(
        trials_file,
        index=False,
    )

    # Print the selected search method.
    print(
        f"Search method: {search_name}",
        flush=True,
    )

    # Print the number of parameter configurations.
    print(
        f"Hyperparameter trials: {len(parameter_combinations)}",
        flush=True,
    )

    # Print the number of spatial folds used by the active method.
    print(
        f"Active spatial folds: {len(active_folds)}",
        flush=True,
    )

    # Calculate the number of validation fits.
    # MultiRMSE trains both targets in one model, so this is one fit per trial-fold pair.
    validation_fit_count = (
        len(parameter_combinations)
        * len(active_folds)
    )

    # Calculate the number of final test fits.
    # Again, one MultiRMSE model predicts both targets for each active fold.
    final_test_fit_count = len(active_folds)

    # Print the expected validation-fit count.
    print(
        f"Expected validation model fits: {validation_fit_count}",
        flush=True,
    )

    # Print the expected final test-fit count.
    print(
        f"Expected final test model fits: {final_test_fit_count}",
        flush=True,
    )

    # =====================================================
    # 9A. Hyperparameter tuning on validation folds
    # =====================================================

    # Create a list for one aggregate row per hyperparameter trial.
    grid_rows = []

    # Store every validation fold row and the final selected-model test rows.
    fold_rows = []

    # Store feature importance for every fitted model.
    feature_importance_rows = []

    # Store row-level predictions for the final selected model only.
    prediction_rows = []

    # Track whether the CatBoost boosting-iteration file needs its header.
    iteration_write_header = True

    # Loop over each scalar CatBoost parameter dictionary.
    for trial_number, params in enumerate(
        parameter_combinations,
        start=1,
    ):

        # Preserve the notebook's run-name format.
        run_name = (
            f"multiReg_catboost_12pm_5yrs_"
            f"{feature_set_code}_"
            f"trial_{trial_number:03d}"
        )

        # Create a list for this trial's validation-fold results.
        trial_fold_rows = []

        # Run the current configuration on every active spatial fold.
        for fold in active_folds:

            # Read this fold's identifier.
            fold_id = fold["fold_id"]

            # Read this fold's training row positions.
            train_idx = fold["train_idx"]

            # Read this fold's validation row positions.
            val_idx = fold["val_idx"]

            # Build fold feature matrices.
            # semantic_pca fits PCA on train rows only and transforms validation rows.
            X_train, X_val, _ = build_fold_feature_matrices(
                X=X,
                train_idx=train_idx,
                eval_idx=val_idx,
                predict_idx=val_idx,
                feature_cols=feature_cols,
                wiki_cols=wiki_cols,
                wiki_embedding_mode=wiki_embedding_mode,
                n_wiki_components=n_wiki_components,
            )

            # Slice the transformed two-target training matrix.
            y_train = y_log[
                train_idx
            ]

            # Slice the transformed two-target validation matrix.
            y_val = y_log[
                val_idx
            ]

            # Fit one native MultiRMSE model and predict validation inflow/outflow.
            # y_train and y_val are two-column matrices in the fixed target order:
            #     column 0 = inflow_count on the log1p scale
            #     column 1 = outflow_count on the log1p scale
            (
                val_predictions_log,
                best_iteration,
                feature_importance_df,
                evals_result,
            ) = fit_and_predict_multi_target(
                params=params,
                X_train=X_train,
                y_train=y_train,
                X_eval=X_val,
                y_eval=y_val,
                X_predict=X_val,
                cat_cols=cat_cols,
                text_cols=text_cols,
            )

            # Convert both predictions back to the original trip-count scale.
            val_predictions = np.clip(
                np.expm1(
                    val_predictions_log
                ),
                0,
                None,
            )

            # Evaluate inflow, outflow, and average validation metrics.
            metrics = evaluate_two_targets(
                metadata_df=df.iloc[
                    val_idx
                ],
                y_true=y_raw[
                    val_idx
                ],
                y_pred=val_predictions,
            )

            # Store this validation fold with identifiers, split sizes, parameters, and metrics.
            fold_result = {
                "df_indx": df_indx,
                "feature_set_name": feature_set_name,
                "feature_set_code": feature_set_code,
                "result_stage": "validation",
                "trial_number": trial_number,
                "run_name": run_name,
                "fold_id": fold_id,
                **run_metadata,
                "n_train_rows": len(train_idx),
                "n_val_rows": len(val_idx),
                "n_test_rows": len(fold["test_idx"]),
                "n_train_cts": df.iloc[train_idx]["loc_id"].nunique(),
                "n_val_cts": df.iloc[val_idx]["loc_id"].nunique(),
                "n_test_cts": df.iloc[fold["test_idx"]]["loc_id"].nunique(),
                "train_groups": list(fold["train_groups"]),
                "val_groups": list(fold["val_groups"]),
                "test_groups": list(fold["test_groups"]),
                "best_iteration": best_iteration,
                **params,
                **metrics,
            }
            trial_fold_rows.append(fold_result)
            fold_rows.append(fold_result)

            iteration_rows_df = pd.DataFrame(
                build_iteration_history_rows(
                    evals_result=evals_result,
                    params=params,
                    df_indx=df_indx,
                    feature_set_name=feature_set_name,
                    feature_set_code=feature_set_code,
                    result_stage="validation",
                    trial_number=trial_number,
                    run_name=run_name,
                    fold_id=fold_id,
                    best_iteration=best_iteration,
                )
            )
            iteration_rows_df.to_csv(
                iterations_file,
                mode="w" if iteration_write_header else "a",
                header=iteration_write_header,
                index=False,
            )
            iteration_write_header = False

            # Store feature importance for this validation model.
            for _, importance_row in feature_importance_df.iterrows():
                feature_importance_rows.append(
                    {
                        "df_indx": df_indx,
                        "feature_set_name": feature_set_name,
                        "feature_set_code": feature_set_code,
                        "result_stage": "validation",
                        "trial_number": trial_number,
                        "run_name": run_name,
                        "fold_id": fold_id,
                        "feature": importance_row["feature"],
                    **run_metadata,
                        "importance": importance_row["importance"],
                    }
                )

            # Convert the zero-based MultiRMSE iteration to a displayed tree count.
            epoch = (
                best_iteration + 1
                if best_iteration >= 0
                else params["iterations"]
            )

            # Print the same compact validation output used in the notebook.
            # The single epoch value belongs to the one multi-target model.
            print(
                f"VALID | "
                f"trial={trial_number:03d} | "
                f"fold={fold_id:02d} | "
                f"epoch={epoch} | "
                f"batch=N/A | "
                f"MSE={metrics['Avg_MSE']:.4f} | "
                f"RMSE={metrics['Avg_RMSE']:.4f} | "
                f"Recall@20={metrics['Avg_Recall@20']:.4f}",
                flush=True,
            )

            # Delete fold-specific feature matrices.
            del X_train
            del X_val

            # Delete fold-specific target matrices.
            del y_train
            del y_val

            # Delete prediction arrays and feature-importance dataframe.
            del val_predictions_log
            del val_predictions
            del evals_result
            del feature_importance_df
            del iteration_rows_df

            # Ask Python to release unreachable memory before the next fit.
            gc.collect()

        # Convert all validation-fold rows for this trial to a dataframe.
        trial_fold_df = pd.DataFrame(
            trial_fold_rows
        )

        # Start the aggregate trial result with identifiers and parameters.
        trial_result = {
            "df_indx": df_indx,
            "feature_set_name": feature_set_name,
            "feature_set_code": feature_set_code,
            "trial_number": trial_number,
            "run_name": run_name,
            **run_metadata,
            **params,
        }

        # Calculate fold mean and standard deviation for every metric column.
        for metric in metric_columns:

            # Store the validation-fold mean.
            trial_result[
                f"{metric}_mean"
            ] = trial_fold_df[
                metric
            ].mean()

            # Store the validation-fold standard deviation.
            trial_result[
                f"{metric}_std"
            ] = trial_fold_df[
                metric
            ].std()

        # Store the mean best iteration across validation folds.
        trial_result[
            "best_iteration_mean"
        ] = trial_fold_df[
            "best_iteration"
        ].mean()

        # Store the standard deviation of best iterations across folds.
        trial_result[
            "best_iteration_std"
        ] = trial_fold_df[
            "best_iteration"
        ].std()

        # Add the completed aggregate trial result to the grid list.
        grid_rows.append(
            trial_result
        )

        # Print completion of this trial immediately.
        print(
            f"Completed trial {trial_number:03d}/{len(parameter_combinations):03d}",
            flush=True,
        )

    # =====================================================
    # 9B. Select the lowest average validation RMSE
    # =====================================================

    # Convert all aggregate trial results to a dataframe.
    grid_results_df = pd.DataFrame(
        grid_rows
    )

    # Sort so the lowest macro-average validation RMSE is first.
    grid_results_df = grid_results_df.sort_values(
        by="Avg_RMSE_mean",
        ascending=True,
    ).reset_index(
        drop=True
    )

    # Read the winning one-based trial number.
    best_trial_number = int(
        grid_results_df.loc[
            0,
            "trial_number",
        ]
    )

    # Recover the winning scalar parameter dictionary.
    best_params = parameter_combinations[
        best_trial_number - 1
    ]

    # Read the winning run name.
    best_run_name = grid_results_df.loc[
        0,
        "run_name",
    ]

    # =====================================================
    # 9C. Final evaluation on test folds
    # =====================================================

    # Append final selected-model test rows to the same fold_rows list.
    # fold_rows already contains every validation trial/fold row.

    # Evaluate only the selected parameter configuration.
    for fold in active_folds:

        # Read this fold's identifier.
        fold_id = fold["fold_id"]

        # Read this fold's training row positions.
        train_idx = fold["train_idx"]

        # Read this fold's validation row positions.
        val_idx = fold["val_idx"]

        # Read this fold's test row positions.
        test_idx = fold["test_idx"]

        # Build fold feature matrices.
        # semantic_pca fits PCA on train rows only and transforms validation/test rows.
        X_train, X_val, X_test = build_fold_feature_matrices(
            X=X,
            train_idx=train_idx,
            eval_idx=val_idx,
            predict_idx=test_idx,
            feature_cols=feature_cols,
            wiki_cols=wiki_cols,
            wiki_embedding_mode=wiki_embedding_mode,
            n_wiki_components=n_wiki_components,
        )

        # Slice transformed training targets.
        y_train = y_log[
            train_idx
        ]

        # Slice transformed validation targets.
        y_val = y_log[
            val_idx
        ]

        # Fit one final native MultiRMSE model and predict test inflow/outflow.
        # The same model outputs both target columns simultaneously.
        (
            test_predictions_log,
            best_iteration,
            feature_importance_df,
            evals_result,
        ) = fit_and_predict_multi_target(
            params=best_params,
            X_train=X_train,
            y_train=y_train,
            X_eval=X_val,
            y_eval=y_val,
            X_predict=X_test,
            cat_cols=cat_cols,
            text_cols=text_cols,
        )

        # Convert predictions back to original daily trip counts.
        test_predictions = np.clip(
            np.expm1(
                test_predictions_log
            ),
            0,
            None,
        )

        # Calculate inflow, outflow, and average test metrics.
        metrics = evaluate_two_targets(
            metadata_df=df.iloc[
                test_idx
            ],
            y_true=y_raw[
                test_idx
            ],
            y_pred=test_predictions,
        )

        # Store inverse-log predictions from the selected test model.
        prediction_df = pd.DataFrame(
            {
                "fold_id": fold_id,
                "date": df.iloc[test_idx]["date"].to_numpy(),
                "loc_id": df.iloc[test_idx]["loc_id"].to_numpy(),
                "station_name": df.iloc[test_idx]["station_name"].to_numpy(),
                "actual_inflow": y_raw[test_idx, 0],
                "predicted_inflow": test_predictions[:, 0],
                "actual_outflow": y_raw[test_idx, 1],
                "predicted_outflow": test_predictions[:, 1],
            }
        )
        prediction_rows.extend(
            prediction_df.to_dict(
                "records"
            )
        )

        # The one MultiRMSE model has one best iteration shared by both targets.
        # Convert the zero-based MultiRMSE iteration to a displayed tree count.
        epoch = (
            best_iteration + 1
            if best_iteration >= 0
            else best_params["iterations"]
        )

        # Print the same compact test output used in the notebook.
        # The single epoch value belongs to the one multi-target model.
        print(
            f"TEST  | "
            f"fold={fold_id:02d} | "
            f"epoch={epoch} | "
            f"batch=N/A | "
            f"MSE={metrics['Avg_MSE']:.4f} | "
            f"RMSE={metrics['Avg_RMSE']:.4f} | "
            f"Recall@20={metrics['Avg_Recall@20']:.4f}",
            flush=True,
        )

        # Store one final result row for this test fold.
        fold_rows.append(
            {
                "df_indx": df_indx,
                "feature_set_name": feature_set_name,
                "feature_set_code": feature_set_code,
                "result_stage": "test_best",
                "trial_number": best_trial_number,
                "run_name": best_run_name,
                **run_metadata,
                "fold_id": fold_id,
                "n_train_rows": len(train_idx),
                "n_val_rows": len(val_idx),
                "n_test_rows": len(test_idx),
                "n_train_cts": df.iloc[
                    train_idx
                ]["loc_id"].nunique(),
                "n_val_cts": df.iloc[
                    val_idx
                ]["loc_id"].nunique(),
                "n_test_cts": df.iloc[
                    test_idx
                ]["loc_id"].nunique(),
                "train_groups": list(
                    fold["train_groups"]
                ),
                "val_groups": list(
                    fold["val_groups"]
                ),
                "test_groups": list(
                    fold["test_groups"]
                ),
                "best_iteration": best_iteration,
                **best_params,
                **metrics,
            }
        )

        iteration_rows_df = pd.DataFrame(
            build_iteration_history_rows(
                evals_result=evals_result,
                params=best_params,
                df_indx=df_indx,
                feature_set_name=feature_set_name,
                feature_set_code=feature_set_code,
                result_stage="test_best",
                trial_number=best_trial_number,
                run_name=best_run_name,
                fold_id=fold_id,
                best_iteration=best_iteration,
            )
        )
        iteration_rows_df.to_csv(
            iterations_file,
            mode="a",
            header=False,
            index=False,
        )

        # Store feature importance for this final selected-model fold.
        for _, importance_row in feature_importance_df.iterrows():
            feature_importance_rows.append(
                {
                    "df_indx": df_indx,
                    "feature_set_name": feature_set_name,
                    "feature_set_code": feature_set_code,
                    "result_stage": "test_best",
                    "trial_number": best_trial_number,
                    "run_name": best_run_name,
                **run_metadata,
                    "fold_id": fold_id,
                    "feature": importance_row["feature"],
                    **run_metadata,
                    "importance": importance_row["importance"],
                }
            )

        # Delete fold-specific feature matrices.
        del X_train
        del X_val
        del X_test

        # Delete fold-specific target matrices.
        del y_train
        del y_val

        # Delete test prediction arrays and feature-importance dataframe.
        del test_predictions_log
        del test_predictions
        del prediction_df
        del evals_result
        del feature_importance_df
        del iteration_rows_df

        # Ask Python to release unreachable memory before the next fold.
        gc.collect()

    # =====================================================
    # 9D. Save final result files after all loops
    # =====================================================

    # Convert every validation-fold row plus final selected-model test rows to a dataframe.
    fold_results_df = pd.DataFrame(
        fold_rows
    )

    # Convert feature importance rows to a dataframe.
    feature_importance_results_df = pd.DataFrame(
        feature_importance_rows
    )

    # Convert selected-model prediction rows to a dataframe.
    prediction_results_df = pd.DataFrame(
        prediction_rows
    )

    # Save aggregate hyperparameter-trial results.
    grid_results_df.to_csv(
        grid_results_file,
        index=False,
    )

    # Save every validation fold row plus the selected-model test fold rows.
    fold_results_df.to_csv(
        fold_results_file,
        index=False,
    )

    # Save feature importance from validation models and final selected models.
    feature_importance_results_df.to_csv(
        feature_importance_file,
        index=False,
    )

    # Save inverse-log predictions from the winning test model.
    prediction_results_df.to_csv(
        predictions_file,
        index=False,
    )

    # Print the winning trial number.
    print(
        f"\nBest trial: {best_trial_number}",
        flush=True,
    )

    # Print the lowest average validation RMSE.
    print(
        "Best average validation RMSE: "
        f"{grid_results_df.loc[0, 'Avg_RMSE_mean']:.4f}",
        flush=True,
    )

    # Print the grid-result location.
    print(
        f"Grid results saved to: {grid_results_file}",
        flush=True,
    )

    # Print the fold-result location.
    print(
        f"Fold results saved to: {fold_results_file}",
        flush=True,
    )

    # Print the feature-importance result location.
    print(
        f"Feature importance saved to: {feature_importance_file}",
        flush=True,
    )

    # Print the target order used throughout the run.
    print(
        f"Target order used: {target_cols}",
        flush=True,
    )

    # Print the number of model input features.
    reported_feature_count = len(feature_cols)
    if wiki_cols:
        reported_feature_count += wiki_model_dim
    print(
        f"Number of model features: {reported_feature_count}",
        flush=True,
    )

def main():
    """Run the three CatBoost dataframe experiments."""

    wiki_embedding_mode, n_wiki_components, wiki_mode_label = get_wiki_embedding_config()

    full_df, full_wiki_cols, node2vec_cols = load_model_dataframe()

    print(
        f"Loading fold assignments: {FOLD_ASSIGNMENTS_FILE}",
        flush=True,
    )
    fold_assignments_df = pd.read_parquet(
        FOLD_ASSIGNMENTS_FILE
    )
    print(
        f"Loaded fold-assignment shape: {fold_assignments_df.shape}",
        flush=True,
    )

    # Run one arm when CATBOOST_DF_INDX is set, otherwise run all three.
    df_indx_env = os.environ.get("CATBOOST_DF_INDX")
    selected_df_indices = (
        [int(df_indx_env)]
        if df_indx_env is not None
        else [0, 1, 2]
    )

    for df_indx in selected_df_indices:
        if df_indx == 0:
            df = full_df.drop(columns=node2vec_cols)
            wiki_cols = full_wiki_cols
            feature_set_name = "df0_baseline_semantic_no_node2vec"
            feature_set_code = "DF0_BASE_SEMANTIC"
        elif df_indx == 1:
            df = full_df.drop(columns=full_wiki_cols)
            wiki_cols = []
            feature_set_name = "df1_baseline_node2vec_no_semantic"
            feature_set_code = "DF1_BASE_NODE2VEC"
        else:
            df = full_df.drop(columns=[*full_wiki_cols, *node2vec_cols])
            wiki_cols = []
            feature_set_name = "df2_baseline"
            feature_set_code = "DF2_BASELINE"

        run_one_dataframe_variant(
            df=df,
            wiki_cols=wiki_cols,
            df_indx=df_indx,
            feature_set_name=feature_set_name,
            feature_set_code=feature_set_code,
            fold_assignments_df=fold_assignments_df,
            wiki_embedding_mode=wiki_embedding_mode,
            n_wiki_components=n_wiki_components,
            wiki_mode_label=wiki_mode_label,
        )
        del df
        gc.collect()

# =========================================================
# 9. Script entry point
# =========================================================

# Execute main only when this file is run directly from the WSL terminal.
if __name__ == "__main__":
    main()
