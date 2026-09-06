# =========================================================
# Auto-converted full-grid semantic script from bike_gat_stations_lvl.ipynb
# This script keeps the notebook logic but runs as a plain Python file.
# RUN_MODE is forced to 'full_grid', so it uses all configured GAT trials and all spatial folds.
# =========================================================


# %% [cell 1: markdown]
# # Station-Level GAT For Bike-Share Inflow And Outflow
#
# This notebook trains a **station-level Graph Attention Network (GAT)** using the prepared graph model data.
#
# The modelling level is important:
#
# - The **graph nodes** are Census Tracts / zones, identified by `loc_id`.
# - The **prediction rows** are bike-share **station-date** rows, identified by `station_name` and `date`.
# - Each station belongs to one host CT through `loc_id`.
# - The full CT graph is kept intact during message passing.
# - The train, validation, and test split is applied through station-level masks, not by deleting graph nodes or edges.
#
# The target is two-dimensional:
#
# 1. `inflow_count`
# 2. `outflow_count`
#
# Both targets are transformed as `log(1 + target)`.
#
# ## Data Used
#
# This notebook assumes the data is already prepared. It does not redo data cleaning.
#
# - Full MiniLM station-date graph model data: desc_semantics/desc_graph_df_minilm.parquet
# - MiniLM full graph dataframe: desc_semantics/desc_graph_df_minilm.parquet
# - GPT-large node embeddings: desc_semantics/desc_graph_gpt.parquet
# - GPT-small node embeddings: desc_semantics/desc_graph_gpt_small.parquet
# - Saved spatial fold assignments: desc_folds_of_2targets_txtTokens_graphFeat.parquet
# - Fixed CT graph nodes: `nodes.parquet`
# - Fixed CT graph edges: `edges.parquet`
#
# ## Four Feature Settings
#
# The same GAT training code is run four times:
#
# 1. graph_baseline: structured CT/node features only, with no semantic branch.
# 2. graph_minilm: structured CT/node features plus a MiniLM semantic branch using fold-fitted PCA32.
# 3. graph_gpt: structured CT/node features plus a GPT-large semantic branch using fold-fitted PCA32.
# 4. graph_gpt_small: structured CT/node features plus a GPT-small semantic branch using fold-fitted PCA32.
#
# ## Three-Branch Architecture
#
# The architecture separates macro spatial context, CT semantic context, and micro station context.
#
# Branch 1: Macro Spatial Context - CT Graph
#
# CT/node structured features
#   -> CT_FeatureEncoder
#   -> GAT Layer 1 with multi-head attention over the CT graph
#   -> GAT Layer 2 over the CT graph
#   -> Final CT graph embeddings
#
# Branch 2: CT Semantic Context - Description Embeddings
#
# MiniLM/GPT/GPT-small embeddings
#   -> PCA32 fitted on training CTs only
#   -> Semantic_FeatureEncoder
#   -> Final semantic embeddings
#
# Branch 3: Micro Local Context - Station Level
#
# Station features:
#   date-derived features
#   station capacity
#   allowed date-derived/context features
#
#   -> Station_FeatureEncoder
#   -> Final station embeddings
#
# Fusion and Prediction - Station-Date Level
#
# Gather each station host CT graph embedding
# Gather the same host CT semantic embedding when the feature mode has text
#   -> concatenate graph embedding, optional semantic embedding, and station embedding
#   -> RegressionHead
#   -> predicted inflow and outflow for each station-date row
#
# ## Feature-Level Rule
#
# - date, station_name, and loc_id are identifiers/keys only. They are not passed to the model as input features.
# - Station capacity is a station-level input feature.
# - inflow_count and outflow_count are station-level targets, not input features.
# - Date-derived fields such as day_type, public_holiday, Main_Weather_Category, and season are used in the station branch.
# - POIs, land use, population, jobs, and graph-derived variables remain in the CT/node branch.
# - Text embeddings do not enter the GAT node feature matrix; they are reduced to PCA32 and encoded separately.
#
# With hidden_dims = (128, 64, 32), N_TEXT_PCA_COMPONENTS = 32, and SEMANTIC_HIDDEN_DIM = 16, the intended shape is:
#
# CT_FeatureEncoder output       = 128
# GAT Layer 1 per-head output    = 64
# GAT Layer 2 final CT embedding = 32
# Semantic PCA input             = 32
# Semantic encoder output        = 16
# Station encoder output         = 32
# Baseline fusion vector         = 32 CT dims + 32 station dims = 64
# Semantic fusion vector         = 32 CT dims + 16 semantic dims + 32 station dims = 80
# RegressionHead output          = 2
#
# ## Snapshot Structure
#
# For each date, one graph snapshot is created:
#
# ```text
# x_ct                 = structured CT node feature matrix for this date
# x_semantic           = static CT semantic PCA32 matrix for semantic feature modes
# edge_index           = fixed Toronto CT graph edges
# edge_weight          = fixed inverse-distance affinity
# edge_attr            = edge_weight as an edge attribute for GATConv
# x_station            = station-level feature matrix for stations observed on this date
# station_ct_index     = CT node-row index for each station
# y_station            = log inflow and log outflow
# train/val/test masks = station-level masks
# ```
#
# No dropout is applied to raw input features. Dropout is only applied after hidden representations and inside GAT attention coefficients.
#
# ## Run Modes
#
# - `full_grid` is the default mode.
# - `smoke_test` is available for checking the pipeline quickly.
# - In `smoke_test`, only one fold and one hyperparameter setting are used.
# - In `full_grid`, all saved spatial folds and all grid settings are used.
# - Validation performance selects the final GAT configuration.

# %% [cell 2: code]
# =========================================================
# 1. Imports, project paths, and random seed
# =========================================================

# Standard library imports.
# These are used for paths, garbage collection, and copying the best model state.
import gc
import sys
import os
import time
from copy import deepcopy
from pathlib import Path

# Numerical and table libraries.
# NumPy handles arrays; pandas handles the parquet tables.
import numpy as np
import pandas as pd

# Script-safe display helper.
# In notebooks, IPython displays dataframes nicely.
# In a plain Python script, display falls back to print.
try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj)

# Scikit-learn utilities.
# ParameterGrid expands the hyperparameter grid.
# PCA reduces MiniLM/GPT embedding dimensions inside each fold.
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterGrid

# PyTorch libraries.
# torch builds tensors and trains the neural network.
import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch Geometric libraries.
# Data stores one daily graph snapshot.
# GATConv is the graph attention layer.
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

# Project paths.
# REPO_ROOT is the repository root.
# PROJECT_ROOT is the bikeshare project folder.
REPO_ROOT = Path('/home/najla/dev/najla-msc')
PROJECT_ROOT = REPO_ROOT / 'bikeshare'

# Add the bikeshare folder to sys.path so the shared helper can be imported.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Import the station-level helper used by the MLP and CatBoost station-level scripts.
# This keeps metric calculation, fold handling, target transforms, and neural helper blocks consistent.
from scripts.model_helper_stations_lvl import (
    BikeShareModelHelper as mh,
    FeatureEncoder,
    RegressionHead,
    get_activation,
    maybe_layer_norm,
)

# Select the compute device.
# CUDA is used only if it is available; otherwise the notebook runs on CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set the default random seed once at the top.
# Individual trials also reset the same seed so trial comparisons are fair.
mh.set_random_seed(42)

# Test call for this cell.
print('Project root:', PROJECT_ROOT)
print('Device:', device)

# %% [cell 3: code]
# =========================================================
# 2. File paths
# =========================================================

# Folder containing the prepared description-semantic model inputs.
DESC_SEMANTICS_MODEL_DF_DIR = PROJECT_ROOT / 'data/processed/model_df/desc_semantics'

# Full MiniLM station-date dataframe with the same structure as graph_df_minilm.parquet.
BASE_GRAPH_DF_FILE = DESC_SEMANTICS_MODEL_DF_DIR / 'desc_graph_df_minilm.parquet'

# The MiniLM embedding source is the same full dataframe; it is reduced to one row per loc_id later.
GRAPH_DF_MINILM_FILE = BASE_GRAPH_DF_FILE

# GPT large embedding table.
# This file contains one row per loc_id and GPT embedding columns gpt_emb_0, gpt_emb_1, ...
GRAPH_GPT_FILE = DESC_SEMANTICS_MODEL_DF_DIR / 'desc_graph_gpt.parquet'

# GPT small embedding table.
# This file contains one row per loc_id and GPT-small embedding columns gpt_small_emb_0, gpt_small_emb_1, ...
GRAPH_GPT_SMALL_FILE = DESC_SEMANTICS_MODEL_DF_DIR / 'desc_graph_gpt_small.parquet'

# Saved spatial fold assignments produced previously.
# This keeps the GAT folds aligned with the MLP and CatBoost experiments.
FOLD_ASSIGNMENTS_FILE = DESC_SEMANTICS_MODEL_DF_DIR / 'desc_folds_of_2targets_txtTokens_graphFeat.parquet'

# Fixed CT graph files.
# nodes.parquet defines the CT node universe.
# edges.parquet defines the Queen-contiguity graph and inverse-distance edge weights.
NODES_FILE = PROJECT_ROOT / 'data/processed/graph/nodes.parquet'
EDGES_FILE = PROJECT_ROOT / 'data/processed/graph/edges.parquet'

# Output folder for the late-fusion semantic-branch GAT result artifacts.
# This keeps the previous early-fusion GAT results separate.
RESULTS_DIR = PROJECT_ROOT / 'results/desc_semantics/gat_semantic_branch'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# The validation metric used to rank hyperparameter trials.
# Lower is better because this is mean squared error.
COMPARISON_METRIC = 'Val_Avg_MSE'

# Fast GAT execution switches.
# DAY_BATCH_SIZE controls how many daily station snapshots are packed into one optimizer step.
# The default is 30 days for the desc-semantic GAT runs.
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

DAY_BATCH_SIZE = int(os.environ.get('GAT_DAY_BATCH', '30'))
TORCH_NUM_THREADS = int(os.environ.get('GAT_TORCH_THREADS', '2'))
MAX_EPOCHS_OVERRIDE = int(os.environ.get('GAT_MAX_EPOCHS', '0')) or None
PATIENCE_OVERRIDE = int(os.environ.get('GAT_PATIENCE', '0')) or None
CHECKPOINT_DIR = RESULTS_DIR / 'checkpoints'
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def _flush(*args):
    '''Print and flush immediately so redirected logs are useful during long runs.'''
    print(*args, flush=True)

# Test call for this cell.
for file_path in [BASE_GRAPH_DF_FILE, GRAPH_DF_MINILM_FILE, GRAPH_GPT_FILE, GRAPH_GPT_SMALL_FILE, FOLD_ASSIGNMENTS_FILE, NODES_FILE, EDGES_FILE]:
    print(file_path.name, 'exists:', file_path.exists())

# %% [cell 4: code]
# =========================================================
# 3. Run mode and hyperparameter grid
# =========================================================

# Choose one of two modes:
# - 'full_grid' runs the complete grid on all spatial folds. This is the default thesis run.
# - 'smoke_test' runs one trial on one fold so the whole script can be checked quickly.
RUN_MODE = 'full_grid'

# Run all four feature variants in this order.
# This follows the requested workflow:
# 1. graph_baseline uses the base station-date graph features without text embeddings
# 2. graph_minilm uses MiniLM embeddings from desc_graph_df_minilm.parquet
# 3. graph_gpt uses GPT-large embeddings from desc_graph_gpt.parquet
# 4. graph_gpt_small uses GPT-small embeddings from desc_graph_gpt_small.parquet
FEATURE_SET_MODES = [
    'graph_baseline',
    'graph_minilm',
    'graph_gpt',
    'graph_gpt_small',
]

# Run one feature setting when GAT_FEATURE_MODE is set, otherwise run all four.
# This is the main parallelization hook for the full experiment: launch one process
# for graph_baseline, graph_minilm, graph_gpt, and graph_gpt_small. Each process
# writes feature-specific filenames, so the arms can run concurrently without
# overwriting one another.
GAT_FEATURE_MODE = os.environ.get('GAT_FEATURE_MODE')
if GAT_FEATURE_MODE is not None:
    if GAT_FEATURE_MODE not in FEATURE_SET_MODES:
        raise ValueError(f'GAT_FEATURE_MODE must be one of {FEATURE_SET_MODES}')
    ACTIVE_FEATURE_SET_MODES = [GAT_FEATURE_MODE]
else:
    ACTIVE_FEATURE_SET_MODES = list(FEATURE_SET_MODES)

# Text embeddings are reduced to 32 PCA dimensions inside each fold.
# PCA is fitted on training CTs only, then applied to all CTs.
N_TEXT_PCA_COMPONENTS = 32

# Semantic PCA features are encoded in their own branch before final fusion.
SEMANTIC_HIDDEN_DIM = 16

# Heavy test calls build full feature objects and snapshots.
# Keep this False during normal setup so test cells do not duplicate expensive preprocessing.
# Set it to True when you want to debug fold preparation before launching main().
RUN_HEAVY_TEST_CALLS = False


def get_gat_full_grid():
    '''Return the full GAT hyperparameter grid.'''

    # These values are selected as follows:
    # - hidden_dims, activation, learning_rate, weight_decay, AdamW, MSE, and early stopping
    #   follow the best/shared neural search space from the MLP experiments.
    # - attention_heads and attention_dropout are GAT-only parameters guided by GAT flow literature.
    # - input_dropout is fixed at 0.00 because raw CT and station inputs should not be randomly erased.
    # - Station_FeatureEncoder output is fixed to hidden_dims[2] so the grid does not expand.
    # - Semantic_FeatureEncoder output is fixed at SEMANTIC_HIDDEN_DIM so the search isolates
    #   the GAT hyperparameters rather than expanding a second architecture grid.
    gat_grid = {
        'hidden_dims': [(128, 64, 32)],
        'activation': ['relu'],
        'feature_dropout': [0.20],
        'learning_rate': [0.001, 0.0005],
        'weight_decay': [0.001],
        'attention_heads': [2, 4, 8],
        'attention_dropout': [0.10, 0.30],
        'input_dropout': [0.00],
        'optimizer_name': ['AdamW'],
        'use_layer_norm': [True],
        'loss_name': ['multi_target_mse'],
        'max_epochs': [120],
        'early_stopping_patience': [10],
        'scheduler_name': ['ReduceLROnPlateau'],
        'scheduler_patience': [4],
        'scheduler_factor': [0.50],
        'random_seed': [42],
    }

    # Expand the grid into a list of trial dictionaries.
    trials = list(ParameterGrid(gat_grid))

    # Add a clear trial id to every trial.
    for trial_id, trial in enumerate(trials):
        trial['trial_id'] = trial_id
        trial['trial_group'] = 'full_grid'

    return trials


def get_gat_smoke_grid():
    '''Return one small trial for checking that the full pipeline runs.'''

    # This is one representative GAT configuration.
    # It is not used for final model selection.
    trial = {
        'hidden_dims': (128, 64, 32),
        'activation': 'relu',
        'feature_dropout': 0.20,
        'learning_rate': 0.0005,
        'weight_decay': 0.001,
        'attention_heads': 4,
        'attention_dropout': 0.20,
        'input_dropout': 0.00,
        'optimizer_name': 'AdamW',
        'use_layer_norm': True,
        'loss_name': 'multi_target_mse',
        'max_epochs': 100,
        'early_stopping_patience': 10,
        'scheduler_name': 'ReduceLROnPlateau',
        'scheduler_patience': 4,
        'scheduler_factor': 0.50,
        'random_seed': 42,
        'trial_id': 0,
        'trial_group': 'smoke_test',
    }

    return [trial]


def get_gat_trials(run_mode):
    '''Return the selected trial list based on RUN_MODE.'''

    # Keep the mode explicit so mistakes fail early.
    if run_mode == 'full_grid':
        return get_gat_full_grid()

    if run_mode == 'smoke_test':
        return get_gat_smoke_grid()

    raise ValueError("RUN_MODE must be either 'full_grid' or 'smoke_test'.")


# Build the active trial list.
gat_trials = get_gat_trials(RUN_MODE)

# Convert the trial list to a dataframe for easy inspection.
gat_trials_df = pd.DataFrame(gat_trials)

# Test call for this cell.
print('RUN_MODE:', RUN_MODE)
print('Feature settings:', FEATURE_SET_MODES)
print('Active feature settings:', ACTIVE_FEATURE_SET_MODES)
print('Number of GAT trials per feature setting:', len(gat_trials_df))
display(gat_trials_df)

# %% [cell 5: code]
# =========================================================
# 4. Column definitions
# =========================================================

# CT identifier.
# This is the node id used by the fixed CT graph.
# It is used only for joins, graph node indexing, and host-CT lookup.
# It is not passed into the model as a feature.
CT_ID_COL = 'loc_id'

# Disconnected island CT removed from the fixed graph.
# This loc_id is filtered from nodes and edges before edge_index is built.
BROKEN_ISLAND_LOC_ID = '5350002.00'

# Internal text-availability column.
# This is used only to decide which CTs can fit text PCA and which PCA rows should be zeroed.
TEXT_AVAILABLE_COL = 'has_text'

# Date column.
# One GAT snapshot is created for each date.
# The raw date itself is not passed into the model as a feature.
DATE_COL = 'date'

# Station identifier.
# This is used only for station-date grouping and evaluation metadata.
# It is not passed into the model as a feature.
STATION_ID_COL = 'station_name'

# Station-level capacity.
# This is station capacity, not CT/node capacity.
STATION_CAPACITY_COL = 'station_capacity_avg'

# Spatial group column used by the saved spatial CV folds.
GROUP_COL = 'spatial_group'

# Two-target prediction setup.
# Target column order must stay consistent everywhere:
# column 0 = inflow_count, column 1 = outflow_count.
TARGET_COLS = ['inflow_count', 'outflow_count']
TRANSFORMED_TARGET_COLS = ['inflow_count_log', 'outflow_count_log']
TARGET_LABELS = ['Inflow', 'Outflow']

# Result dataframe names used in saved file names.
# These names prevent MiniLM, no-text baseline, and GPT outputs from overwriting each other.
RESULT_DF_NAME_BY_FEATURE_SET = {
    'graph_baseline': 'graph_baseline',
    'graph_minilm': 'graph_minilm',
    'graph_gpt': 'graph_gpt',
    'graph_gpt_small': 'graph_gpt_small',
}

# Columns that must not be used as model input features.
# These columns are metadata, identifiers, target variables, leakage-prone station identifiers,
# raw coordinates, or raw text. Some are still used for joins, snapshot grouping, masks, or evaluation.
DROP_FEATURE_COLS = [
    'loc_id',
    'loc_name',
    'loc_id_key',
    'lon',
    'lat',
    'spatial_group',
    'date',
    'inflow_count',
    'outflow_count',
    'end_station_name',
    'start_station_name',
    'station_name',
    'start_station_count',
    'end_station_count',
    'start_stations_count',
    'end_stations_count',
    'start_capacity_avg',
    'wiki_items_text',
]

# Date-derived/context fields for the station branch.
# These are not raw identifiers like date, station_name, or loc_id.
# If you want a strict no-temporal-feature setup, set this list to [].
STATION_DATE_DERIVED_COLS = [
    'day_type',
    'public_holiday',
    'Main_Weather_Category',
    'season',
]

# CT categorical features.
# Keep this empty because date-derived fields are encoded in the station branch.
# POIs, land-use, population, jobs, and text PCA stay in the CT branch as numerical features.
CT_CATEGORICAL_COLS = []

# Station-level numerical features.
# Capacity is a station property and is encoded before fusion through Station_FeatureEncoder.
STATION_NUMERICAL_COLS = [STATION_CAPACITY_COL]

# Station-level categorical/context features for sklearn preprocessing.
# station_name is intentionally excluded because it is an identifier.
STATION_CATEGORICAL_COLS = STATION_DATE_DERIVED_COLS

# Raw text columns should never be passed directly into StandardScaler or OneHotEncoder.
RAW_TEXT_COLS = [
    'wiki_items_text',
]

# Raw coordinate columns are excluded from node features here because spatial structure is represented by edge_index/edge_attr.
# If you intentionally want raw coordinates as features later, remove them from this list.
RAW_COORD_COLS = [
    'lat',
    'lon',
    'latitude',
    'longitude',
]

# Test call for this cell.
print('CT id column:', CT_ID_COL)
print('Station id column:', STATION_ID_COL)
print('Targets:', TARGET_COLS)
print('CT categorical features:', CT_CATEGORICAL_COLS)
print('Station numerical features:', STATION_NUMERICAL_COLS)
print('Station categorical/context features:', STATION_CATEGORICAL_COLS)
print('Identifier columns excluded from model inputs:', [CT_ID_COL, DATE_COL, STATION_ID_COL])
print('Number of blocked feature columns:', len(DROP_FEATURE_COLS))

# %% [cell 6: code]
# =========================================================
# 5. Read prepared data
# =========================================================

# Read the full prepared station-date dataframe.
base_model_df = pd.read_parquet(BASE_GRAPH_DF_FILE)

# Read GPT-large embeddings.
# This table is node-level: one row per loc_id.
gpt_embedding_df = pd.read_parquet(GRAPH_GPT_FILE)

# Read GPT-small embeddings.
# This table is node-level: one row per loc_id.
gpt_small_embedding_df = pd.read_parquet(GRAPH_GPT_SMALL_FILE)

# Read saved spatial fold assignment table.
fold_assignments_df = pd.read_parquet(FOLD_ASSIGNMENTS_FILE)

# Read the fixed CT graph tables.
nodes_df = pd.read_parquet(NODES_FILE)
edges_df = pd.read_parquet(EDGES_FILE)

# Filter the disconnected island CT from the node table before building the fixed graph.
# loc_id is compared as text here because the graph parquet stores CT ids with trailing decimals.
if CT_ID_COL in nodes_df.columns:
    nodes_df = (
        nodes_df
        .loc[~nodes_df[CT_ID_COL].astype(str).str.strip().eq(BROKEN_ISLAND_LOC_ID)]
        .reset_index(drop=True)
    )
else:
    nodes_df = nodes_df.reset_index().rename(columns={'index': CT_ID_COL})
    nodes_df = (
        nodes_df
        .loc[~nodes_df[CT_ID_COL].astype(str).str.strip().eq(BROKEN_ISLAND_LOC_ID)]
        .reset_index(drop=True)
    )

# Filter the disconnected island CT from the edge table as well.
# For this island the connected-edge count is zero, but this keeps the graph construction explicit.
if isinstance(edges_df.index, pd.MultiIndex):
    edge_source_ids = edges_df.index.get_level_values(0).astype(str).str.strip()
    edge_target_ids = edges_df.index.get_level_values(1).astype(str).str.strip()
    edges_df = edges_df.loc[~((edge_source_ids == BROKEN_ISLAND_LOC_ID) | (edge_target_ids == BROKEN_ISLAND_LOC_ID))].copy()
else:
    edges_df = edges_df.copy()
    edges_df['source'] = edges_df['source'].astype(str).str.strip()
    edges_df['target'] = edges_df['target'].astype(str).str.strip()
    edges_df = edges_df.loc[~(edges_df['source'].eq(BROKEN_ISLAND_LOC_ID) | edges_df['target'].eq(BROKEN_ISLAND_LOC_ID))].reset_index(drop=True)

# Make dates real datetime values.
# The data is already prepared; this only guarantees pandas grouping behaves correctly.
base_model_df[DATE_COL] = pd.to_datetime(base_model_df[DATE_COL])

# Keep loc_id as numeric so it matches nodes_df and embedding tables.
base_model_df[CT_ID_COL] = pd.to_numeric(base_model_df[CT_ID_COL], errors='coerce')
gpt_embedding_df[CT_ID_COL] = pd.to_numeric(gpt_embedding_df[CT_ID_COL], errors='coerce')
gpt_small_embedding_df[CT_ID_COL] = pd.to_numeric(gpt_small_embedding_df[CT_ID_COL], errors='coerce')

# Find MiniLM, GPT-large, and GPT-small embedding columns by prefix.
minilm_cols = [col for col in base_model_df.columns if col.startswith('wiki_emb_')]

# Keep one MiniLM embedding row per loc_id for fold-level text PCA.
minilm_embedding_df = (
    base_model_df[[CT_ID_COL] + minilm_cols]
    .drop_duplicates(subset=[CT_ID_COL], keep='first')
    .reset_index(drop=True)
)
gpt_cols = [col for col in gpt_embedding_df.columns if col.startswith('gpt_emb_')]
gpt_small_cols = [col for col in gpt_small_embedding_df.columns if col.startswith('gpt_small_emb_')]

# Test call for this cell.
print('base_model_df shape:', base_model_df.shape)
print('minilm_embedding_df shape:', minilm_embedding_df.shape)
print('gpt_embedding_df shape:', gpt_embedding_df.shape)
print('fold_assignments_df shape:', fold_assignments_df.shape)
print('nodes_df shape:', nodes_df.shape)
print('edges_df shape:', edges_df.shape)
print('MiniLM embedding columns:', len(minilm_cols))
print('GPT embedding columns:', len(gpt_cols))
print('Distinct loc_id in base_model_df:', base_model_df[CT_ID_COL].nunique())
print('Distinct loc_id in MiniLM embeddings:', minilm_embedding_df[CT_ID_COL].nunique())
print('Distinct loc_id in GPT embeddings:', gpt_embedding_df[CT_ID_COL].nunique())
print('Distinct loc_id in GPT-small embeddings:', gpt_small_embedding_df[CT_ID_COL].nunique())

# %% [cell 7: code]
# =========================================================
# 6. Build the fixed CT graph tensors
# =========================================================


def normalise_nodes_df(nodes_df):
    '''Return nodes_df with loc_id as a regular column.'''

    # Some graph files store loc_id as the index instead of a normal column.
    # This function makes both cases behave the same way.
    if CT_ID_COL in nodes_df.columns:
        out = nodes_df.copy()
    else:
        out = nodes_df.reset_index().rename(columns={'index': CT_ID_COL})

    # Make loc_id numeric for stable joining and mapping.
    out[CT_ID_COL] = pd.to_numeric(out[CT_ID_COL], errors='coerce')

    # Keep one row per CT node.
    out = out.drop_duplicates(subset=[CT_ID_COL]).reset_index(drop=True)

    return out


def build_fixed_graph_tensors(nodes_df, edges_df):
    '''Build edge_index, edge_weight, edge_attr, node_ids, and node_to_idx.'''

    # Normalize nodes so loc_id is a normal column.
    graph_nodes_df = normalise_nodes_df(nodes_df)

    # node_ids gives the fixed tensor row order.
    # Example: node_ids[0] is the loc_id stored in row 0 of x_ct.
    node_ids = graph_nodes_df[CT_ID_COL].to_numpy()

    # node_to_idx maps real loc_id values to tensor row indices.
    # PyTorch tensors must be indexed by 0, 1, 2, ... not by loc_id values such as 5350001.00.
    node_to_idx = {node_id: node_idx for node_idx, node_id in enumerate(node_ids)}

    # edges.parquet stores edge endpoints in its MultiIndex.
    # Convert the MultiIndex into a two-column dataframe.
    if isinstance(edges_df.index, pd.MultiIndex):
        edge_pairs_df = edges_df.reset_index()
        source_col = edge_pairs_df.columns[0]
        target_col = edge_pairs_df.columns[1]
    else:
        edge_pairs_df = edges_df.copy()
        source_col = 'source'
        target_col = 'target'

    # Convert endpoint ids to numeric values.
    edge_pairs_df[source_col] = pd.to_numeric(edge_pairs_df[source_col], errors='coerce')
    edge_pairs_df[target_col] = pd.to_numeric(edge_pairs_df[target_col], errors='coerce')

    # Keep only edges whose endpoints are present in the node list.
    edge_pairs_df['source_idx'] = edge_pairs_df[source_col].map(node_to_idx)
    edge_pairs_df['target_idx'] = edge_pairs_df[target_col].map(node_to_idx)
    edge_pairs_df = edge_pairs_df.dropna(subset=['source_idx', 'target_idx']).copy()

    # Use affinity_weight directly if it already exists.
    # Otherwise, treat weight as distance in metres and convert it to inverse-distance affinity.
    if 'affinity_weight' in edge_pairs_df.columns:
        edge_weight_np = edge_pairs_df['affinity_weight'].to_numpy(dtype=np.float32)
    elif 'weight' in edge_pairs_df.columns:
        distance_m = edge_pairs_df['weight'].to_numpy(dtype=np.float32)
        edge_weight_np = 1.0 / (distance_m + 1e-6)
    else:
        raise KeyError('Expected affinity_weight or distance weight.')

    # Build directed edge arrays.
    # GAT message passing uses directed edges, so an undirected spatial graph is represented by both directions.
    source_idx = edge_pairs_df['source_idx'].to_numpy(dtype=np.int64)
    target_idx = edge_pairs_df['target_idx'].to_numpy(dtype=np.int64)

    # Add reverse edges so information can pass both ways between adjacent CTs.
    edge_index_np = np.vstack([
        np.concatenate([source_idx, target_idx]),
        np.concatenate([target_idx, source_idx]),
    ])

    # Duplicate weights for the reverse edges.
    edge_weight_np = np.concatenate([edge_weight_np, edge_weight_np]).astype(np.float32)

    # Convert to PyTorch tensors.
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)
    edge_weight = torch.tensor(edge_weight_np, dtype=torch.float32)

    # GATConv expects edge attributes to be shaped [num_edges, edge_dim].
    # Here edge_dim = 1 because each edge has one inverse-distance affinity value.
    edge_attr = edge_weight.view(-1, 1)

    return {
        'graph_nodes_df': graph_nodes_df,
        'node_ids': node_ids,
        'node_to_idx': node_to_idx,
        'edge_index': edge_index,
        'edge_weight': edge_weight,
        'edge_attr': edge_attr,
    }


# Build the fixed graph once.
graph_tensors = build_fixed_graph_tensors(nodes_df, edges_df)

# Test call for this cell.
print('Number of CT nodes:', len(graph_tensors['node_ids']))
print('edge_index shape:', tuple(graph_tensors['edge_index'].shape))
print('edge_attr shape:', tuple(graph_tensors['edge_attr'].shape))
print('First 5 node ids:', graph_tensors['node_ids'][:5])

# %% [cell 8: code]
# =========================================================
# 7. Build feature-set inputs
# =========================================================


def build_text_status_df(row_df):
    '''Build one text-availability flag per CT node.'''

    # The prepared graph dataframe uses attraction_count = -1 and attraction_missing_flag = 1
    # to identify CTs without real Wikidata text.
    text_status_df = (
        row_df
        .drop_duplicates(subset=[CT_ID_COL], keep='first')
        [[CT_ID_COL, 'attraction_count', 'attraction_missing_flag']]
        .reset_index(drop=True)
    )

    # A CT has usable text only when attraction_count is not the missing sentinel
    # and the missing flag is not active.
    text_status_df[TEXT_AVAILABLE_COL] = ~(
        text_status_df['attraction_count'].eq(-1)
        | text_status_df['attraction_missing_flag'].eq(1)
    )

    return text_status_df[[CT_ID_COL, TEXT_AVAILABLE_COL]]


def make_unique_embedding_df(df, embedding_cols, text_status_df):
    '''Keep one embedding row per loc_id with one text-availability flag.'''

    # Text embeddings are node-level but base_model_df repeats them for many station-date rows.
    # Drop to one row per loc_id first, then select embedding columns.
    # This is much lighter than selecting 2.1M rows x 384 MiniLM columns before deduplication.
    embedding_df = (
        df.drop_duplicates(subset=[CT_ID_COL], keep='first')
        [[CT_ID_COL] + embedding_cols]
        .reset_index(drop=True)
    )

    # Attach the text-availability flag without exposing it to the model as an input feature.
    embedding_df = embedding_df.merge(
        text_status_df,
        on=CT_ID_COL,
        how='left',
    )
    embedding_df[TEXT_AVAILABLE_COL] = embedding_df[TEXT_AVAILABLE_COL].fillna(False)

    return embedding_df


def build_station_base_rows(row_df):
    '''Build station-date rows used for station-level prediction.'''

    # Keep only metadata, station capacity, allowed station-date context fields, and targets.
    # loc_id/date/station_name are kept here only for joins, snapshots, grouping, and evaluation.
    # They are not passed into the model as features.
    active_station_date_cols = [col for col in STATION_DATE_DERIVED_COLS if col in row_df.columns]
    keep_cols = [
        CT_ID_COL,
        DATE_COL,
        STATION_ID_COL,
        STATION_CAPACITY_COL,
        GROUP_COL,
    ] + active_station_date_cols + TARGET_COLS
    keep_cols = [col for col in keep_cols if col in row_df.columns]

    # A station row is labelled only when station_name and both targets are present.
    station_rows = row_df[keep_cols].copy()
    station_rows = station_rows.dropna(subset=[STATION_ID_COL] + TARGET_COLS)

    # The input is already unique at station-date level.
    # Grouping keeps the function robust if duplicated rows ever appear.
    # Capacity is averaged in case a duplicate station-date appears.
    # Date-derived fields use first because they are constant for a station-date.
    station_agg_dict = {
        STATION_CAPACITY_COL: (STATION_CAPACITY_COL, 'mean'),
        GROUP_COL: (GROUP_COL, 'first'),
        TARGET_COLS[0]: (TARGET_COLS[0], 'sum'),
        TARGET_COLS[1]: (TARGET_COLS[1], 'sum'),
    }
    for col in active_station_date_cols:
        station_agg_dict[col] = (col, 'first')

    station_rows = (
        station_rows.groupby([CT_ID_COL, DATE_COL, STATION_ID_COL], as_index=False)
        .agg(**station_agg_dict)
        .reset_index(drop=True)
    )

    return station_rows


def build_ct_base_rows(row_df):
    '''Build one CT-date feature row for each loc_id and date.'''

    # These columns are blocked from CT/node features.
    # The explicit DROP_FEATURE_COLS list is the main guardrail against identifier/target leakage.
    # STATION_CAPACITY_COL is also blocked from the CT branch because it belongs to the station branch.
    blocked_cols = set(
        DROP_FEATURE_COLS
        + [STATION_CAPACITY_COL]
        + RAW_TEXT_COLS
        + RAW_COORD_COLS
        + STATION_DATE_DERIVED_COLS
    )

    # Raw MiniLM/GPT embedding columns are handled separately through PCA.
    blocked_prefixes = ('wiki_emb_', 'gpt_emb_', 'gpt_small_emb_')

    # Keep only categorical columns that exist in the dataframe.
    ct_categorical_cols = [col for col in CT_CATEGORICAL_COLS if col in row_df.columns]

    # Numerical CT features are all numeric columns that are not blocked and are not raw embeddings.
    ct_numerical_cols = []
    for col in row_df.columns:
        if col in blocked_cols:
            continue
        if col in ct_categorical_cols:
            continue
        if col.startswith(blocked_prefixes):
            continue
        if pd.api.types.is_numeric_dtype(row_df[col]):
            ct_numerical_cols.append(col)

    # The CT-date table keeps loc_id/date/spatial_group as metadata only, plus allowed CT features.
    # loc_id/date/spatial_group are needed to build snapshots and masks, not passed as model features.
    keep_cols = [CT_ID_COL, DATE_COL, GROUP_COL] + ct_numerical_cols + ct_categorical_cols
    keep_cols = [col for col in keep_cols if col in row_df.columns]

    # Numeric CT features use mean because they are repeated node attributes.
    # Categorical CT features use first when present.
    agg_dict = {col: (col, 'mean') for col in ct_numerical_cols}
    agg_dict.update({col: (col, 'first') for col in ct_categorical_cols})
    agg_dict[GROUP_COL] = (GROUP_COL, 'first')

    # Build one row per CT-date.
    ct_rows = (
        row_df[keep_cols]
        .groupby([CT_ID_COL, DATE_COL], as_index=False)
        .agg(**agg_dict)
        .reset_index(drop=True)
    )

    return ct_rows, ct_numerical_cols, ct_categorical_cols


def build_feature_set_objects(feature_set_mode):
    '''Create base rows and text embedding table for one feature setting.'''

    # Drop any raw embedding columns still present in the base station-date dataframe.
    # Text is added later through fold-level PCA only for the text feature settings.
    embedding_cols_to_drop = [
        col for col in base_model_df.columns
        if col.startswith(('wiki_emb_', 'gpt_emb_', 'gpt_small_emb_'))
    ]
    row_df = base_model_df.drop(columns=embedding_cols_to_drop, errors='ignore').copy()

    # Build the node-level text-availability table once from the prepared graph dataframe.
    # This table controls PCA fitting and zeroing for no-text CTs, but it is not a model feature.
    text_status_df = build_text_status_df(row_df)

    # Decide which text source to use.
    # Build only the embedding table needed for the selected mode so baseline/GPT runs do not copy MiniLM columns.
    if feature_set_mode == 'graph_baseline':
        text_embedding_df = None
        text_embedding_cols = []
        text_source = 'none'
    elif feature_set_mode == 'graph_minilm':
        text_embedding_df = make_unique_embedding_df(minilm_embedding_df, minilm_cols, text_status_df)
        text_embedding_cols = minilm_cols
        text_source = 'MiniLM'
    elif feature_set_mode == 'graph_gpt':
        text_embedding_df = make_unique_embedding_df(gpt_embedding_df, gpt_cols, text_status_df)
        text_embedding_cols = gpt_cols
        text_source = 'GPT'
    elif feature_set_mode == 'graph_gpt_small':
        text_embedding_df = make_unique_embedding_df(gpt_small_embedding_df, gpt_small_cols, text_status_df)
        text_embedding_cols = gpt_small_cols
        text_source = 'GPT-small'
    else:
        raise ValueError("feature_set_mode must be one of FEATURE_SET_MODES.")

    # Build base station rows once for this feature setting.
    station_base_rows = build_station_base_rows(row_df)

    # Build base CT-date rows once for this feature setting.
    ct_base_rows, ct_base_numerical_cols, ct_base_categorical_cols = build_ct_base_rows(row_df)

    return {
        'feature_set_mode': feature_set_mode,
        'result_df_name': RESULT_DF_NAME_BY_FEATURE_SET[feature_set_mode],
        'text_source': text_source,
        'row_df': row_df,
        'station_base_rows': station_base_rows,
        'ct_base_rows': ct_base_rows,
        'ct_base_numerical_cols': ct_base_numerical_cols,
        'ct_base_categorical_cols': ct_base_categorical_cols,
        'text_embedding_df': text_embedding_df,
        'text_embedding_cols': text_embedding_cols,
    }


# Test call for this cell.
# This is intentionally lightweight; the full feature objects are built inside main().
for mode in FEATURE_SET_MODES:
    if mode == 'graph_minilm':
        text_source = 'MiniLM'
        text_col_count = len(minilm_cols)
    elif mode == 'graph_gpt':
        text_source = 'GPT'
        text_col_count = len(gpt_cols)
    elif mode == 'graph_gpt_small':
        text_source = 'GPT-small'
        text_col_count = len(gpt_small_cols)
    else:
        text_source = 'none'
        text_col_count = 0

    print(mode)
    print('  text source:', text_source)
    print('  raw text embedding columns:', text_col_count)

# %% [cell 9: code]
# =========================================================
# 8. Spatial folds
# =========================================================

# Build fold dictionaries from the saved fold assignment table.
# This follows the same station-level helper style used by MLP and CatBoost.
spcv_folds = mh.prepare_spatial_folds_from_assignments(
    df=base_model_df,
    fold_assignments_df=fold_assignments_df,
    group_col=GROUP_COL,
    include_dataframes=False,
)

# full_grid uses all folds.
# smoke_test uses only the first fold so the notebook can be tested quickly.
active_folds = spcv_folds[:1] if RUN_MODE == 'smoke_test' else spcv_folds

# Test call for this cell.
print('Total saved folds:', len(spcv_folds))
print('Active folds for this run:', len(active_folds))
print('First fold id:', active_folds[0]['fold_id'])
print('First fold train groups:', active_folds[0]['train_groups'][:5], '...')

# %% [cell 10: code]
# =========================================================
# 9. Fold-level preprocessing
# =========================================================


def build_split_by_group(fold):
    '''Create a spatial_group -> split lookup for one fold.'''

    # Store split labels by spatial group.
    split_by_group = {}

    # Assign train groups.
    for group in fold['train_groups']:
        split_by_group[group] = 'train'

    # Assign validation groups.
    for group in fold['val_groups']:
        split_by_group[group] = 'val'

    # Assign test groups.
    for group in fold['test_groups']:
        split_by_group[group] = 'test'

    return split_by_group


def fill_preprocessor_inputs(df, numerical_cols, categorical_cols):
    '''Fill missing values before sklearn preprocessing.'''

    # Work on a copy so the original dataframe is not changed accidentally.
    out = df.copy()

    # Numeric missing values are filled with 0.
    for col in numerical_cols:
        out[col] = pd.to_numeric(out[col], errors='coerce').fillna(0)

    # Categorical missing values are filled with a clear missing label.
    for col in categorical_cols:
        out[col] = out[col].astype('object').fillna('missing')

    return out


def fit_transform_text_pca(text_embedding_df, text_embedding_cols, train_ct_ids):
    '''Fit text PCA on training CTs with real text and transform all CTs.'''

    # Baseline mode has no text embeddings.
    if text_embedding_df is None or len(text_embedding_cols) == 0:
        return None, [], None

    # Keep embeddings for CTs that are present in the fixed graph.
    graph_node_ids = pd.Series(graph_tensors['node_ids'], name=CT_ID_COL)
    text_all_df = graph_node_ids.to_frame().merge(
        text_embedding_df[[CT_ID_COL, TEXT_AVAILABLE_COL] + text_embedding_cols],
        on=CT_ID_COL,
        how='left',
    )

    # CTs without real Wikidata text should not shape the PCA basis.
    text_all_df[TEXT_AVAILABLE_COL] = text_all_df[TEXT_AVAILABLE_COL].fillna(False).astype(bool)

    # Fit PCA only on training host CTs that actually have text.
    train_text_df = text_all_df[
        text_all_df[CT_ID_COL].isin(train_ct_ids)
        & text_all_df[TEXT_AVAILABLE_COL]
    ].copy()

    # Keep PCA fixed at N_TEXT_PCA_COMPONENTS for every semantic experiment.
    # If a fold cannot support the requested count, fail clearly instead of silently changing it.
    n_real_wiki_locations = len(train_text_df)
    n_wiki_components = N_TEXT_PCA_COMPONENTS

    if n_real_wiki_locations < n_wiki_components:
        raise ValueError(
            f'Fold has only {n_real_wiki_locations} real-Wikidata locations '
            f'but PCA requests {n_wiki_components} components.'
        )

    if len(text_embedding_cols) < n_wiki_components:
        raise ValueError(
            f'PCA requires at least {n_wiki_components} embedding columns; '
            f'found {len(text_embedding_cols)}.'
        )

    n_components = n_wiki_components

    # Build stable PCA column names.
    text_pca_cols = [f'text_pca_{i}' for i in range(n_components)]

    # Fit PCA on training CT embeddings with real text only.
    pca = PCA(n_components=n_components, random_state=42)
    train_matrix = train_text_df[text_embedding_cols].fillna(0).to_numpy(dtype=np.float32)
    pca.fit(train_matrix)

    # Transform all graph CT embeddings so every CT keeps the same feature width.
    all_matrix = text_all_df[text_embedding_cols].fillna(0).to_numpy(dtype=np.float32)
    pca_values = pca.transform(all_matrix).astype(np.float32)

    # CTs without real text receive a zero PCA vector after transformation.
    # This prevents the embedding for the literal no_wikidata token from becoming a learned semantic signal.
    no_text_mask = ~text_all_df[TEXT_AVAILABLE_COL].to_numpy(dtype=bool)
    pca_values[no_text_mask, :] = 0.00

    # Return one PCA row per loc_id.
    text_pca_df = pd.concat(
        [
            text_all_df[[CT_ID_COL]].reset_index(drop=True),
            pd.DataFrame(pca_values, columns=text_pca_cols),
        ],
        axis=1,
    )

    return text_pca_df, text_pca_cols, pca


def prepare_hierarchical_gat_fold(feature_data, fold):
    '''Prepare one fold for station-level hierarchical GAT training.'''

    # Build the spatial_group -> split lookup for this fold.
    split_by_group = build_split_by_group(fold)

    # Attach train/val/test labels to station rows using their host CT spatial group.
    station_rows = feature_data['station_base_rows'].copy()
    station_rows['split'] = station_rows[GROUP_COL].map(split_by_group)
    station_rows = station_rows[station_rows['split'].isin(['train', 'val', 'test'])].reset_index(drop=True)

    # Add log-transformed targets for both inflow and outflow.
    for raw_col, transformed_col in zip(TARGET_COLS, TRANSFORMED_TARGET_COLS):
        station_rows[transformed_col] = mh.target_transform(station_rows[raw_col], transform='log1p')

    # Training CT ids are host CTs of training station rows.
    train_ct_ids = station_rows.loc[station_rows['split'] == 'train', CT_ID_COL].dropna().unique()
    if len(train_ct_ids) == 0:
        raise ValueError(f"Fold {fold['fold_id']} has no training CT ids.")

    # Start from CT-date rows without raw text embeddings.
    ct_rows = feature_data['ct_base_rows'].copy()

    # Build fold-level text PCA when this feature setting uses text.
    # The PCA basis is fitted only on training CTs with real text, then applied to all CTs.
    # The resulting PCA32 matrix is deliberately not merged into ct_rows; it is kept for
    # the late-fusion semantic branch so text does not participate in GAT message passing.
    text_pca_df, text_pca_cols, pca = fit_transform_text_pca(
        text_embedding_df=feature_data['text_embedding_df'],
        text_embedding_cols=feature_data['text_embedding_cols'],
        train_ct_ids=train_ct_ids,
    )

    # Structured CT features only enter the GAT branch.
    # Semantic PCA vectors stay outside ct_numerical_cols, keeping x_ct as pure spatial/CT data.
    ct_numerical_cols = list(feature_data['ct_base_numerical_cols'])

    # semantic_features is aligned to the fixed graph node order, not to station-date rows.
    # This keeps the description signal static per CT and avoids repeating the same vector
    # across 1,551 dates before the model actually needs it.
    if text_pca_df is None:
        semantic_features = np.empty(
            (len(graph_tensors['node_ids']), 0),
            dtype=np.float32,
        )
        semantic_input_dim = 0
    else:
        semantic_node_df = pd.DataFrame({CT_ID_COL: graph_tensors['node_ids']})
        semantic_node_df = semantic_node_df.merge(
            text_pca_df,
            on=CT_ID_COL,
            how='left',
        )
        semantic_node_df[text_pca_cols] = semantic_node_df[text_pca_cols].fillna(0.0)
        semantic_features = semantic_node_df[text_pca_cols].to_numpy(dtype=np.float32)
        semantic_input_dim = len(text_pca_cols)

    # CT categorical features are temporal/weather categories already present in the CT-date table.
    ct_categorical_cols = list(feature_data['ct_base_categorical_cols'])

    # Fit the CT preprocessor only on training host CT rows.
    ct_fit_rows = ct_rows[ct_rows[CT_ID_COL].isin(train_ct_ids)].copy()
    ct_fit_rows = fill_preprocessor_inputs(ct_fit_rows, ct_numerical_cols, ct_categorical_cols)
    ct_transform_rows = fill_preprocessor_inputs(ct_rows, ct_numerical_cols, ct_categorical_cols)

    ct_preprocessor = mh.build_feature_preprocessor(
        numerical_cols=ct_numerical_cols,
        categorical_cols=ct_categorical_cols,
    )
    ct_preprocessor.fit(ct_fit_rows)
    X_ct_all = ct_preprocessor.transform(ct_transform_rows)
    X_ct_all = np.asarray(X_ct_all, dtype=np.float32)

    # Keep CT features as a compact NumPy matrix instead of adding hundreds of dataframe columns.
    # ct_processed_rows stores only metadata; X_ct_all stores feature values in the same row order.
    ct_feature_cols = [f'ct_x_{i}' for i in range(X_ct_all.shape[1])]
    ct_processed_rows = ct_rows[[CT_ID_COL, DATE_COL, GROUP_COL]].reset_index(drop=True)

    # Fit the station preprocessor only on training station rows.
    station_fit_rows = station_rows[station_rows['split'] == 'train'].copy()
    station_fit_rows = fill_preprocessor_inputs(station_fit_rows, STATION_NUMERICAL_COLS, STATION_CATEGORICAL_COLS)
    station_transform_rows = fill_preprocessor_inputs(station_rows, STATION_NUMERICAL_COLS, STATION_CATEGORICAL_COLS)

    station_preprocessor = mh.build_feature_preprocessor(
        numerical_cols=STATION_NUMERICAL_COLS,
        categorical_cols=STATION_CATEGORICAL_COLS,
    )
    station_preprocessor.fit(station_fit_rows)
    X_station_all = station_preprocessor.transform(station_transform_rows)
    X_station_all = np.asarray(X_station_all, dtype=np.float32)

    # Keep station features as a compact NumPy matrix instead of adding many dataframe columns.
    # station_processed_rows stores only metadata; X_station_all stores feature values in the same row order.
    station_feature_cols = [f'station_x_{i}' for i in range(X_station_all.shape[1])]
    station_metadata_cols = [
        CT_ID_COL,
        DATE_COL,
        STATION_ID_COL,
        STATION_CAPACITY_COL,
        GROUP_COL,
        'split',
    ] + TARGET_COLS + TRANSFORMED_TARGET_COLS

    station_processed_rows = station_rows[station_metadata_cols].reset_index(drop=True)

    return {
        'fold_id': fold['fold_id'],
        'feature_set_mode': feature_data['feature_set_mode'],
        'result_df_name': feature_data['result_df_name'],
        'text_source': feature_data['text_source'],
        'ct_rows': ct_processed_rows,
        'station_rows': station_processed_rows,
        'ct_features': X_ct_all,
        'station_features': X_station_all,
        'semantic_features': semantic_features,
        'semantic_input_dim': semantic_input_dim,
        'ct_feature_cols': ct_feature_cols,
        'station_feature_cols': station_feature_cols,
        'ct_input_dim': len(ct_feature_cols),
        'station_input_dim': len(station_feature_cols),
        'ct_preprocessor': ct_preprocessor,
        'station_preprocessor': station_preprocessor,
        'text_pca': pca,
        'text_pca_cols': text_pca_cols,
        'train_groups': fold['train_groups'],
        'val_groups': fold['val_groups'],
        'test_groups': fold['test_groups'],
    }


# Test call for this cell.
# The heavy branch prepares one real baseline fold.
# It is off by default because main() will do this work again during training.
_test_gat_fold = None
if RUN_HEAVY_TEST_CALLS:
    _test_feature_data = build_feature_set_objects('graph_baseline')
    _test_gat_fold = prepare_hierarchical_gat_fold(_test_feature_data, active_folds[0])
    print('Test fold id:', _test_gat_fold['fold_id'])
    print('CT input dim:', _test_gat_fold['ct_input_dim'])
    print('Station input dim:', _test_gat_fold['station_input_dim'])
    print('CT processed rows:', _test_gat_fold['ct_rows'].shape)
    print('Station processed rows:', _test_gat_fold['station_rows'].shape)
    print(_test_gat_fold['station_rows']['split'].value_counts().sort_index())
else:
    print('Fold preprocessing functions are ready.')
    print('Set RUN_HEAVY_TEST_CALLS = True to run one full fold-prep test.')

# %% [cell 11: code]
# =========================================================
# 10. Build station-level graph snapshots
# =========================================================



class DailyStationSnapshotDataset:
    '''Cached daily snapshot dataset for station-level GAT training.'''

    def __init__(self, gat_fold):
        '''Store processed fold rows, build date lookups, and cache snapshots on device.'''

        # Pull fixed graph objects.
        self.node_ids = graph_tensors['node_ids']
        self.node_to_idx = graph_tensors['node_to_idx']
        self.edge_index = graph_tensors['edge_index']
        self.edge_weight = graph_tensors['edge_weight']
        self.edge_attr = graph_tensors['edge_attr']

        # Pull fold-specific processed rows.
        self.ct_rows = gat_fold['ct_rows'].reset_index(drop=True)
        self.station_rows = gat_fold['station_rows'].reset_index(drop=True)
        self.ct_features = gat_fold['ct_features']
        self.station_features = gat_fold['station_features']
        self.semantic_features = gat_fold['semantic_features']
        self.ct_feature_cols = gat_fold['ct_feature_cols']
        self.station_feature_cols = gat_fold['station_feature_cols']

        # Store input dimensions so the object can be inspected quickly.
        self.ct_input_dim = len(self.ct_feature_cols)
        self.station_input_dim = len(self.station_feature_cols)
        self.semantic_input_dim = gat_fold['semantic_input_dim']
        self.x_semantic_static = torch.tensor(
            self.semantic_features,
            dtype=torch.float32,
        )

        # Use the union of CT dates and station dates.
        # This keeps the graph timeline complete while allowing dates with no station labels to be skipped in loss.
        self.snapshot_dates = sorted(set(self.ct_rows[DATE_COL]).union(set(self.station_rows[DATE_COL])))

        # Precompute integer row indices by date.
        # This avoids repeatedly scanning the whole dataframe inside every epoch.
        self.ct_date_indices = self.ct_rows.groupby(DATE_COL, sort=False).indices
        self.station_date_indices = self.station_rows.groupby(DATE_COL, sort=False).indices

        # Move the shared graph tensors to the device once.
        # Every cached snapshot then references the same graph tensors instead of re-transferring them each epoch.
        self.edge_index = self.edge_index.to(device)
        self.edge_weight = self.edge_weight.to(device)
        self.edge_attr = self.edge_attr.to(device)

        # Build every daily snapshot once and keep it on the device.
        # Within a fold these tensors never change, so rebuilding them every epoch is wasted work.
        self._cache = [
            self._build_snapshot(snapshot_index).to(device)
            for snapshot_index in range(len(self.snapshot_dates))
        ]

    def __len__(self):
        '''Return the number of daily snapshots.'''

        return len(self.snapshot_dates)

    def __getitem__(self, snapshot_index):
        '''Return the prebuilt snapshot for this date.'''

        return self._cache[snapshot_index]

    def __iter__(self):
        '''Yield one daily snapshot at a time.'''

        # Snapshots are already cached, so iteration only returns references to prebuilt Data objects.
        for snapshot_index in range(len(self)):
            yield self[snapshot_index]

    def _build_snapshot(self, snapshot_index):
        '''Create one PyTorch Geometric Data snapshot for the requested date.'''

        # Select the date represented by this snapshot.
        snapshot_date = self.snapshot_dates[snapshot_index]

        # Select CT rows for this date using precomputed integer positions.
        ct_positions = np.asarray(self.ct_date_indices.get(snapshot_date, []), dtype=np.int64)
        date_ct_rows = self.ct_rows.iloc[ct_positions]
        date_ct_features = self.ct_features[ct_positions] if len(ct_positions) > 0 else np.empty((0, self.ct_input_dim), dtype=np.float32)

        # Select station rows for this date using precomputed integer positions.
        station_positions = np.asarray(self.station_date_indices.get(snapshot_date, []), dtype=np.int64)
        date_station_rows = self.station_rows.iloc[station_positions].reset_index(drop=True)
        date_station_features = self.station_features[station_positions] if len(station_positions) > 0 else np.empty((0, self.station_input_dim), dtype=np.float32)

        # Build an empty CT feature matrix for the full fixed graph.
        # Every date has the same number of CT rows and the same edge_index.
        x_ct = torch.zeros((len(self.node_ids), len(self.ct_feature_cols)), dtype=torch.float32)

        # Convert loc_id values to tensor row indices.
        ct_idx = date_ct_rows[CT_ID_COL].map(self.node_to_idx)
        ct_keep = ct_idx.notna()
        ct_tensor_idx = torch.tensor(ct_idx[ct_keep].to_numpy(dtype=np.int64), dtype=torch.long)

        # Fill the CT rows observed for this date.
        # The source panel is expected to be complete after upstream data preparation.
        if len(ct_tensor_idx) > 0:
            x_ct[ct_tensor_idx] = torch.tensor(
                date_ct_features[ct_keep.to_numpy()],
                dtype=torch.float32,
            )

        # Convert each station's host loc_id to the CT tensor row index.
        station_ct_index_series = date_station_rows[CT_ID_COL].map(self.node_to_idx)
        station_keep = station_ct_index_series.notna()
        station_keep_np = station_keep.to_numpy()
        date_station_rows = date_station_rows.loc[station_keep].reset_index(drop=True)
        date_station_features = date_station_features[station_keep_np]
        station_ct_index_series = station_ct_index_series.loc[station_keep].reset_index(drop=True)

        # Build station tensors for this date.
        x_station = torch.tensor(
            date_station_features,
            dtype=torch.float32,
        )
        station_ct_index = torch.tensor(
            station_ct_index_series.to_numpy(dtype=np.int64),
            dtype=torch.long,
        )

        # Build log-scale target matrix [num_stations_on_date, 2].
        y_station = torch.tensor(
            date_station_rows[TRANSFORMED_TARGET_COLS].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )

        # Build raw target matrix [num_stations_on_date, 2].
        y_raw_station = torch.tensor(
            date_station_rows[TARGET_COLS].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )

        # Station-level masks decide where loss/metrics are calculated.
        train_mask_station = torch.tensor(date_station_rows['split'].eq('train').to_numpy(), dtype=torch.bool)
        val_mask_station = torch.tensor(date_station_rows['split'].eq('val').to_numpy(), dtype=torch.bool)
        test_mask_station = torch.tensor(date_station_rows['split'].eq('test').to_numpy(), dtype=torch.bool)
        label_mask_station = train_mask_station | val_mask_station | test_mask_station

        # Store one date snapshot.
        snapshot = Data(
            x_ct=x_ct,
            x_semantic=self.x_semantic_static,
            edge_index=self.edge_index,
            edge_weight=self.edge_weight,
            edge_attr=self.edge_attr,
            x_station=x_station,
            station_ct_index=station_ct_index,
            y_station=y_station,
            y_raw_station=y_raw_station,
            label_mask_station=label_mask_station,
            train_mask_station=train_mask_station,
            val_mask_station=val_mask_station,
            test_mask_station=test_mask_station,
        )

        # Keep date, loc_id, and station names as Python metadata for evaluation output.
        snapshot.snapshot_date = snapshot_date
        snapshot.station_loc_ids = date_station_rows[CT_ID_COL].tolist()
        snapshot.station_names = date_station_rows[STATION_ID_COL].tolist()

        return snapshot


def build_hierarchical_gat_snapshots(gat_fold):
    '''Create a cached daily snapshot dataset for one prepared fold.'''

    # The training loop still sees one graph snapshot per date.
    # The difference is that snapshots are built once per fold instead of rebuilt every epoch.
    return DailyStationSnapshotDataset(gat_fold)


# Test call for this cell.
# This uses the optional fold-prep test output from the previous cell.
_test_snapshots = None
if _test_gat_fold is not None:
    _test_snapshots = build_hierarchical_gat_snapshots(_test_gat_fold)
    first_snapshot = _test_snapshots[0]
    print('Number of daily snapshots:', len(_test_snapshots))
    print('First snapshot:', first_snapshot)
    print('First snapshot CT x shape:', tuple(first_snapshot.x_ct.shape))
    print('First snapshot station x shape:', tuple(first_snapshot.x_station.shape))
    print('First snapshot train stations:', int(first_snapshot.train_mask_station.sum()))
else:
    print('Cached snapshot builder is ready.')
    print('Set RUN_HEAVY_TEST_CALLS = True, rerun fold prep, then rerun this cell for a real snapshot test.')


# %% [cell 11b: code]
# =========================================================
# 10b. Fast day-batched fold tensors
# =========================================================


class FoldTensors:
    '''All prepared fold data as flat tensors sliced by contiguous day ranges.'''

    def __init__(self, gat_fold, device, day_batch_size=DAY_BATCH_SIZE):
        self.device = device
        self.day_batch_size = int(day_batch_size)

        node_ids = graph_tensors['node_ids']
        node_to_idx = graph_tensors['node_to_idx']
        self.num_nodes = len(node_ids)
        self.base_edge_index = graph_tensors['edge_index'].to(device)
        self.base_edge_attr = graph_tensors['edge_attr'].to(device)
        self._tiled_edge_cache = {}

        ct_rows = gat_fold['ct_rows']
        station_rows = gat_fold['station_rows']
        ct_features = gat_fold['ct_features']
        station_features = gat_fold['station_features']

        self.ct_input_dim = gat_fold['ct_input_dim']
        self.station_input_dim = gat_fold['station_input_dim']
        self.semantic_input_dim = gat_fold['semantic_input_dim']
        # Semantic vectors are CT-level and date-invariant. They are stored once per fold
        # in graph node order. Dynamic multi-day batches repeat them only when the CT graph
        # itself is block-diagonalized across days.
        if self.semantic_input_dim > 0:
            self.x_semantic_static = torch.from_numpy(
                np.ascontiguousarray(gat_fold['semantic_features']).copy()
            ).to(device)
        else:
            self.x_semantic_static = None

        all_dates = np.union1d(
            ct_rows[DATE_COL].to_numpy(),
            station_rows[DATE_COL].to_numpy(),
        )
        self.dates = np.sort(all_dates)
        self.num_days = len(self.dates)
        date_to_day = pd.Series(np.arange(self.num_days), index=self.dates)

        ct_day = date_to_day.reindex(ct_rows[DATE_COL].to_numpy()).to_numpy()
        ct_node = ct_rows[CT_ID_COL].map(node_to_idx).to_numpy()
        ct_valid = ~pd.isna(ct_node) & ~pd.isna(ct_day)
        ct_day = ct_day[ct_valid].astype(np.int64)
        ct_node = ct_node[ct_valid].astype(np.int64)
        ct_vals = ct_features[ct_valid]

        x_ct_all = np.zeros(
            (self.num_days, self.num_nodes, self.ct_input_dim),
            dtype=np.float32,
        )
        x_ct_all[ct_day, ct_node] = ct_vals

        day0 = x_ct_all[0]
        self.static_ct = True
        for start in range(0, self.num_days, 64):
            chunk = x_ct_all[start:start + 64]
            if not np.array_equal(chunk, np.broadcast_to(day0, chunk.shape)):
                self.static_ct = False
                break

        if self.static_ct:
            self.x_ct_static = torch.from_numpy(day0.copy()).to(device)
            self.x_ct_all = None
        else:
            self.x_ct_static = None
            self.x_ct_all = torch.from_numpy(x_ct_all).to(device)
        del x_ct_all

        st_day = date_to_day.reindex(station_rows[DATE_COL].to_numpy()).to_numpy()
        st_node = station_rows[CT_ID_COL].map(node_to_idx).to_numpy()
        st_valid = ~pd.isna(st_node) & ~pd.isna(st_day)

        st_day = st_day[st_valid].astype(np.int64)
        st_node = st_node[st_valid].astype(np.int64)
        st_feat = station_features[st_valid]
        st_meta = station_rows.loc[st_valid].reset_index(drop=True)

        order = np.argsort(st_day, kind='stable')
        st_day = st_day[order]
        st_node = st_node[order]
        st_feat = st_feat[order]
        st_meta = st_meta.iloc[order].reset_index(drop=True)

        counts = np.bincount(st_day, minlength=self.num_days)
        self.station_ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

        split_np = st_meta['split'].to_numpy()
        self.masks_np = {
            'train': split_np == 'train',
            'val': split_np == 'val',
            'test': split_np == 'test',
        }

        self._batch_bounds = [
            (d0, min(d0 + self.day_batch_size, self.num_days))
            for d0 in range(0, self.num_days, self.day_batch_size)
        ]
        self.batch_counts = {}
        for split_name, mask_np in self.masks_np.items():
            csum = np.concatenate([[0], np.cumsum(mask_np)])
            self.batch_counts[split_name] = [
                int(csum[self.station_ptr[d1]] - csum[self.station_ptr[d0]])
                for d0, d1 in self._batch_bounds
            ]

        self.x_station = torch.from_numpy(np.ascontiguousarray(st_feat).copy()).to(device)
        self.node_idx = torch.from_numpy(np.ascontiguousarray(st_node).copy()).to(device)
        self.day_idx = torch.from_numpy(np.ascontiguousarray(st_day).copy()).to(device)
        self.y_log = torch.from_numpy(
            np.ascontiguousarray(st_meta[TRANSFORMED_TARGET_COLS].to_numpy(dtype=np.float32)).copy()
        ).to(device)
        self.y_raw = torch.from_numpy(
            np.ascontiguousarray(st_meta[TARGET_COLS].to_numpy(dtype=np.float32)).copy()
        ).to(device)
        self.masks = {
            split_name: torch.from_numpy(np.ascontiguousarray(mask_np).copy()).to(device)
            for split_name, mask_np in self.masks_np.items()
        }

        self.meta_date = st_meta[DATE_COL].to_numpy()
        self.meta_loc = st_meta[CT_ID_COL].to_numpy()
        self.meta_station = st_meta[STATION_ID_COL].to_numpy()

    def _tiled_edges(self, num_batch_days):
        '''Block-diagonal edge_index and edge_attr for a multi-day CT graph batch.'''
        cached = self._tiled_edge_cache.get(num_batch_days)
        if cached is not None:
            return cached

        offsets = (
            torch.arange(num_batch_days, device=self.device) * self.num_nodes
        ).view(num_batch_days, 1, 1)
        edge_index = (
            (self.base_edge_index.unsqueeze(0) + offsets)
            .permute(1, 0, 2)
            .reshape(2, -1)
            .contiguous()
        )
        edge_attr = self.base_edge_attr.repeat(num_batch_days, 1)
        self._tiled_edge_cache[num_batch_days] = (edge_index, edge_attr)
        return edge_index, edge_attr

    def batches(self, split_name):
        '''Yield one 30-day forward-pass batch at a time for the requested split.'''
        for batch_id, (d0, d1) in enumerate(self._batch_bounds):
            selected_count = self.batch_counts[split_name][batch_id]
            if selected_count == 0:
                continue

            r0 = int(self.station_ptr[d0])
            r1 = int(self.station_ptr[d1])
            rows = slice(r0, r1)

            if self.static_ct:
                x_ct = self.x_ct_static
                edge_index = self.base_edge_index
                edge_attr = self.base_edge_attr
                node_index = self.node_idx[rows]
                x_semantic = self.x_semantic_static
            else:
                num_batch_days = d1 - d0
                x_ct = self.x_ct_all[d0:d1].reshape(-1, self.ct_input_dim)
                edge_index, edge_attr = self._tiled_edges(num_batch_days)
                node_index = self.node_idx[rows] + (self.day_idx[rows] - d0) * self.num_nodes
                if self.x_semantic_static is not None:
                    x_semantic = self.x_semantic_static.repeat(num_batch_days, 1)
                else:
                    x_semantic = None

            yield {
                'x_ct': x_ct,
                'x_semantic': x_semantic,
                'edge_index': edge_index,
                'edge_attr': edge_attr,
                'x_station': self.x_station[rows],
                'node_index': node_index,
                'rows': rows,
                'mask': self.masks[split_name][rows],
                'selected_count': selected_count,
            }


def build_fold_tensors(gat_fold):
    '''Create the flat tensor container used by the fast training loop.'''
    return FoldTensors(gat_fold, device=device, day_batch_size=DAY_BATCH_SIZE)

# %% [cell 12: code]
# =========================================================
# 11. GAT architecture
# =========================================================


class StationLevelGAT(nn.Module):
    # Three-branch hierarchical GAT for station-level inflow/outflow.

    def __init__(
        self,
        ct_input_dim,
        station_input_dim,
        semantic_input_dim,
        hidden_dims,
        activation,
        attention_heads,
        attention_dropout,
        feature_dropout,
        input_dropout,
        use_layer_norm,
        semantic_hidden_dim=SEMANTIC_HIDDEN_DIM,
    ):
        # Create the graph, station, semantic, and regression layers.
        super().__init__()

        # Raw input dropout is intentionally fixed to zero.
        # Dropping raw CT features or raw station features would remove stable spatial/semantic anchors.
        if input_dropout != 0.00:
            raise ValueError('input_dropout must be 0.00 for this GAT setup.')

        # hidden_dims = (encoder_hidden_dim, gat_hidden_dim, final_ct_hidden_dim)
        encoder_hidden_dim, gat_hidden_dim, final_ct_hidden_dim = hidden_dims

        self.activation = get_activation(activation)
        self.feature_dropout = nn.Dropout(feature_dropout)
        self.semantic_input_dim = semantic_input_dim
        self.semantic_hidden_dim = semantic_hidden_dim if semantic_input_dim > 0 else 0

        # -----------------------------
        # Branch 1: structured CT graph
        # -----------------------------

        # x_ct contains only structured CT features now.
        # MiniLM/GPT PCA values are intentionally excluded from this encoder and the GAT layers.
        self.ct_encoder = FeatureEncoder(
            input_dim=ct_input_dim,
            hidden_dim=encoder_hidden_dim,
            activation=activation,
            use_layer_norm=use_layer_norm,
            dropout=0.00,
        )

        self.gat1 = GATConv(
            in_channels=encoder_hidden_dim,
            out_channels=gat_hidden_dim,
            heads=attention_heads,
            concat=True,
            dropout=attention_dropout,
            edge_dim=1,
            fill_value='mean',
        )
        self.norm1 = maybe_layer_norm(gat_hidden_dim * attention_heads, use_layer_norm)

        self.gat2 = GATConv(
            in_channels=gat_hidden_dim * attention_heads,
            out_channels=final_ct_hidden_dim,
            heads=1,
            concat=False,
            dropout=attention_dropout,
            edge_dim=1,
            fill_value='mean',
        )
        self.norm2 = maybe_layer_norm(final_ct_hidden_dim, use_layer_norm)

        # -----------------------------
        # Branch 2: CT semantic context
        # -----------------------------

        # Semantic input is already PCA32 and fold-safe. This encoder compresses that
        # representation to 16 dimensions before fusion so text can help the regression
        # head without forcing the GAT attention layers to propagate text features.
        if semantic_input_dim > 0:
            self.semantic_encoder = FeatureEncoder(
                input_dim=semantic_input_dim,
                hidden_dim=semantic_hidden_dim,
                activation=activation,
                use_layer_norm=use_layer_norm,
                dropout=feature_dropout,
            )
        else:
            self.semantic_encoder = None

        # -----------------------------
        # Branch 3: station/date context
        # -----------------------------

        self.station_encoder = FeatureEncoder(
            input_dim=station_input_dim,
            hidden_dim=final_ct_hidden_dim,
            activation=activation,
            use_layer_norm=use_layer_norm,
            dropout=feature_dropout,
        )

        # -----------------------------
        # Late fusion and prediction
        # -----------------------------

        # Baseline: 32 graph + 32 station = 64.
        # Semantic modes: 32 graph + 16 semantic + 32 station = 80.
        fusion_dim = final_ct_hidden_dim + self.semantic_hidden_dim + final_ct_hidden_dim
        self.regression_head = RegressionHead(hidden_dim=fusion_dim, output_dim=2)

    def forward(self, x_ct, x_semantic, edge_index, edge_attr, x_station, station_ct_index):
        # Predict station-level inflow and outflow for one date or one day batch.

        # Branch 1 learns a structured spatial CT representation through GAT message passing.
        h_ct = self.ct_encoder(x_ct)
        h_ct = self.gat1(h_ct, edge_index, edge_attr=edge_attr)
        h_ct = self.activation(h_ct)
        h_ct = self.norm1(h_ct)
        h_ct = self.feature_dropout(h_ct)

        h_ct = self.gat2(h_ct, edge_index, edge_attr=edge_attr)
        h_ct = self.activation(h_ct)
        h_ct = self.norm2(h_ct)
        h_ct = self.feature_dropout(h_ct)

        # Gather each station's host CT graph embedding.
        station_ct_embeddings = h_ct[station_ct_index]

        # Branch 2 encodes CT semantics separately, then gathers the same host CT
        # used by the station row. This is late fusion: semantic vectors never alter
        # the neighbourhood attention weights learned by gat1/gat2.
        if self.semantic_encoder is not None:
            h_semantic = self.semantic_encoder(x_semantic)
            station_semantic_embeddings = h_semantic[station_ct_index]
        else:
            station_semantic_embeddings = None

        # Branch 3 encodes station/date context for the station row being predicted.
        h_station = self.station_encoder(x_station)

        if station_semantic_embeddings is not None:
            station_hidden = torch.cat(
                [
                    station_ct_embeddings,
                    station_semantic_embeddings,
                    h_station,
                ],
                dim=1,
            )
        else:
            # Baseline remains graph + station only.
            station_hidden = torch.cat([station_ct_embeddings, h_station], dim=1)

        return self.regression_head(station_hidden)


# Test call for this cell.
_dummy_model = StationLevelGAT(
    ct_input_dim=5,
    station_input_dim=3,
    semantic_input_dim=32,
    hidden_dims=(128, 64, 32),
    activation='relu',
    attention_heads=4,
    attention_dropout=0.10,
    feature_dropout=0.20,
    input_dropout=0.00,
    use_layer_norm=True,
)
_dummy_x_ct = torch.randn(4, 5)
_dummy_edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
_dummy_edge_attr = torch.ones((4, 1), dtype=torch.float32)
_dummy_x_semantic = torch.randn(4, 32)
_dummy_x_station = torch.randn(2, 3)
_dummy_station_ct_index = torch.tensor([0, 2], dtype=torch.long)
_dummy_output = _dummy_model(
    _dummy_x_ct,
    _dummy_x_semantic,
    _dummy_edge_index,
    _dummy_edge_attr,
    _dummy_x_station,
    _dummy_station_ct_index,
)
print('Dummy output shape:', tuple(_dummy_output.shape))
del _dummy_model, _dummy_x_ct, _dummy_x_semantic, _dummy_edge_index, _dummy_edge_attr, _dummy_x_station, _dummy_station_ct_index, _dummy_output

# %% [cell 13: code]
# =========================================================
# 12. Training and evaluation functions
# =========================================================


def move_snapshot_tensors(snapshot, device):
    '''Move tensor attributes from one snapshot to the selected device.'''

    # Build a lightweight dictionary instead of mutating the original snapshot.
    return {
        'x_ct': snapshot.x_ct.to(device),
        'x_semantic': snapshot.x_semantic.to(device),
        'edge_index': snapshot.edge_index.to(device),
        'edge_attr': snapshot.edge_attr.to(device),
        'x_station': snapshot.x_station.to(device),
        'station_ct_index': snapshot.station_ct_index.to(device),
        'y_station': snapshot.y_station.to(device),
        'y_raw_station': snapshot.y_raw_station.to(device),
        'train_mask_station': snapshot.train_mask_station.to(device),
        'val_mask_station': snapshot.val_mask_station.to(device),
        'test_mask_station': snapshot.test_mask_station.to(device),
    }


def forward_snapshot(model, snapshot, device):
    '''Run the model on one daily graph snapshot.'''

    # Move tensors to CPU/GPU.
    batch = move_snapshot_tensors(snapshot, device)

    # Forward pass through CT encoder, GAT layers, station gather, station encoder, and regression head.
    pred_log = model(
        x_ct=batch['x_ct'],
        x_semantic=batch['x_semantic'],
        edge_index=batch['edge_index'],
        edge_attr=batch['edge_attr'],
        x_station=batch['x_station'],
        station_ct_index=batch['station_ct_index'],
    )

    return pred_log, batch


def masked_station_mse(pred_log, y_log, mask):
    '''Calculate MSE only on selected station rows.'''

    # If the date has no stations for this split, return None so the caller can skip it.
    if int(mask.sum()) == 0:
        return None

    # Multi-target MSE compares both inflow and outflow together.
    return F.mse_loss(pred_log[mask], y_log[mask])


def masked_station_sse_and_count(pred_log, y_log, mask):
    '''Return squared-error sum and value count for observation-weighted MSE.'''

    # If the date has no stations for this split, return no contribution.
    if int(mask.sum()) == 0:
        return None, 0

    # Sum squared error over all selected station rows and both targets.
    # Dividing the accumulated sum by the accumulated count gives row-level MSE across station-date observations.
    squared_error_sum = F.mse_loss(pred_log[mask], y_log[mask], reduction='sum')
    value_count = int(pred_log[mask].numel())

    return squared_error_sum, value_count


def evaluate_loss_for_split(model, snapshots, split_name, device):
    '''Calculate observation-weighted average log-scale MSE for one split.'''

    # Put the model in evaluation mode.
    model.eval()

    # Accumulate squared error across station-date rows, not across dates.
    total_squared_error = 0.0
    total_value_count = 0

    # No gradients are needed for validation/testing.
    with torch.no_grad():
        for snapshot in snapshots:
            pred_log, batch = forward_snapshot(model, snapshot, device)
            mask = batch[f'{split_name}_mask_station']
            squared_error_sum, value_count = masked_station_sse_and_count(pred_log, batch['y_station'], mask)
            if squared_error_sum is not None:
                total_squared_error += float(squared_error_sum.item())
                total_value_count += value_count

    # Return NaN if no rows exist for the split.
    if total_value_count == 0:
        return np.nan

    return float(total_squared_error / total_value_count)


def collect_station_predictions(model, snapshots, split_name, device):
    '''Collect raw-scale predictions and targets for one split.'''

    # Evaluation mode disables dropout.
    model.eval()

    # Store metadata, true targets, and predictions.
    metadata_parts = []
    y_true_parts = []
    y_pred_parts = []

    with torch.no_grad():
        for snapshot in snapshots:
            pred_log, batch = forward_snapshot(model, snapshot, device)
            mask = batch[f'{split_name}_mask_station'].detach().cpu().numpy()

            # Skip dates with no selected station rows.
            if int(mask.sum()) == 0:
                continue

            # Convert predicted log targets back to original trip-count scale.
            pred_raw = mh.inverse_target_transform(
                pred_log.detach().cpu().numpy()[mask],
                transform='log1p',
            )

            # Use original raw targets for evaluation.
            true_raw = batch['y_raw_station'].detach().cpu().numpy()[mask]

            # Build station-date metadata for ranking metrics and prediction export.
            # date, loc_id, and station_name are used here only for evaluation output, not as model features.
            station_loc_ids = np.asarray(snapshot.station_loc_ids, dtype=object)[mask]
            station_names = np.asarray(snapshot.station_names, dtype=object)[mask]
            metadata_df = pd.DataFrame({
                DATE_COL: [snapshot.snapshot_date] * int(mask.sum()),
                CT_ID_COL: station_loc_ids,
                STATION_ID_COL: station_names,
            })

            metadata_parts.append(metadata_df)
            y_true_parts.append(true_raw)
            y_pred_parts.append(pred_raw)

    # If no rows exist for this split, return empty values.
    if len(metadata_parts) == 0:
        return pd.DataFrame(columns=[DATE_COL, STATION_ID_COL]), np.empty((0, 2)), np.empty((0, 2))

    # Combine all dates.
    metadata_df = pd.concat(metadata_parts, axis=0, ignore_index=True)
    y_true = np.vstack(y_true_parts)
    y_pred = np.vstack(y_pred_parts)

    return metadata_df, y_true, y_pred


def build_prediction_output_df(metadata_df, y_true, y_pred):
    '''Build raw-scale station prediction rows for export.'''

    # Save predictions after inverse log transform.
    # Column order keeps the station-date identifiers first, followed by predictions and actual values.
    prediction_df = metadata_df.copy()
    prediction_df['predicted_inflow'] = y_pred[:, 0]
    prediction_df['predicted_outflow'] = y_pred[:, 1]
    prediction_df['actual_inflow'] = y_true[:, 0]
    prediction_df['actual_outflow'] = y_true[:, 1]

    return prediction_df[
        [
            DATE_COL,
            CT_ID_COL,
            STATION_ID_COL,
            'predicted_inflow',
            'predicted_outflow',
            'actual_inflow',
            'actual_outflow',
        ]
    ]


def evaluate_metrics_for_split(model, snapshots, split_name, device):
    '''Evaluate two-target station metrics for one split.'''

    # Collect raw-scale station predictions.
    metadata_df, y_true, y_pred = collect_station_predictions(model, snapshots, split_name, device)

    # If no rows exist, return an empty metric dictionary.
    if len(metadata_df) == 0:
        return {}

    # Reuse the same station-level helper metrics as MLP and CatBoost.
    metrics = mh.evaluate_two_targets(
        metadata_df=metadata_df,
        y_true=y_true,
        y_pred=y_pred,
        date_col=DATE_COL,
        id_col=STATION_ID_COL,
        target_labels=TARGET_LABELS,
        k_values=(10, 20),
    )

    return metrics


def train_one_gat_fold(
    gat_fold,
    snapshots,
    params,
    device,
    metric_split='val',
    return_predictions=False,
    return_epoch_history=False,
):
    '''Train one GAT trial on one spatial fold and evaluate one selected split.'''

    # metric_split controls what metrics are returned after training.
    # During hyperparameter search this must be 'val'.
    # During final unbiased evaluation this must be 'test'.
    if metric_split not in ['val', 'test']:
        raise ValueError("metric_split must be either 'val' or 'test'.")

    # Reset random seeds at the start of every fold/trial.
    mh.set_random_seed(params['random_seed'])

    # Create the model for this fold.
    model = StationLevelGAT(
        ct_input_dim=gat_fold['ct_input_dim'],
        station_input_dim=gat_fold['station_input_dim'],
        semantic_input_dim=gat_fold['semantic_input_dim'],
        hidden_dims=params['hidden_dims'],
        activation=params['activation'],
        attention_heads=params['attention_heads'],
        attention_dropout=params['attention_dropout'],
        feature_dropout=params['feature_dropout'],
        input_dropout=params['input_dropout'],
        use_layer_norm=params['use_layer_norm'],
    ).to(device)

    # Use AdamW to match the MLP-guided neural search setup.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params['learning_rate'],
        weight_decay=params['weight_decay'],
    )

    # ReduceLROnPlateau lowers the learning rate when validation loss stops improving.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=params['scheduler_factor'],
        patience=params['scheduler_patience'],
    )

    # Track early stopping state.
    best_val_loss = np.inf
    best_state_dict = None
    best_epoch = -1
    epochs_without_improvement = 0
    last_train_loss = np.nan
    train_loss_at_best_epoch = np.nan

    # Store one row per epoch so convergence and early stopping can be inspected later.
    epoch_history_rows = []

    # Training loop.
    for epoch in range(params['max_epochs']):
        model.train()
        train_squared_error_sum = 0.0
        train_value_count = 0

        # Loop through daily snapshots.
        for snapshot in snapshots:
            # Forward pass.
            pred_log, batch = forward_snapshot(model, snapshot, device)

            # Calculate loss only on training station rows.
            loss = masked_station_mse(pred_log, batch['y_station'], batch['train_mask_station'])

            # Skip dates with no training rows.
            if loss is None:
                continue

            # Count selected target values so the reported training loss is weighted by station-date rows.
            selected_value_count = int(batch['train_mask_station'].sum().item()) * batch['y_station'].shape[1]

            # Backpropagate this date loss.
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            # Store weighted training-loss components for epoch diagnostics.
            train_squared_error_sum += float(loss.item()) * selected_value_count
            train_value_count += selected_value_count

        # Report training loss across station-date observations, not as an unweighted average across dates.
        last_train_loss = float(train_squared_error_sum / train_value_count) if train_value_count > 0 else np.nan

        # Calculate validation loss on validation station rows only.
        # Validation loss is used for early stopping in both the search phase and the final test phase.
        val_loss = evaluate_loss_for_split(model, snapshots, 'val', device)
        scheduler.step(val_loss)

        # Early stopping check.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = deepcopy(model.state_dict())
            best_epoch = epoch
            train_loss_at_best_epoch = last_train_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Store epoch-level optimization diagnostics.
        epoch_history_rows.append({
            'feature_set_mode': gat_fold['feature_set_mode'],
            'result_df_name': gat_fold['result_df_name'],
            'text_source': gat_fold['text_source'],
            'fold_id': gat_fold['fold_id'],
            'trial_id': params['trial_id'],
            'metric_split': metric_split,
            'epoch': epoch,
            'train_log_mse': last_train_loss,
            'val_log_mse': val_loss,
            'best_val_log_mse_so_far': best_val_loss,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'epochs_without_improvement': epochs_without_improvement,
        })

        # Stop training if validation loss has not improved for the patience window.
        if epochs_without_improvement >= params['early_stopping_patience']:
            break

    # Restore the best validation checkpoint.
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # Convert epoch diagnostics to a dataframe after best_epoch is known.
    epoch_history_df = pd.DataFrame(epoch_history_rows)
    if len(epoch_history_df) > 0:
        epoch_history_df['best_epoch'] = best_epoch
        epoch_history_df['is_best_epoch'] = epoch_history_df['epoch'].eq(best_epoch)
        epoch_history_df['train_log_mse_at_best_epoch'] = train_loss_at_best_epoch
        epoch_history_df['last_epoch_train_log_mse'] = last_train_loss
        for key, value in params.items():
            if isinstance(value, (tuple, list, dict, set)):
                epoch_history_df[key] = [value] * len(epoch_history_df)
            else:
                epoch_history_df[key] = value

    # Evaluate exactly one split after training.
    # Hyperparameter search calls this with metric_split='val'.
    # Final unbiased evaluation calls this with metric_split='test'.
    metadata_df, y_true, y_pred = collect_station_predictions(model, snapshots, metric_split, device)
    if len(metadata_df) == 0:
        split_metrics = {}
    else:
        split_metrics = mh.evaluate_two_targets(
            metadata_df=metadata_df,
            y_true=y_true,
            y_pred=y_pred,
            date_col=DATE_COL,
            id_col=STATION_ID_COL,
            target_labels=TARGET_LABELS,
            k_values=(10, 20),
        )
    metric_prefix = 'Val' if metric_split == 'val' else 'Test'

    # Build one result row for this fold/trial.
    result = {
        'feature_set_mode': gat_fold['feature_set_mode'],
        'result_df_name': gat_fold['result_df_name'],
        'text_source': gat_fold['text_source'],
        'fold_id': gat_fold['fold_id'],
        'trial_id': params['trial_id'],
        'metric_split': metric_split,
        'best_epoch': best_epoch,
        'train_log_mse': train_loss_at_best_epoch,
        'train_log_mse_at_best_epoch': train_loss_at_best_epoch,
        'last_epoch_train_log_mse': last_train_loss,
        'val_log_mse': best_val_loss,
        'ct_input_dim': gat_fold['ct_input_dim'],
        'station_input_dim': gat_fold['station_input_dim'],
        'semantic_input_dim': gat_fold['semantic_input_dim'],
        'semantic_hidden_dim': SEMANTIC_HIDDEN_DIM if gat_fold['semantic_input_dim'] > 0 else 0,
    }

    # Add all hyperparameters to the result row.
    for key, value in params.items():
        result[key] = value

    # Prefix the selected split metrics.
    for metric_name, metric_value in split_metrics.items():
        result[f'{metric_prefix}_{metric_name}'] = metric_value

    outputs = [result]

    if return_predictions:
        prediction_df = build_prediction_output_df(metadata_df, y_true, y_pred)
        prediction_df['feature_set_mode'] = gat_fold['feature_set_mode']
        prediction_df['result_df_name'] = gat_fold['result_df_name']
        prediction_df['fold_id'] = gat_fold['fold_id']
        prediction_df['trial_id'] = params['trial_id']
        outputs.append(prediction_df)

    if return_epoch_history:
        outputs.append(epoch_history_df)

    if len(outputs) == 1:
        return result

    return tuple(outputs)



# =========================================================
# 12b. Fast day-batched training and evaluation functions
# =========================================================


def forward_fast_batch(model, batch):
    # Run one prepared fast GAT batch through all active branches.
    return model(
        x_ct=batch['x_ct'],
        x_semantic=batch['x_semantic'],
        edge_index=batch['edge_index'],
        edge_attr=batch['edge_attr'],
        x_station=batch['x_station'],
        station_ct_index=batch['node_index'],
    )


def _evaluate_split_loss_fast(model, tensors, split_name):
    '''Observation-weighted log-scale MSE for one split.'''
    model.eval()
    total_squared_error = torch.zeros((), device=device)
    total_row_count = 0
    num_targets = tensors.y_log.shape[1]

    with torch.no_grad():
        for batch in tensors.batches(split_name):
            pred_log = forward_fast_batch(model, batch)
            mask = batch['mask']
            y_log = tensors.y_log[batch['rows']]
            total_squared_error += F.mse_loss(pred_log[mask], y_log[mask], reduction='sum')
            total_row_count += batch['selected_count']

    if total_row_count == 0:
        return np.nan

    return float(total_squared_error.item() / (total_row_count * num_targets))


def _collect_predictions_fast(model, tensors, split_name):
    '''Collect raw-scale predictions, targets, and metadata for one split.'''
    model.eval()
    pred_parts = []
    true_parts = []
    row_index_parts = []

    with torch.no_grad():
        for batch in tensors.batches(split_name):
            pred_log = forward_fast_batch(model, batch)
            mask = batch['mask']
            pred_parts.append(pred_log[mask])
            true_parts.append(tensors.y_raw[batch['rows']][mask])

            start = batch['rows'].start
            local_row_index = torch.nonzero(mask, as_tuple=False).flatten()
            row_index_parts.append((local_row_index + start).cpu().numpy())

    if len(pred_parts) == 0:
        metadata_df = pd.DataFrame(columns=[DATE_COL, CT_ID_COL, STATION_ID_COL])
        return metadata_df, np.empty((0, 2)), np.empty((0, 2))

    y_pred_log = torch.cat(pred_parts, dim=0).cpu().numpy()
    y_true = torch.cat(true_parts, dim=0).cpu().numpy()
    row_index = np.concatenate(row_index_parts)
    y_pred = mh.inverse_target_transform(y_pred_log, transform='log1p')

    metadata_df = pd.DataFrame({
        DATE_COL: tensors.meta_date[row_index],
        CT_ID_COL: tensors.meta_loc[row_index],
        STATION_ID_COL: tensors.meta_station[row_index],
    })

    return metadata_df, y_true, y_pred


def train_one_gat_fold_fast(
    gat_fold,
    tensors,
    params,
    device,
    metric_split='val',
    return_predictions=False,
    return_epoch_history=False,
):
    '''Train one GAT trial using 30-day station snapshot batches.'''
    if metric_split not in ['val', 'test']:
        raise ValueError("metric_split must be either 'val' or 'test'.")

    mh.set_random_seed(params['random_seed'])

    configured_max_epochs = params['max_epochs']
    configured_patience = params['early_stopping_patience']
    effective_max_epochs = MAX_EPOCHS_OVERRIDE or configured_max_epochs
    effective_patience = PATIENCE_OVERRIDE or configured_patience

    model = StationLevelGAT(
        ct_input_dim=gat_fold['ct_input_dim'],
        station_input_dim=gat_fold['station_input_dim'],
        semantic_input_dim=gat_fold['semantic_input_dim'],
        hidden_dims=params['hidden_dims'],
        activation=params['activation'],
        attention_heads=params['attention_heads'],
        attention_dropout=params['attention_dropout'],
        feature_dropout=params['feature_dropout'],
        input_dropout=params['input_dropout'],
        use_layer_norm=params['use_layer_norm'],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params['learning_rate'],
        weight_decay=params['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=params['scheduler_factor'],
        patience=params['scheduler_patience'],
    )

    best_val_loss = np.inf
    best_state_dict = None
    best_epoch = -1
    epochs_without_improvement = 0
    last_train_loss = np.nan
    train_loss_at_best_epoch = np.nan
    epoch_history_rows = []
    num_targets = tensors.y_log.shape[1]

    for epoch in range(effective_max_epochs):
        model.train()
        epoch_squared_error = torch.zeros((), device=device)
        epoch_row_count = 0

        for batch in tensors.batches('train'):
            pred_log = forward_fast_batch(model, batch)
            mask = batch['mask']
            y_log = tensors.y_log[batch['rows']]
            loss = F.mse_loss(pred_log[mask], y_log[mask])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            with torch.no_grad():
                epoch_squared_error += F.mse_loss(pred_log[mask].detach(), y_log[mask], reduction='sum')
                epoch_row_count += batch['selected_count']

        last_train_loss = (
            float(epoch_squared_error.item() / (epoch_row_count * num_targets))
            if epoch_row_count > 0
            else np.nan
        )

        val_loss = _evaluate_split_loss_fast(model, tensors, 'val')
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {
                key: value.detach().to('cpu', copy=True)
                for key, value in model.state_dict().items()
            }
            best_epoch = epoch
            train_loss_at_best_epoch = last_train_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        epoch_history_rows.append({
            'feature_set_mode': gat_fold['feature_set_mode'],
            'result_df_name': gat_fold['result_df_name'],
            'text_source': gat_fold['text_source'],
            'fold_id': gat_fold['fold_id'],
            'trial_id': params['trial_id'],
            'metric_split': metric_split,
            'epoch': epoch,
            'train_log_mse': last_train_loss,
            'val_log_mse': val_loss,
            'best_val_log_mse_so_far': best_val_loss,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'epochs_without_improvement': epochs_without_improvement,
        })

        if epochs_without_improvement >= effective_patience:
            break

    if best_state_dict is not None:
        model.load_state_dict({key: value.to(device) for key, value in best_state_dict.items()})

    epoch_history_df = pd.DataFrame(epoch_history_rows)
    if len(epoch_history_df) > 0:
        epoch_history_df['best_epoch'] = best_epoch
        epoch_history_df['is_best_epoch'] = epoch_history_df['epoch'].eq(best_epoch)
        epoch_history_df['train_log_mse_at_best_epoch'] = train_loss_at_best_epoch
        epoch_history_df['last_epoch_train_log_mse'] = last_train_loss
        epoch_history_df['day_batch_size'] = tensors.day_batch_size
        epoch_history_df['static_ct_features'] = tensors.static_ct
        epoch_history_df['semantic_input_dim'] = gat_fold['semantic_input_dim']
        epoch_history_df['semantic_hidden_dim'] = SEMANTIC_HIDDEN_DIM if gat_fold['semantic_input_dim'] > 0 else 0
        epoch_history_df['configured_max_epochs'] = configured_max_epochs
        epoch_history_df['effective_max_epochs'] = effective_max_epochs
        epoch_history_df['configured_early_stopping_patience'] = configured_patience
        epoch_history_df['effective_early_stopping_patience'] = effective_patience
        for key, value in params.items():
            if isinstance(value, (tuple, list, dict, set)):
                epoch_history_df[key] = [value] * len(epoch_history_df)
            else:
                epoch_history_df[key] = value

    metadata_df, y_true, y_pred = _collect_predictions_fast(model, tensors, metric_split)
    if len(metadata_df) == 0:
        split_metrics = {}
    else:
        split_metrics = mh.evaluate_two_targets(
            metadata_df=metadata_df,
            y_true=y_true,
            y_pred=y_pred,
            date_col=DATE_COL,
            id_col=STATION_ID_COL,
            target_labels=TARGET_LABELS,
            k_values=(10, 20),
        )
    metric_prefix = 'Val' if metric_split == 'val' else 'Test'

    result = {
        'feature_set_mode': gat_fold['feature_set_mode'],
        'result_df_name': gat_fold['result_df_name'],
        'text_source': gat_fold['text_source'],
        'fold_id': gat_fold['fold_id'],
        'trial_id': params['trial_id'],
        'metric_split': metric_split,
        'best_epoch': best_epoch,
        'train_log_mse': train_loss_at_best_epoch,
        'train_log_mse_at_best_epoch': train_loss_at_best_epoch,
        'last_epoch_train_log_mse': last_train_loss,
        'val_log_mse': best_val_loss,
        'ct_input_dim': gat_fold['ct_input_dim'],
        'station_input_dim': gat_fold['station_input_dim'],
        'semantic_input_dim': gat_fold['semantic_input_dim'],
        'semantic_hidden_dim': SEMANTIC_HIDDEN_DIM if gat_fold['semantic_input_dim'] > 0 else 0,
        'day_batch_size': tensors.day_batch_size,
        'static_ct_features': tensors.static_ct,
        'configured_max_epochs': configured_max_epochs,
        'effective_max_epochs': effective_max_epochs,
        'configured_early_stopping_patience': configured_patience,
        'effective_early_stopping_patience': effective_patience,
    }

    for key, value in params.items():
        result[key] = value

    for metric_name, metric_value in split_metrics.items():
        result[f'{metric_prefix}_{metric_name}'] = metric_value

    outputs = [result]
    if return_predictions:
        prediction_df = build_prediction_output_df(metadata_df, y_true, y_pred)
        prediction_df['feature_set_mode'] = gat_fold['feature_set_mode']
        prediction_df['result_df_name'] = gat_fold['result_df_name']
        prediction_df['fold_id'] = gat_fold['fold_id']
        prediction_df['trial_id'] = params['trial_id']
        outputs.append(prediction_df)
    if return_epoch_history:
        outputs.append(epoch_history_df)

    if len(outputs) == 1:
        return result

    return tuple(outputs)

# Test call for this cell.
print('Training functions are ready.')
print('Main comparison metric:', COMPARISON_METRIC)
print('Search phase evaluates validation metrics only; final phase evaluates test metrics only.')

# %% [cell 14: code]
# =========================================================
# 13. Main experiment loop
# =========================================================


def summarize_trial_results(fold_results_df):
    '''Aggregate fold-level rows into trial-level mean/std rows.'''

    # Identify metric columns that should be averaged across folds.
    metric_cols = [
        col for col in fold_results_df.columns
        if (col.startswith('Val_') or col.startswith('Test_') or col in ['train_log_mse', 'val_log_mse'])
        and pd.api.types.is_numeric_dtype(fold_results_df[col])
    ]

    # Store trial summaries.
    summaries = []

    # Group by feature setting and trial.
    group_cols = ['feature_set_mode', 'trial_id']
    for (feature_set_mode, trial_id), trial_df in fold_results_df.groupby(group_cols, sort=False):
        # Reuse helper aggregation logic for mean and standard deviation.
        summary = mh.aggregate_fold_metrics(trial_df, metric_cols=metric_cols)

        # Add identifying columns and hyperparameters from the first fold row.
        first_row = trial_df.iloc[0].to_dict()
        summary['feature_set_mode'] = feature_set_mode
        summary['result_df_name'] = first_row['result_df_name']
        summary['trial_id'] = trial_id
        summary['text_source'] = first_row['text_source']
        summary['metric_split'] = first_row.get('metric_split')
        summary['num_folds'] = len(trial_df)

        # Keep the hyperparameter columns in the summary.
        hyperparameter_cols = [
            'hidden_dims',
            'activation',
            'feature_dropout',
            'learning_rate',
            'weight_decay',
            'attention_heads',
            'attention_dropout',
            'input_dropout',
            'optimizer_name',
            'use_layer_norm',
            'loss_name',
            'max_epochs',
            'early_stopping_patience',
            'scheduler_name',
            'scheduler_patience',
            'scheduler_factor',
            'random_seed',
            'trial_group',
            'day_batch_size',
            'static_ct_features',
            'semantic_input_dim',
            'semantic_hidden_dim',
            'configured_max_epochs',
            'effective_max_epochs',
            'configured_early_stopping_patience',
            'effective_early_stopping_patience',
        ]
        for col in hyperparameter_cols:
            summary[col] = first_row.get(col)

        summaries.append(summary)

    # Convert summaries into a dataframe.
    summary_df = pd.DataFrame(summaries)

    # Sort validation-search summaries by validation Avg_MSE when it exists.
    comparison_col = f'{COMPARISON_METRIC}_mean'
    if comparison_col in summary_df.columns:
        summary_df = summary_df.sort_values(comparison_col, ascending=True).reset_index(drop=True)

    return summary_df


def get_trial_params_by_id(trials, trial_id):
    '''Return one hyperparameter dictionary by trial id.'''

    # Trial ids are added when the grid is created.
    for trial in trials:
        if int(trial['trial_id']) == int(trial_id):
            return trial

    raise ValueError(f'No trial found for trial_id={trial_id}.')



def _checkpoint(df, name):
    '''Write a partial result table immediately after each completed fold.'''
    df.to_csv(CHECKPOINT_DIR / name, index=False)


def run_one_feature_search(feature_set_mode):
    '''Run validation-only hyperparameter search for one feature setting.'''

    _flush('\n' + '=' * 70)
    _flush('Validation search feature setting:', feature_set_mode)
    _flush('=' * 70)

    feature_data = build_feature_set_objects(feature_set_mode)
    _flush('Text source:', feature_data['text_source'])
    _flush('Station base rows:', feature_data['station_base_rows'].shape)
    _flush('CT base rows:', feature_data['ct_base_rows'].shape)
    _flush('Raw text embedding columns:', len(feature_data['text_embedding_cols']))

    validation_fold_rows = []
    validation_epoch_dfs = []

    for fold_number, fold in enumerate(active_folds, start=1):
        fold_start = time.time()
        _flush(f'\nFold {fold_number} of {len(active_folds)} | fold_id: {fold["fold_id"]}')

        gat_fold = prepare_hierarchical_gat_fold(feature_data, fold)
        tensors = build_fold_tensors(gat_fold)
        ct_dim = gat_fold['ct_input_dim']
        station_dim = gat_fold['station_input_dim']
        semantic_dim = gat_fold['semantic_input_dim']
        semantic_hidden = SEMANTIC_HIDDEN_DIM if semantic_dim > 0 else 0
        _flush(
            f'  days={tensors.num_days} station_rows={len(tensors.meta_date)} '
            f'ct_dim={ct_dim} station_dim={station_dim} '
            f'semantic_dim={semantic_dim} semantic_hidden={semantic_hidden} '
            f'static_ct={tensors.static_ct} day_batch={tensors.day_batch_size} '
            f'prep={time.time() - fold_start:.1f}s'
        )

        for trial_number, params in enumerate(gat_trials, start=1):
            trial_start = time.time()
            fold_result, fold_epoch_df = train_one_gat_fold_fast(
                gat_fold=gat_fold,
                tensors=tensors,
                params=params,
                device=device,
                metric_split='val',
                return_epoch_history=True,
            )
            fold_epoch_df['phase'] = 'validation_search'
            validation_fold_rows.append(fold_result)
            validation_epoch_dfs.append(fold_epoch_df)

            _flush(
                f'  trial {trial_number}/{len(gat_trials)} '
                f'(id {params["trial_id"]}) '
                f'{COMPARISON_METRIC}={fold_result.get(COMPARISON_METRIC, float("nan")):.5f} '
                f'best_epoch={fold_result["best_epoch"]} '
                f'{time.time() - trial_start:.1f}s'
            )

        _checkpoint(
            pd.DataFrame(validation_fold_rows),
            f'{feature_set_mode}_validation_folds_partial.csv',
        )

        del gat_fold, tensors
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        _flush(f'  fold total {time.time() - fold_start:.1f}s')

    validation_fold_df = pd.DataFrame(validation_fold_rows)
    validation_fold_df['phase'] = 'validation_search'
    validation_epoch_df = pd.concat(validation_epoch_dfs, axis=0, ignore_index=True)
    validation_summary_df = summarize_trial_results(validation_fold_df)

    comparison_col = f'{COMPARISON_METRIC}_mean'
    best_summary_row = validation_summary_df.sort_values(comparison_col, ascending=True).iloc[0]
    best_trial_id = int(best_summary_row['trial_id'])
    best_params = get_trial_params_by_id(gat_trials, best_trial_id)

    _flush('\nSelected best validation trial for', feature_set_mode)
    _flush('Best trial_id:', best_trial_id)
    _flush(comparison_col + ':', best_summary_row[comparison_col])

    return feature_data, validation_fold_df, validation_summary_df, validation_epoch_df, best_params


def run_one_feature_final_test(feature_data, best_params):
    '''Rerun the validation-selected configuration and evaluate test metrics only.'''

    feature_set_mode = feature_data['feature_set_mode']

    _flush('\n' + '-' * 70)
    _flush('Final test evaluation feature setting:', feature_set_mode)
    _flush('Using validation-selected trial_id:', best_params['trial_id'])
    _flush('-' * 70)

    final_test_fold_rows = []
    final_test_prediction_rows = []
    final_test_epoch_dfs = []

    for fold_number, fold in enumerate(active_folds, start=1):
        fold_start = time.time()
        _flush(f'  Final test fold {fold_number} of {len(active_folds)} | fold_id: {fold["fold_id"]}')

        gat_fold = prepare_hierarchical_gat_fold(feature_data, fold)
        tensors = build_fold_tensors(gat_fold)
        ct_dim = gat_fold['ct_input_dim']
        station_dim = gat_fold['station_input_dim']
        semantic_dim = gat_fold['semantic_input_dim']
        semantic_hidden = SEMANTIC_HIDDEN_DIM if semantic_dim > 0 else 0
        _flush(
            f'    ct_dim={ct_dim} station_dim={station_dim} '
            f'semantic_dim={semantic_dim} semantic_hidden={semantic_hidden} '
            f'day_batch={tensors.day_batch_size}'
        )

        fold_result, fold_prediction_df, fold_epoch_df = train_one_gat_fold_fast(
            gat_fold=gat_fold,
            tensors=tensors,
            params=best_params,
            device=device,
            metric_split='test',
            return_predictions=True,
            return_epoch_history=True,
        )

        fold_epoch_df['phase'] = 'final_test'
        final_test_fold_rows.append(fold_result)
        final_test_prediction_rows.append(fold_prediction_df)
        final_test_epoch_dfs.append(fold_epoch_df)

        _flush(
            f'    Test_Avg_MSE={fold_result.get("Test_Avg_MSE", float("nan")):.5f} '
            f'{time.time() - fold_start:.1f}s'
        )

        _checkpoint(
            pd.DataFrame(final_test_fold_rows),
            f'{feature_set_mode}_final_test_folds_partial.csv',
        )
        _checkpoint(
            pd.concat(final_test_prediction_rows, axis=0, ignore_index=True),
            f'{feature_set_mode}_final_test_predictions_partial.csv',
        )

        del gat_fold, tensors
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    final_test_fold_df = pd.DataFrame(final_test_fold_rows)
    final_test_fold_df['phase'] = 'final_test'
    final_test_summary_df = summarize_trial_results(final_test_fold_df)
    final_test_prediction_df = pd.concat(final_test_prediction_rows, axis=0, ignore_index=True)
    final_test_epoch_df = pd.concat(final_test_epoch_dfs, axis=0, ignore_index=True)

    return final_test_fold_df, final_test_summary_df, final_test_prediction_df, final_test_epoch_df


RUN_TIMING_DIAGNOSTIC = os.environ.get('GAT_DIAGNOSTIC', '0') == '1'

if RUN_TIMING_DIAGNOSTIC:
    _flush('=== GAT timing diagnostic ===')
    diagnostic_mode = ACTIVE_FEATURE_SET_MODES[0]

    t0 = time.time()
    diagnostic_feature_data = build_feature_set_objects(diagnostic_mode)
    _flush(f'build_feature_set_objects: {time.time() - t0:.1f}s')

    t0 = time.time()
    diagnostic_fold = prepare_hierarchical_gat_fold(diagnostic_feature_data, active_folds[0])
    prep_seconds = time.time() - t0
    _flush(f'prepare_hierarchical_gat_fold: {prep_seconds:.1f}s')

    t0 = time.time()
    diagnostic_tensors = build_fold_tensors(diagnostic_fold)
    _flush(f'build_fold_tensors: {time.time() - t0:.1f}s')
    _flush(
        f'days={diagnostic_tensors.num_days} nodes={diagnostic_tensors.num_nodes} '
        f'station_rows={len(diagnostic_tensors.meta_date)}'
    )
    _flush(
        f'ct_input_dim={diagnostic_fold["ct_input_dim"]} '
        f'station_input_dim={diagnostic_fold["station_input_dim"]}'
    )
    _flush(f'STATIC CT FEATURES: {diagnostic_tensors.static_ct}')
    _flush(
        f'day_batch_size={diagnostic_tensors.day_batch_size} '
        f'train batches/epoch={sum(1 for count in diagnostic_tensors.batch_counts["train"] if count > 0)}'
    )

    diagnostic_params = dict(gat_trials[0])
    diagnostic_params['max_epochs'] = 5
    diagnostic_params['early_stopping_patience'] = 999

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    _ = train_one_gat_fold_fast(
        gat_fold=diagnostic_fold,
        tensors=diagnostic_tensors,
        params=diagnostic_params,
        device=device,
        metric_split='val',
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    seconds_per_epoch = (time.time() - t0) / 5
    _flush(f'seconds per epoch: {seconds_per_epoch:.3f}')

    n_runs = len(gat_trials) * len(active_folds) + len(active_folds)
    for assumed_epochs in (60, 120, 200):
        train_hours = n_runs * assumed_epochs * seconds_per_epoch / 3600
        prep_hours = 2 * len(active_folds) * prep_seconds / 3600
        _flush(
            f'  at ~{assumed_epochs} epochs/run: '
            f'{train_hours + prep_hours:.2f} h per feature arm'
        )
    _flush('=== end diagnostic ===')
    sys.exit(0)


def build_ablation_results_df(gat_trials_df):
    '''Compare final test performance between baseline and semantic GAT variants.'''

    # The ablation table is built from final test summaries only.
    # Negative deltas on error metrics mean the semantic variant reduced error.
    summary_by_mode = gat_trials_df.set_index('feature_set_mode')
    metric_cols = [
        col for col in gat_trials_df.columns
        if col.startswith('Test_')
        and col.endswith('_mean')
        and pd.api.types.is_numeric_dtype(gat_trials_df[col])
    ]

    comparisons = [
        ('graph_minilm_vs_graph_baseline', 'graph_baseline', 'graph_minilm'),
        ('graph_gpt_vs_graph_baseline', 'graph_baseline', 'graph_gpt'),
        ('graph_gpt_small_vs_graph_baseline', 'graph_baseline', 'graph_gpt_small'),
        ('graph_gpt_vs_graph_minilm', 'graph_minilm', 'graph_gpt'),
        ('graph_gpt_small_vs_graph_minilm', 'graph_minilm', 'graph_gpt_small'),
        ('graph_gpt_vs_graph_gpt_small', 'graph_gpt_small', 'graph_gpt'),
    ]

    ablation_rows = []
    for comparison_name, reference_mode, candidate_mode in comparisons:
        if reference_mode not in summary_by_mode.index or candidate_mode not in summary_by_mode.index:
            continue

        row = {
            'comparison': comparison_name,
            'reference_feature_set_mode': reference_mode,
            'candidate_feature_set_mode': candidate_mode,
            'reference_trial_id': summary_by_mode.loc[reference_mode, 'trial_id'],
            'candidate_trial_id': summary_by_mode.loc[candidate_mode, 'trial_id'],
        }

        for metric_col in metric_cols:
            reference_value = summary_by_mode.loc[reference_mode, metric_col]
            candidate_value = summary_by_mode.loc[candidate_mode, metric_col]
            row[f'reference_{metric_col}'] = reference_value
            row[f'candidate_{metric_col}'] = candidate_value
            row[f'delta_{metric_col}'] = candidate_value - reference_value
            row[f'pct_change_{metric_col}'] = ((candidate_value - reference_value) / reference_value) * 100

        ablation_rows.append(row)

    return pd.DataFrame(ablation_rows)


def try_write_parallel_ablation_results():
    '''Write ablation results when feature settings were run as separate processes.'''

    # In parallel mode each process owns one feature setting through GAT_FEATURE_MODE.
    # The per-feature trials filenames are unique, so no process overwrites another.
    # Once all four trials files exist, this function combines them into the same
    # ablation table produced by the single-process all-feature run.
    trial_paths = []
    for feature_set_mode in FEATURE_SET_MODES:
        result_df_name = RESULT_DF_NAME_BY_FEATURE_SET[feature_set_mode]
        trial_path = RESULTS_DIR / f'gat_{result_df_name}_{RUN_MODE}_trials.csv'
        if not trial_path.exists():
            return pd.DataFrame()
        trial_paths.append(trial_path)

    parallel_trials_df = pd.concat(
        [pd.read_csv(trial_path) for trial_path in trial_paths],
        axis=0,
        ignore_index=True,
    )

    if parallel_trials_df['feature_set_mode'].nunique() != len(FEATURE_SET_MODES):
        return pd.DataFrame()

    ablation_df_name = 'graph_baseline_graph_minilm_graph_gpt_graph_gpt_small'
    parallel_ablation_df = build_ablation_results_df(parallel_trials_df)
    output_path = RESULTS_DIR / f'gat_{ablation_df_name}_{RUN_MODE}_ablation_results.csv'
    parallel_ablation_df.to_csv(output_path, index=False)
    _flush('Parallel ablation results saved:', output_path)

    return parallel_ablation_df

def main():
    '''Run validation-only hyperparameter search, then final unbiased test evaluation.'''

    # Limit CPU worker threads so parallel feature-setting processes do not each claim all cores.
    torch.set_num_threads(TORCH_NUM_THREADS)

    # Print run settings before training begins.
    print('Starting station-level GAT experiment')
    print('RUN_MODE:', RUN_MODE)
    print('GAT_FEATURE_MODE:', GAT_FEATURE_MODE)
    print('Active feature settings:', ACTIVE_FEATURE_SET_MODES)
    print('N_TEXT_PCA_COMPONENTS:', N_TEXT_PCA_COMPONENTS)
    print('SEMANTIC_HIDDEN_DIM:', SEMANTIC_HIDDEN_DIM)
    print('DAY_BATCH_SIZE:', DAY_BATCH_SIZE)
    print('TORCH_NUM_THREADS:', TORCH_NUM_THREADS)
    print('Results directory:', RESULTS_DIR)
    print('Number of feature settings:', len(ACTIVE_FEATURE_SET_MODES))
    print('Number of trials per feature setting:', len(gat_trials))
    print('Number of active folds:', len(active_folds))
    print('Device:', device)
    print('Workflow: validation selects hyperparameters; test evaluates selected model only.')

    # Store all feature-setting results.
    all_validation_fold_dfs = []
    all_validation_summary_dfs = []
    all_final_test_fold_dfs = []
    all_final_test_summary_dfs = []
    all_final_test_prediction_dfs = []
    all_epoch_dfs = []

    # Loop over the requested feature settings.
    for feature_set_mode in ACTIVE_FEATURE_SET_MODES:
        # Phase 1: validation-only search.
        feature_data, validation_fold_df, validation_summary_df, validation_epoch_df, best_params = run_one_feature_search(feature_set_mode)

        # Phase 2: final test evaluation using only the validation-selected hyperparameters.
        final_test_fold_df, final_test_summary_df, final_test_prediction_df, final_test_epoch_df = run_one_feature_final_test(feature_data, best_params)

        # Store results for this feature setting.
        all_validation_fold_dfs.append(validation_fold_df)
        all_validation_summary_dfs.append(validation_summary_df)
        all_final_test_fold_dfs.append(final_test_fold_df)
        all_final_test_summary_dfs.append(final_test_summary_df)
        all_final_test_prediction_dfs.append(final_test_prediction_df)
        all_epoch_dfs.append(validation_epoch_df)
        all_epoch_dfs.append(final_test_epoch_df)

        # Clear feature-level cached rows before the next feature setting.
        del feature_data, validation_fold_df, validation_summary_df, validation_epoch_df, final_test_fold_df, final_test_summary_df, final_test_prediction_df, final_test_epoch_df
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Combine results across feature settings.
    validation_fold_results_df = pd.concat(all_validation_fold_dfs, axis=0, ignore_index=True)
    validation_summary_df = pd.concat(all_validation_summary_dfs, axis=0, ignore_index=True)
    final_test_fold_results_df = pd.concat(all_final_test_fold_dfs, axis=0, ignore_index=True)
    final_test_summary_df = pd.concat(all_final_test_summary_dfs, axis=0, ignore_index=True)
    final_test_predictions_df = pd.concat(all_final_test_prediction_dfs, axis=0, ignore_index=True)
    gat_epochs_df = pd.concat(all_epoch_dfs, axis=0, ignore_index=True)

    # gat_grid.csv stores the validation search space and validation-selected performance.
    gat_grid_df = validation_summary_df.copy()
    gat_grid_df['validation_rank'] = (
        gat_grid_df
        .groupby('feature_set_mode')[f'{COMPARISON_METRIC}_mean']
        .rank(method='dense', ascending=True)
        .astype(int)
    )

    # gat_trials.csv stores the final validation-selected trial performance per feature setting.
    gat_trials_results_df = final_test_summary_df.copy()
    gat_trials_results_df['test_rank'] = gat_trials_results_df['Test_Avg_MSE_mean'].rank(method='dense', ascending=True).astype(int)
    gat_trials_results_df = gat_trials_results_df.sort_values('test_rank', ascending=True).reset_index(drop=True)

    # gat_fold.csv combines fold-level validation search rows and final test rows.
    gat_fold_df = pd.concat(
        [
            validation_fold_results_df,
            final_test_fold_results_df,
        ],
        axis=0,
        ignore_index=True,
        sort=False,
    )

    # Keep raw-scale predictions with internal identifiers for per-feature-setting filtering.
    gat_predictions_df = final_test_predictions_df.reset_index(drop=True)

    # Prediction export columns match the agreed station-level error-analysis format.
    prediction_export_cols = [
        'fold_id',
        DATE_COL,
        CT_ID_COL,
        STATION_ID_COL,
        'actual_inflow',
        'predicted_inflow',
        'actual_outflow',
        'predicted_outflow',
    ]

    # gat_ablation_results.csv compares baseline, MiniLM, GPT-large, and GPT-small final test performance.
    # It is created only when all four feature settings run in the same process.
    # When the four settings run in parallel, build ablation afterwards by reading the four trials files.
    if len(ACTIVE_FEATURE_SET_MODES) == len(FEATURE_SET_MODES):
        gat_ablation_results_df = build_ablation_results_df(gat_trials_results_df)
    else:
        gat_ablation_results_df = pd.DataFrame()

    # Save the six GAT result artifact types in the desc_semantics/gat folder.
    # Each filename includes the dataframe/source name and RUN_MODE so files do not overwrite each other.
    for feature_set_mode in ACTIVE_FEATURE_SET_MODES:
        result_df_name = RESULT_DF_NAME_BY_FEATURE_SET[feature_set_mode]
        df_epochs = gat_epochs_df.loc[gat_epochs_df['result_df_name'].eq(result_df_name)].reset_index(drop=True)
        df_fold = gat_fold_df.loc[gat_fold_df['result_df_name'].eq(result_df_name)].reset_index(drop=True)
        df_grid = gat_grid_df.loc[gat_grid_df['result_df_name'].eq(result_df_name)].reset_index(drop=True)
        df_trials = gat_trials_results_df.loc[gat_trials_results_df['result_df_name'].eq(result_df_name)].reset_index(drop=True)
        df_predictions = (
            gat_predictions_df
            .loc[gat_predictions_df['result_df_name'].eq(result_df_name), prediction_export_cols]
            .reset_index(drop=True)
        )

        df_epochs.to_csv(RESULTS_DIR / f'gat_{result_df_name}_{RUN_MODE}_epochs.csv', index=False)
        df_fold.to_csv(RESULTS_DIR / f'gat_{result_df_name}_{RUN_MODE}_fold.csv', index=False)
        df_grid.to_csv(RESULTS_DIR / f'gat_{result_df_name}_{RUN_MODE}_grid.csv', index=False)
        df_trials.to_csv(RESULTS_DIR / f'gat_{result_df_name}_{RUN_MODE}_trials.csv', index=False)
        df_predictions.to_csv(RESULTS_DIR / f'gat_{result_df_name}_{RUN_MODE}_predictions.csv', index=False)

    if len(ACTIVE_FEATURE_SET_MODES) == len(FEATURE_SET_MODES):
        ablation_df_name = 'graph_baseline_graph_minilm_graph_gpt_graph_gpt_small'
        gat_ablation_results_df.to_csv(
            RESULTS_DIR / f'gat_{ablation_df_name}_{RUN_MODE}_ablation_results.csv',
            index=False,
        )
    else:
        parallel_ablation_df = try_write_parallel_ablation_results()
        if len(parallel_ablation_df) > 0:
            gat_ablation_results_df = parallel_ablation_df

    print('\nBest validation configurations:')
    display(gat_grid_df.sort_values(f'{COMPARISON_METRIC}_mean', ascending=True).head(10))

    print('\nFinal test summaries for validation-selected configurations:')
    display(gat_trials_results_df)

    if len(ACTIVE_FEATURE_SET_MODES) == len(FEATURE_SET_MODES):
        print('\nAblation results:')
        display(gat_ablation_results_df)

    return gat_grid_df, gat_trials_results_df, gat_fold_df, gat_epochs_df, gat_predictions_df, gat_ablation_results_df


# Test call for this cell.
print('Main loop is ready.')
print('To execute training, run the next cell.')
print('This main loop does validation-only tuning first, then test-only final evaluation.')

# %% [cell 15: code]
# =========================================================
# 14. Main call
# =========================================================

# This is the only cell that launches model training.
# The call order inside main() is:
# 1. loop over feature settings: graph_baseline, graph_minilm, graph_gpt, graph_gpt_small
# 2. validation phase: loop over all GAT hyperparameter trials
# 3. validation phase: train on train_mask and select by validation metrics only
# 4. choose the best trial per feature setting using validation Avg_MSE_mean
# 5. final phase: rerun only the selected trial per feature setting
# 6. final phase: evaluate test metrics only after hyperparameters are fixed
# 7. save the six requested GAT result artifacts:
#    gat_epochs.csv, gat_fold.csv, gat_grid.csv, gat_trials.csv, gat_predictions.csv, gat_ablation_results.csv

if __name__ == '__main__':
    gat_grid_df, gat_trials_results_df, gat_fold_df, gat_epochs_df, gat_predictions_df, gat_ablation_results_df = main()
