# Toronto Bike-Share Demand Modelling

Station-level prediction of Toronto bike-share inflow and outflow using urban
context, network structure, and Wikidata-derived semantic features.

The project compares three model families:

- CatBoost regression
- Multi-layer perceptrons (MLP)
- Graph attention networks (GAT)

Each model predicts two targets:

1. `inflow_count`
2. `outflow_count`

Model selection uses spatial cross-validation. Hyperparameters are selected from
validation results, and the held-out test split is evaluated only after the
winning configuration has been fixed.

## Research design

The modelling unit is a station-date observation. Experiments compare baseline
urban and temporal predictors with two forms of contextual information:

- **Graph features:** Node2Vec and census-tract network attributes.
- **Semantic features:** Wikidata category text or place-description embeddings
  produced by MiniLM and OpenAI embedding models.

The principal feature ablations are:

| Index | Feature set |
| ---: | --- |
| 0 | Baseline features plus semantic embeddings, excluding Node2Vec |
| 1 | Baseline features plus Node2Vec, excluding semantic embeddings |
| 2 | Baseline features without semantic embeddings or Node2Vec |

GAT experiments use four graph configurations:

| Mode | Description |
| --- | --- |
| `graph_baseline` | Graph and station features without text embeddings |
| `graph_minilm` | Baseline graph with MiniLM embeddings |
| `graph_gpt` | Baseline graph with GPT large embeddings |
| `graph_gpt_small` | Baseline graph with GPT small embeddings |

Text PCA is fitted on training census tracts within each fold before being
applied to validation and test data.

## Repository structure

```text
.
├── pyproject.toml
├── uv.lock
└── bikeshare
    ├── data
    │   ├── raw
    │   │   ├── bike_ridership
    │   │   └── graph
    │   └── processed
    │       ├── bike_ridership
    │       ├── graph
    │       ├── model_df
    │       └── statistics
    ├── notebooks
    │   └── final_notebook
    ├── results
    │   ├── cat_semantics
    │   ├── desc_semantics
    │   └── semantic_embedings_test
    └── scripts
        ├── cat_semantics
        ├── desc_semantics
        ├── semantic_embedings_test
        └── model_helper_stations_lvl.py
```

`semantic_embedings_test` retains the historical directory spelling used by
the experiment scripts and result paths.

## Environment

### Requirements

- Linux or WSL
- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- [Git LFS](https://git-lfs.com/)
- Sufficient storage for the approximately 4.7 GB project dataset
- NVIDIA GPU for CatBoost experiments
- CUDA-capable GPU recommended for MLP and GAT experiments

PyTorch is pinned to version 2.8.0 and configured for CUDA 12.6 wheels. MLP and
GAT scripts can fall back to CPU, although full searches may be prohibitively
slow. CatBoost configurations explicitly use GPU device 0.

### Installation

Run from the repository root:

```bash
cd /home/najla/dev/najla-msc
git lfs install
git lfs pull
uv sync --locked --dev
```

Check that the lockfile matches the project configuration:

```bash
uv lock --check
```

Several scripts currently set the project path to
`/home/najla/dev/najla-msc/bikeshare`. Keep the repository at that location or
update the `PROJECT_ROOT` and `REPO_ROOT` constants before running it
elsewhere.

## Data

### Raw data

`bikeshare/data/raw/bike_ridership` contains:

- Monthly and annual Toronto Bike Share ridership extracts for 2022–2026
- Station metadata and coordinate matching files
- Toronto holiday, weekend, and weather features
- Visitor-attraction point data

`bikeshare/data/raw/graph` contains station-flow and duration data used by the
graph workflow.

### Processed data

| Directory | Contents |
| --- | --- |
| `processed/bike_ridership` | Station-level trips, inflow, outflow, static features, dynamic features, and Wikidata joins |
| `processed/graph` | Graph topology, location geometries, semantic tokens, and embedding tables |
| `processed/model_df` | Model-ready baseline, category-semantic, and description-semantic datasets |
| `processed/statistics` | Wikidata entity and place-article reference tables |

The following files are canonical source artifacts:

- `bikeshare/data/processed/graph/nodes.parquet`
- `bikeshare/data/processed/graph/edges.parquet`
- `bikeshare/data/processed/graph/locations_gdf.parquet`

They are intentionally treated as standalone data. Their original graph-building
scripts are not part of this repository.

## Notebook workflow

Notebooks are stored in `bikeshare/notebooks/final_notebook`.

| Notebook | Purpose |
| --- | --- |
| `bike_riders_explore_jupyter_12pm.ipynb` | Explore and consolidate ridership records |
| `bike_base_table_stations_12pm.ipynb` | Build station-level bike-share tables and graph joins |
| `wikidata_semantic_categories.ipynb` | Group Wikidata type labels into semantic categories |
| `text_semantic.ipynb` | Prepare category-semantic text and embeddings |
| `wiki_desc_prep.ipynb` | Prepare Wikidata place descriptions and embedding inputs |
| `desc_minillm_embedings.ipynb` | Generate description-based MiniLM representations |
| `cat_graph_dfs.ipynb` | Build category-semantic graph model tables |
| `desc_graph_dfs.ipynb` | Build description-semantic graph model tables |
| `create_folds_and_final_df.ipynb` | Create category-semantic model data and frozen spatial folds |
| `desc_folds_and_final_df.ipynb` | Create description-semantic model data and folds |
| `graph_explore.ipynb` | Inspect the cached Toronto graph |
| `gat_explore.ipynb` | Develop and inspect the station-level GAT workflow |

Start JupyterLab with:

```bash
uv run jupyter lab bikeshare/notebooks/final_notebook
```

## Running experiments

Commands below assume a Bash or WSL shell at the repository root.

### Smoke tests

Smoke tests use a reduced search or a single spatial fold. They validate the
pipeline and are not final experimental results.

```bash
uv run python bikeshare/scripts/cat_semantics/catboost_smoke_test.py
uv run python bikeshare/scripts/cat_semantics/mlp_smoke_test.py
uv run python bikeshare/scripts/desc_semantics/gat_smoke_test.py
```

The description-semantic GAT smoke test defaults to three epochs. Override the
limit when required:

```bash
GAT_SMOKE_MAX_EPOCHS=5 GAT_SMOKE_PATIENCE=2 \
  uv run python bikeshare/scripts/desc_semantics/gat_smoke_test.py
```

### CatBoost

Run one category-semantic ablation with PCA32 embeddings:

```bash
CATBOOST_DF_INDX=0 \
CATBOOST_WIKI_MODE=pca32 \
CATBOOST_SEARCH_NAME=full_grid \
  uv run python bikeshare/scripts/cat_semantics/catboost.py
```

Run the description-semantic experiment:

```bash
CATBOOST_DF_INDX=0 \
CATBOOST_WIKI_MODE=pca32 \
CATBOOST_SEARCH_NAME=full_grid \
  uv run python bikeshare/scripts/desc_semantics/catboost.py
```

Set `CATBOOST_SEARCH_NAME=parameter_sampler` and `CATBOOST_N_ITER=<n>` for a
reduced random search.

### MLP

Run one category-semantic ablation:

```bash
MLP_DF_INDX=0 \
MLP_WIKI_MODE=pca32 \
MLP_SEARCH_NAME=full_grid \
  uv run python bikeshare/scripts/cat_semantics/mlp.py
```

Run the description-semantic experiment:

```bash
MLP_DF_INDX=0 \
MLP_WIKI_MODE=pca32 \
MLP_SEARCH_NAME=full_grid \
  uv run python bikeshare/scripts/desc_semantics/mlp.py
```

Description-semantic MLP runs support `MLP_TORCH_THREADS`, `MLP_N_JOBS`, and
`MLP_PARALLEL_START_METHOD` for process and thread control.

### GAT

Run one category-semantic feature mode:

```bash
GAT_FEATURE_MODE=graph_minilm \
  uv run python bikeshare/scripts/cat_semantics/gat.py
```

Run one description-semantic feature mode:

```bash
GAT_FEATURE_MODE=graph_minilm \
GAT_DAY_BATCH=30 \
GAT_TORCH_THREADS=2 \
  uv run python bikeshare/scripts/desc_semantics/gat.py
```

Omit `GAT_FEATURE_MODE` to run all four feature modes in sequence. Supported
description-semantic controls include `GAT_MAX_EPOCHS`, `GAT_PATIENCE`, and
`GAT_DIAGNOSTIC=1`.

## Experiment controls

| Variable | Accepted values | Effect |
| --- | --- | --- |
| `CATBOOST_SEARCH_NAME` | `smoke_test`, `full_grid`, `parameter_sampler` | Selects the CatBoost search profile |
| `CATBOOST_N_ITER` | Positive integer | Sets the random-search iteration count |
| `CATBOOST_DF_INDX` | `0`, `1`, `2` | Runs one CatBoost feature ablation |
| `CATBOOST_WIKI_MODE` | `raw`, `pca`, `pca16`, `pca32`, … | Selects semantic embedding reduction |
| `MLP_SEARCH_NAME` | `smoke_test`, `full_grid` | Selects the MLP search profile |
| `MLP_DF_INDX` | `0`, `1`, `2` | Runs one MLP feature ablation |
| `MLP_WIKI_MODE` | `raw`, `pca16`, `pca32`, `pca64` | Selects semantic embedding reduction |
| `GAT_FEATURE_MODE` | `graph_baseline`, `graph_minilm`, `graph_gpt`, `graph_gpt_small` | Runs one GAT feature mode |

## Results

Experiment outputs are written beneath:

- `bikeshare/results/cat_semantics`
- `bikeshare/results/desc_semantics`
- `bikeshare/results/semantic_embedings_test`

Depending on the model, generated CSV files include:

- Hyperparameter grids and trial summaries
- Fold-level validation and test metrics
- Epoch histories
- Row-level predictions
- Feature importance
- Semantic-feature ablation comparisons

Reported regression metrics are MAE, MSE, RMSE, R², and RMSLE. Ranking metrics
include Recall@10, NDCG@10, Recall@20, and NDCG@20. Metrics are recorded for
inflow, outflow, and their average.

## Validation and maintenance

Run the non-training checks before committing changes:

```bash
uv lock --check
uv run ruff check bikeshare/scripts
```

Full-grid runs are computationally expensive and write directly into the
corresponding results directory. Preserve completed result files before
re-running an experiment with the same configuration.
