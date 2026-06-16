"""
Top-level experiment runners: model builders, baseline evaluation, and result saving.

This module is the main orchestration layer for the comparison experiment.  It
provides:

  - Builder functions (build_gat, build_gps, build_mlp, build_full_mlp): each
    constructs a specific neural-network model with the correct input/output
    dimensions derived from the dataset.

  - save_results(): persists result DataFrames as numbered parquet files so
    that multiple experiment runs do not overwrite each other.

  - measure_baselines(): fits and evaluates all four frequency-based baselines
    (Uniform, GlobalMarginal, NodeMarginal, ConditionalNodeMarginal).

  - comparison_experiment(): the main driver that builds all models, runs them,
    evaluates the baselines, and saves everything to disk.

Models compared:
  - MLP (node features only)
  - GATSkip (Graph Attention Network with skip connections)
  - FullyConnectedMLP (uses all nodes' features concatenated)

Population-density variants (no-offset / population-as-offset / population-as-feature)
are run for each model family.
"""

import functools
from datetime import datetime
from pathlib import Path

import polars as pl

from activitygraphs.config import Config
from activitygraphs.ml.baselines import (
    ConditionalNodeBaseline,
    GlobalBaseline,
    NodeBaseline,
    UniformBaseline,
)
from activitygraphs.ml.datamodule import ActivityDataModule
from activitygraphs.ml.dataset import ActivityDataset
from activitygraphs.ml.experiment import evaluate_baseline, run_experiment, WandBParams
from activitygraphs.ml.lightning_module import extracted_features_dim
from activitygraphs.ml.models import FullyConnectedMLP, GATSkip, GraphTransformer, NodeMLP


def build_gat(
    dataset: ActivityDataset,
    num_gcn_layers: int,
    hidden_channels: int,
    dropout: float,
    use_demographics: bool = True,
    use_pop_feature: bool = False,
) -> GATSkip:
    """
    Description: Instantiate a GATSkip (Graph Attention Network with skip/residual
    connections) model whose input and output dimensions are derived automatically
    from the provided dataset.  The architecture always uses 1 pre-processing MLP
    layer and 3 post-processing MLP layers around the configurable number of GAT
    message-passing layers.

    Input:
      - dataset (ActivityDataset): the PyG dataset used for training; used to
                                   read edge feature dimensions and output class count.
      - num_gcn_layers (int): number of GAT message-passing (graph convolution) layers.
      - hidden_channels (int): width (number of hidden units) in every layer.
      - dropout (float): dropout probability applied during training (0.0 = no dropout).
      - use_demographics (bool): if True, append household demographic features
                                 (num_adults, num_children) to node input features.
                                 Defaults to True.
      - use_pop_feature (bool): if True, append a population-density value as an
                                additional input feature. Defaults to False.

    Output:
      - (GATSkip): configured but untrained GATSkip model instance ready for training.
    """
    # Read the number of edge features from the first graph in the dataset.
    # edge_attr.size(-1) gives the last (feature) dimension of the edge attribute tensor.
    edge_dim = dataset[0].edge_attr.size(-1)

    return GATSkip(
        num_pre_layers=1,       # 1 MLP layer applied to node features before the GAT layers.
        num_gcn_layers=num_gcn_layers,  # Number of GAT message-passing layers.
        num_post_layers=3,      # 3 MLP layers applied to node embeddings after the GAT layers.
        in_channels=extracted_features_dim(dataset, use_demographics=use_demographics, use_pop_feature=use_pop_feature),
        # in_channels: total number of input features per node after feature extraction.
        hidden_channels=hidden_channels,  # Width of hidden layers throughout the network.
        out_channels=dataset.num_classes - 1,
        # out_channels: number of output logits = number of location classes minus 1
        # (the home location is excluded from the prediction target).
        edge_dim=edge_dim,      # Dimensionality of the edge attribute vectors.
        dropout=dropout,        # Probability of zeroing each activation during training.
        residuals=True,         # Enable skip connections to ease gradient flow.
    )


def build_gps(
    dataset: ActivityDataset,
    num_gps_layers: int,
    hidden_channels: int,
    dropout: float,
    use_demographics: bool = True,
    use_pop_feature: bool = False,
):
    """
    Description: Instantiate a GraphTransformer (GPS — Graph Positional
    Strategies + Transformers) model whose dimensions are derived from the
    dataset.  GPS combines local message passing with a global Transformer
    attention mechanism.  This model is currently commented-out in the main
    comparison loop but the builder is retained for future experiments.

    Input:
      - dataset (ActivityDataset): the PyG dataset; used to infer feature and
                                   class count dimensions.
      - num_gps_layers (int): number of GPS layers (each layer = local MPNN +
                              global Transformer attention).
      - hidden_channels (int): width of all hidden representations.
      - dropout (float): dropout probability (0.0 = no dropout).
      - use_demographics (bool): if True, include demographic features in the
                                 input. Defaults to True.
      - use_pop_feature (bool): if True, include population density as a feature.
                                Defaults to False.

    Output:
      - (GraphTransformer): configured but untrained GraphTransformer model.
    """
    # Read the edge feature dimension from the first graph sample.
    edge_dim = dataset[0].edge_attr.size(-1)

    return GraphTransformer(
        # Total number of input node features (depends on feature flags above).
        in_channels=extracted_features_dim(dataset, use_demographics=use_demographics, use_pop_feature=use_pop_feature),
        hidden_channels=hidden_channels,   # Width of all hidden layers.
        out_channels=dataset.num_classes - 1,  # One logit per candidate location (minus home).
        edge_dim=edge_dim,      # Dimensionality of edge attribute vectors.
        num_layers=num_gps_layers,  # Number of stacked GPS layers.
        num_heads=4,            # Number of parallel attention heads in the Transformer block.
        dropout=dropout,        # Dropout probability for regularisation.
    )


def build_mlp(
    dataset: ActivityDataset,
    mlp_layers: int,
    hidden_channels: int,
    dropout: float,
    use_demographics: bool = True,
    full_info: bool = False,
    use_pop_feature: bool = False,
) -> NodeMLP:
    """
    Description: Instantiate a NodeMLP (node-level Multi-Layer Perceptron) model.
    This is the graph-free baseline: it makes predictions for each node using
    only that node's own feature vector, with no message passing between nodes.

    When ``full_info=True``, an extra distance-from-home feature is appended to
    each node's input, creating the "distance-augmented MLP" variant that has
    access to spatial information without graph structure.

    Input:
      - dataset (ActivityDataset): dataset used to infer feature and class dimensions.
      - mlp_layers (int): number of hidden layers in the MLP.
      - hidden_channels (int): width of each hidden layer.
      - dropout (float): dropout probability during training.
      - use_demographics (bool): if True, include demographic features. Defaults to True.
      - full_info (bool): if True, include the distance-from-home feature as an
                          additional input. Defaults to False.
      - use_pop_feature (bool): if True, include population density as a feature.
                                Defaults to False.

    Output:
      - (NodeMLP): configured but untrained NodeMLP model instance.
    """
    return NodeMLP(
        mlp_layers,  # Number of hidden MLP layers.
        in_channels=extracted_features_dim(
            dataset, use_demographics=use_demographics, full_info=full_info, use_pop_feature=use_pop_feature
        ),
        # in_channels: total number of input features per node, including any optional extras.
        hidden_channels=hidden_channels,       # Width of all hidden layers.
        out_channels=dataset.num_classes - 1,  # One output logit per candidate location.
        dropout=dropout,                        # Dropout probability for regularisation.
    )


def build_full_mlp(
    dataset: ActivityDataset,
    mlp_layers: int,
    hidden_channels: int,
    dropout: float,
    use_demographics: bool = True,
    use_pop_feature: bool = False,
) -> FullyConnectedMLP:
    """
    Description: Instantiate a FullyConnectedMLP model.  Unlike the NodeMLP which
    operates on one node at a time, the FullyConnectedMLP takes ALL nodes' features
    concatenated into a single flat vector and outputs predictions for all nodes at
    once.  This gives it full global information about the graph without using
    message passing.  Because the input is large (num_nodes × features), it is
    regularised more aggressively than other models.

    Input:
      - dataset (ActivityDataset): dataset used to read the number of nodes and
                                   feature dimensions.
      - mlp_layers (int): number of hidden layers in the MLP.
      - hidden_channels (int): width of each hidden layer.
      - dropout (float): dropout probability during training.
      - use_demographics (bool): if True, include demographic features in the
                                 per-node input. Defaults to True.
      - use_pop_feature (bool): if True, include population density as a feature.
                                Defaults to False.

    Output:
      - (FullyConnectedMLP): configured but untrained FullyConnectedMLP instance.
    """
    return FullyConnectedMLP(
        num_nodes=dataset.num_nodes,  # Total number of location nodes in the graph.
        in_features=extracted_features_dim(dataset, use_demographics=use_demographics, use_pop_feature=use_pop_feature),
        # in_features: number of features per node before concatenation.
        hidden_channels=hidden_channels,  # Width of all hidden layers.
        num_layers=mlp_layers,            # Number of hidden layers in the MLP.
        dropout=dropout,                  # Dropout probability for regularisation.
    )


def save_results(path: str | Path, name: str, *results: pl.DataFrame):
    """
    Description: Concatenate one or more result DataFrames (one per model or
    baseline) and write the combined table to a uniquely numbered parquet file
    so that successive experiment runs do not overwrite previous results.

    The output file name format is: ``<path>/data/<name>-results-<N>.parquet``
    where N is one greater than the highest existing N for files with the same
    prefix.  If no matching files exist, N starts at 1.

    Input:
      - path (str | Path): directory under which the ``data/`` sub-folder is found.
                           Typically ``cfg.paths.reports``.
      - name (str): dataset name prefix used to name the output file
                    (e.g. "toronto" → "toronto-results-1.parquet").
      - *results (pl.DataFrame): any number of Polars DataFrames to concatenate.
                                 Each typically contains metrics for one model.
                                 Uses ``how="diagonal"`` so columns that exist only
                                 in some frames are filled with nulls elsewhere.

    Output:
      - (None): writes a parquet file to disk; returns nothing.
    """
    # Resolve the output directory: reports/data/
    path = Path(path) / "data"

    # Generator over the trailing numeric part of existing result file names.
    # e.g. "toronto-results-3.parquet" → stem = "toronto-results-3" → end = "3"
    ends = (f.stem.split("-")[-1] for f in path.iterdir() if f.suffix == ".parquet" and f.name.startswith(name))
    # Convert trailing parts to integers; default to 0 for non-numeric suffixes.
    nums = (int(end) if end.isdecimal() else 0 for end in ends)
    # Find the highest existing run number; start list with 0 so max never fails.
    max_num = max([0, *nums])

    # Concatenate all result frames (diagonal allows mismatched columns) and
    # write to a new file numbered max_num + 1 to avoid overwriting.
    pl.concat(results, how="diagonal").write_parquet(path / f"{name}-results-{max_num + 1}.parquet")


def measure_baselines(num_nodes, datamodule: ActivityDataModule, wandb_params: WandBParams):
    """
    Description: Fit and evaluate all four non-learned frequency baselines.
    These baselines do not use any graph structure or neural networks; they
    predict choice sets based purely on historical visitation frequencies.
    Their performance provides a lower bound that learned models should beat.

    The four baselines are:
      - UniformBaseline: assigns equal probability to every location.
      - GlobalBaseline: uses the global marginal visit frequency across all users.
      - NodeBaseline: uses the per-location marginal visit frequency.
      - ConditionalNodeBaseline: uses per-location frequency conditioned on
                                 whether the trip starts from home.

    Input:
      - num_nodes (int): total number of location nodes in the graph; passed to
                         baselines that need to allocate per-node frequency tables.
      - datamodule (ActivityDataModule): PyTorch Lightning data module that provides
                                         train/val/test data loaders and dataset info.
      - wandb_params (WandBParams): Weights & Biases logging configuration; passed
                                    through to evaluate_baseline.

    Output:
      - (list[pl.DataFrame]): list of result DataFrames, one per baseline, each
                              containing metrics (e.g. Precision@k, Recall@k, F1).
    """
    # Ensure the data module is set up (splits created, datasets loaded).
    datamodule.setup()
    # Obtain the training data loader to fit frequency-based baselines.
    train_loader = datamodule.train_dataloader()
    # Column index of the "is_home" flag in the node feature matrix.
    # ConditionalNodeBaseline uses this to condition predictions on home-start trips.
    is_home_idx = datamodule.train_dataset.is_home_col_idx

    # Build each baseline; fit() counts visit frequencies from the training set.
    uniform_base = UniformBaseline()                                         # No fitting needed.
    global_base = GlobalBaseline().fit(train_loader)                        # Global frequency over all locations.
    node_base = NodeBaseline(num_nodes).fit(train_loader)                   # Per-location visit frequency.
    conditional_base = ConditionalNodeBaseline(num_nodes, is_home_idx).fit(train_loader)  # Conditioned on home.

    # Accumulate one result DataFrame per baseline.
    results = []

    for name, baseline in [
        ("Uniform", uniform_base),
        ("GlobalMarginal", global_base),
        ("NodeMarginal", node_base),
        ("ConditionalNodeMarginal", conditional_base),
    ]:
        # evaluate_baseline runs the baseline on val/test splits and returns a metrics DataFrame.
        res = evaluate_baseline(
            baseline, datamodule, name,
            k=datamodule.train_dataset.median_realised_size,  # k = median choice-set size in training data
            wandb_params=wandb_params
        )
        results.append(res)

    return results


def comparison_experiment(cfg: Config):
    """
    Description: Main experiment driver.  Compares the location choice-set prediction
    performance of multiple models (MLP, GATSkip, FullyConnectedMLP) in three
    population-density variants (no population info / population as additive offset /
    population as input feature) plus four frequency baselines.

    The experiment proceeds in this order:
      1. Set up W&B logging and data splits.
      2. Build all model instances with the correct dimensions.
      3. Evaluate all four baselines on the training splits.
      4. Train and evaluate each neural model via ``run_experiment``.
      5. Write all result DataFrames to a numbered parquet file.

    Fixed hyperparameters (normal mode): batch_size=64, hidden_channels=128,
    dropout=0.2, weight_decay=1e-4.
    When overfit_batches > 0 (debug/overfit mode): dropout=0.0, lr=0.01, epochs=500.

    Input:
      - cfg (Config): fully resolved Hydra Config object containing dataset config,
                      output paths, and training hyper-parameters.

    Output:
      - (None): all model results are saved to
                ``cfg.paths.reports/data/<dataset>-results-<N>.parquet``.
                Returns early without saving if ``cfg.train.fast_dev_run`` is True.
    """

    # Create a unique group name for this experiment run using the dataset name
    # and the current timestamp (format: YYYYMMDD-HHMMSS).
    run_group = f"{cfg.data.name}-{datetime.now():%Y%m%d-%H%M%S}"

    # Bundle all W&B (Weights & Biases) logging settings into a single dataclass.
    wandb_params = WandBParams(
        use_wandb=cfg.train.wandb,          # Whether to actually log to W&B.
        project=cfg.train.wandb_project,    # W&B project name.
        entity=cfg.train.wandb_entity,      # W&B team/user entity (or None for default).
        group=run_group,                    # Group name to cluster runs from this experiment.
        dataset_name=cfg.data.name,         # Dataset name tag stored with each run.
    )

    # --- Data split sizes ---
    batch_size = 64    # Number of graph samples processed in one forward/backward pass.
    val_size = 0.1     # Fraction of data reserved for validation (10%).
    test_size = 0.2    # Fraction of data reserved for testing (20%).
    seed = 42          # Random seed for reproducible train/val/test splits.

    # Create the data module and prepare datasets (splits are created inside setup()).
    datamodule = ActivityDataModule(cfg, val_size=val_size, test_size=test_size, seed=seed, batch_size=batch_size)
    datamodule.setup()

    # Convenience reference to the training portion of the dataset.
    train_dataset = datamodule.train_dataset

    # --- Model architecture hyperparameters ---
    hidden_channels = 128  # Number of hidden units in all layers across all models.
    gat_layers = 8         # Number of GAT message-passing layers for GATSkip.
    gps_layers = 2         # Number of GPS layers for GraphTransformer (currently unused).
    mlp_layers = 3         # Number of hidden layers for all MLP variants.

    # Check whether we are in "overfitting mode" (intentionally over-fit a small batch
    # to verify the model can memorise data — useful for debugging model capacity).
    overfitting = cfg.train.overfit_batches > 0
    # debug=True when overfitting or when explicitly requested — enables extra logging.
    debug = overfitting or cfg.train.debug

    if overfitting:
        # In overfitting mode: no regularisation, high learning rate, many epochs.
        dropout = 0.0         # No dropout so the model can fully memorise the data.
        epochs = 500          # Many epochs to ensure convergence on a tiny batch.
        lr = 1e-2             # Higher learning rate to speed up convergence.
        weight_decay = 0.0    # No L2 regularisation so nothing prevents memorisation.
    else:
        # Normal training mode: standard regularisation settings.
        dropout = 0.2         # 20% dropout probability for all models.
        epochs = cfg.train.epochs  # Number of epochs from the config YAML.
        lr = 1e-3             # Default Adam learning rate.
        weight_decay = 1e-4   # Mild L2 weight decay for regularisation.

    verbose = 1  # Verbosity level for PyTorch Lightning trainer (1 = progress bar).

    # Read the number of nodes from the first graph sample in the training set.
    num_nodes = train_dataset[0].num_nodes
    # Evaluate all four frequency baselines before training any neural models.
    baseline_results = measure_baselines(num_nodes, datamodule, wandb_params)

    # Directory where trained model checkpoints will be saved.
    models_dir = cfg.paths.models

    # Per-model learning rates. GPS (GraphTransformer) is not as good as GATSkip at 1e-3. If `overfit`, all models get
    # the same lr=0.01
    # Maps model family name → best learning rate from preliminary experiments.
    lr_by_model = {"MLP": 1e-3, "GATSkip": 1e-3, "GTransformer": 1e-4, "MLP-dist": 1e-3, "FullMLP": 1e-3}

    def lr_for(key: str) -> float:
        """
        Description: Return the learning rate for the specified model family.
        In overfitting mode, all models use the same debug learning rate.

        Input:
          - key (str): model family name, must be a key in lr_by_model.

        Output:
          - (float): the learning rate to use for this model.
        """
        # In overfitting mode use the single debug lr; otherwise use the per-model value.
        return lr if overfitting else lr_by_model[key]

    # Regularise full model more to avoid overfitting.
    # FullyConnectedMLP has a much larger input, so stronger regularisation is needed.
    full_dropout = 0.0 if overfitting else 0.5     # 50% dropout for FullMLP in normal mode.
    full_weight_decay = weight_decay if overfitting else 1e-3  # Stronger L2 for FullMLP.

    # Human-readable names for model checkpoints and result tables.
    gat_name = f"GATSkip-{gat_layers}-res"       # e.g. "GATSkip-8-res"
    gps_name = f"GTransformer-{gps_layers}-res"   # e.g. "GTransformer-2-res" (unused)

    # --- Build all model instances ---
    # Each model family has three variants: no-population, population-as-offset, population-as-feature.

    # MLP variants (no graph structure):
    mlp = build_mlp(train_dataset, mlp_layers, hidden_channels, dropout)
    # MLP with population as offset (pop density added to the output logit, not the input).
    mlp_pop_off = build_mlp(train_dataset, mlp_layers, hidden_channels, dropout)
    # MLP with population density as an additional input feature.
    mlp_pop_feat = build_mlp(train_dataset, mlp_layers, hidden_channels, dropout, use_pop_feature=True)

    # GATSkip variants (graph attention with residuals):
    gat = build_gat(train_dataset, gat_layers, hidden_channels, dropout)
    # GATSkip with population as offset.
    gat_pop_off = build_gat(train_dataset, gat_layers, hidden_channels, dropout)
    # GATSkip with population density as an additional input feature.
    gat_pop_feat = build_gat(train_dataset, gat_layers, hidden_channels, dropout, use_pop_feature=True)

    # FullyConnectedMLP variants (all nodes' features concatenated):
    full_mlp = build_full_mlp(train_dataset, mlp_layers, hidden_channels, full_dropout)
    # FullyConnectedMLP with population as offset.
    full_mlp_off = build_full_mlp(train_dataset, mlp_layers, hidden_channels, full_dropout)
    # FullyConnectedMLP with population density as an additional input feature.
    full_mlp_feat = build_full_mlp(train_dataset, mlp_layers, hidden_channels, full_dropout, use_pop_feature=True)

    # gps = build_gps(train_dataset, gps_layers, hidden_channels, dropout)
    # mlp_dist = build_mlp(train_dataset, mlp_layers, hidden_channels, dropout, full_info=True)
    # full_mlp = build_full_mlp(train_dataset, mlp_layers, hidden_channels, full_dropout)

    # Create a partial version of run_experiment with all shared arguments pre-filled.
    # Only model-specific arguments (model, name, lr, pop_mode, weight_decay) vary per call.
    my_run_experiment = functools.partial(
        run_experiment,
        datamodule=datamodule,               # Provides train/val/test data loaders.
        num_epochs=epochs,                   # Total training epochs.
        verbose=verbose,                     # Trainer verbosity level.
        weight_decay=weight_decay,           # Default L2 weight decay (may be overridden per model).
        model_save_dir=models_dir,           # Directory for saving model checkpoints.
        fast_dev_run=cfg.train.fast_dev_run, # Run only 1 batch if True (sanity check).
        overfit_batches=cfg.train.overfit_batches,  # Number of batches to overfit on.
        schedule_lr=cfg.train.schedule_lr,   # Whether to use LR scheduling.
        wandb_params=wandb_params,           # W&B logging config.
        debug=debug,                         # Debug flag for extra logging.
    )

    # --- Train and evaluate each model ---
    # pop_mode controls how population density is incorporated:
    #   (not passed)  = ignore population
    #   "offset"      = add pop density to the output logit as a constant offset
    #   "feature"     = append pop density to node input features

    # MLP runs:
    results_mlp = my_run_experiment(model=mlp, name="MLP", lr=lr_for("MLP"))
    results_mlp_off = my_run_experiment(model=mlp_pop_off, name="MLP-off", lr=lr_for("MLP"), pop_mode="offset")
    results_mlp_feat = my_run_experiment(model=mlp_pop_feat, name="MLP-feat", lr=lr_for("MLP"), pop_mode="feature")

    # FullyConnectedMLP runs (use full_weight_decay for stronger regularisation):
    results_full = my_run_experiment(
        model=full_mlp, name="FullMLP", lr=lr_for("FullMLP"), weight_decay=full_weight_decay
    )
    results_full_off = my_run_experiment(
        model=full_mlp_off,
        name="FullMLP-pop-offset",
        lr=lr_for("FullMLP"),
        weight_decay=full_weight_decay,
        pop_mode="offset",
    )
    results_full_feat = my_run_experiment(
        model=full_mlp_feat,
        name="FullMLP-pop-feat",
        lr=lr_for("FullMLP"),
        weight_decay=full_weight_decay,
        pop_mode="feature",
    )

    # GATSkip runs:
    results_gat = my_run_experiment(model=gat, name=gat_name, lr=lr_for("GATSkip"))
    results_gat_off = my_run_experiment(
        model=gat_pop_off, name=gat_name + "-pop-offset", lr=lr_for("GATSkip"), pop_mode="offset"
    )
    results_gat_feat = my_run_experiment(
        model=gat_pop_feat, name=gat_name + "-pop-feat", lr=lr_for("GATSkip"), pop_mode="feature"
    )

    # results_gps = my_run_experiment(model=gps, name=gps_name, lr=lr_for("GTransformer"))
    # results_mlp_dist = my_run_experiment(model=mlp_dist, name="MLP-dist", lr=lr_for("MLP-dist"), full_info=True)

    # model_results = [results_mlp, results_gat, results_gps, results_mlp_dist, results_full]

    # Collect all neural-model result DataFrames in a list for saving.
    model_results = [
        results_mlp,        # Standard MLP, no population information.
        results_mlp_off,    # MLP with population as additive offset.
        results_mlp_feat,   # MLP with population as input feature.
        results_gat,        # GATSkip, no population information.
        results_gat_off,    # GATSkip with population as additive offset.
        results_gat_feat,   # GATSkip with population as input feature.
        results_full,       # FullyConnectedMLP, no population information.
        results_full_off,   # FullyConnectedMLP with population as additive offset.
        results_full_feat,  # FullyConnectedMLP with population as input feature.
    ]

    # In fast_dev_run mode, skip saving results (the run is only a sanity check).
    if cfg.train.fast_dev_run:
        return

    # Concatenate and persist all results (model + baseline) to a numbered parquet file.
    save_results(cfg.paths.reports, cfg.data.name, *model_results, *baseline_results)
