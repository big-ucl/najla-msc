from pathlib import Path

import polars as pl
import torch_geometric as pyg

from activitygraphs.config import Config
from activitygraphs.ml.baselines import (
    UniformBaseline,
    GlobalBaseline,
    NodeBaseline,
    ConditionalNodeBaseline,
)
from activitygraphs.ml.dataset import GenevaDataset, load_dataset
from activitygraphs.ml.experiment import run_experiment, evaluate_baseline, compute_training_weights
from activitygraphs.ml.models import GATSkip, NodeMLP, GraphTransformer


def build_gat(dataset: GenevaDataset, num_gcn_layers: int, hidden_channels: int, dropout: float) -> GATSkip:
    edge_dim = dataset[0].edge_attr.size(-1)

    return GATSkip(
        num_pre_layers=1,
        num_gcn_layers=num_gcn_layers,
        num_post_layers=3,
        in_channels=dataset.num_features,
        hidden_channels=hidden_channels,
        out_channels=dataset.num_classes,
        edge_dim=edge_dim,
        dropout=dropout,
        residuals=True,
    )


def build_gps(dataset: GenevaDataset, num_gps_layers: int, hidden_channels: int, dropout: float):
    edge_dim = dataset[0].edge_attr.size(-1)

    return GraphTransformer(
        in_channels=dataset.num_features,
        hidden_channels=hidden_channels,
        out_channels=dataset.num_classes,
        edge_dim=edge_dim,
        num_layers=num_gps_layers,
        num_heads=4,
        dropout=dropout,
    )


def build_mlp(dataset: GenevaDataset, mlp_layers: int, hidden_channels: int, dropout: float) -> NodeMLP:
    return NodeMLP(
        mlp_layers,
        in_channels=dataset.num_features,
        hidden_channels=hidden_channels,
        out_channels=dataset.num_classes,
        dropout=dropout,
    )


def save_results(path: str | Path, name: str, *results: dict):
    path = Path(path) / "data"
    results_df = pl.concat(pl.DataFrame(result) for result in results)
    results_df.write_parquet(path / f"{name}-results.parquet")


def measure_baselines(num_nodes, train_loader, test_loader):
    uniform_base = UniformBaseline()
    global_base = GlobalBaseline().fit(train_loader)
    node_base = NodeBaseline(num_nodes).fit(train_loader)
    conditional_base = ConditionalNodeBaseline(num_nodes).fit(train_loader)

    results = []
    pos_weight = compute_training_weights(train_loader)

    for name, baseline in [
        ("Uniform", uniform_base),
        ("GlobalMarginal", global_base),
        ("NodeMarginal", node_base),
        ("ConditionalNodeMarginal", conditional_base),
    ]:
        res = evaluate_baseline(baseline, test_loader, name, pos_weight=pos_weight)
        results.append(res)

    return results


def geneva_experiment(cfg: Config):
    batch_size = 64
    test_size = 0.2
    seed = 42

    train_dataset, test_dataset = load_dataset(cfg, test_size, seed)
    train_loader = pyg.loader.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = pyg.loader.DataLoader(test_dataset, batch_size=batch_size)

    hidden_channels = 128
    dropout = 0.2
    gat_layers = 8
    gps_layers = 2
    mlp_layers = 3

    mlp = build_mlp(train_dataset, mlp_layers, hidden_channels, dropout)
    gat = build_gat(train_dataset, gat_layers, hidden_channels, dropout)
    gps = build_gps(train_dataset, gps_layers, hidden_channels, dropout)

    mlp_l1 = build_mlp(train_dataset, mlp_layers, hidden_channels, dropout)
    gat_l1 = build_gat(train_dataset, gat_layers, hidden_channels, dropout)
    gps_l1 = build_gps(train_dataset, gps_layers, hidden_channels, dropout)

    epochs = 50
    verbose = 5
    lr = 1e-4

    gname = f"GATSkip-{gat_layers}-res"
    tname = f"GTransformer-{gps_layers}-res"
    glname = gname + "-l1"
    tlname = tname + "-l1"

    num_nodes = train_dataset[0].num_nodes
    baseline_results = measure_baselines(num_nodes, train_loader, test_loader)

    results_mlp = run_experiment(
        mlp, train_loader, test_loader, num_epochs=epochs, verbose=verbose, name="MLP", lr=lr, save=True
    )
    results_gat = run_experiment(
        gat, train_loader, test_loader, num_epochs=epochs, verbose=verbose, name=gname, lr=lr, save=True
    )
    results_gps = run_experiment(
        gps, train_loader, test_loader, num_epochs=epochs, verbose=verbose, name=tname, lr=lr, save=True
    )
    results_mlp_l1 = run_experiment(
        mlp_l1, train_loader, test_loader, num_epochs=epochs, verbose=verbose, name="MLP-l1", lr=lr, reg="l1", save=True
    )
    results_gat_l1 = run_experiment(
        gat_l1,
        train_loader,
        test_loader,
        num_epochs=epochs,
        verbose=verbose,
        name=glname,
        lr=lr,
        reg="l1",
        save=True,
    )
    results_gps_l1 = run_experiment(
        gps_l1,
        train_loader,
        test_loader,
        num_epochs=epochs,
        verbose=verbose,
        name=tlname,
        lr=lr,
        reg="l1",
        save=True,
    )

    save_results(
        cfg.paths.reports,
        "geneva",
        results_mlp,
        results_gat,
        results_gps,
        results_mlp_l1,
        results_gat_l1,
        results_gps_l1,
        *baseline_results,
    )
