import pickle
from pathlib import Path

import polars as pl
import torch_geometric as pyg
from sklearn.model_selection import train_test_split

from activitygraphs.config import Config
from activitygraphs.experiment import run_experiment
from activitygraphs.models import GATSkip, NodeMLP, GraphTransformer


class GenevaDataset(pyg.data.InMemoryDataset):
    def __init__(self, dataset_path):
        super().__init__()

        print("Loading dataset...")

        with open(dataset_path, "rb") as f:
            graphs = pickle.load(f)

        print(f"Loaded {len(graphs)} graphs. Applying transforms...")

        self._graphs = graphs

        print("Transforms applied.")

    @property
    def num_classes(self):
        return 1  # self._infer_num_classes(self._graphs[0].y)

    def len(self) -> int:
        return len(self._graphs)

    def get(self, idx: int):
        return self._graphs[idx]


def load_dataset(cfg: Config) -> GenevaDataset:
    processed_path = Path(cfg.data.paths.processed) / "GenevaTPG2"
    dataset_path = processed_path / "dataset.pickle"

    return GenevaDataset(dataset_path)


def split_dataset(dataset: GenevaDataset, test_size: float, batch_size: int, seed: int):
    train_indices, test_indices = train_test_split(range(len(dataset)), test_size=test_size, random_state=seed)

    train_dataset = dataset[train_indices]
    test_dataset = dataset[test_indices]

    train_loader = pyg.loader.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = pyg.loader.DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader


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


def geneva_experiment(cfg: Config):
    batch_size = 64
    test_size = 0.2
    seed = 42

    dataset = load_dataset(cfg)
    train_loader, test_loader = split_dataset(dataset, test_size, batch_size, seed)

    hidden_channels = 128
    dropout = 0.2
    gat_layers = 8
    gps_layers = 4
    mlp_layers = 3

    mlp = build_mlp(dataset, mlp_layers, hidden_channels, dropout)
    gat = build_gat(dataset, gat_layers, hidden_channels, dropout)
    gps = build_gps(dataset, gps_layers, hidden_channels, dropout)

    mlp_l1 = build_mlp(dataset, mlp_layers, hidden_channels, dropout)
    gat_l1 = build_gat(dataset, gat_layers, hidden_channels, dropout)
    gps_l1 = build_gps(dataset, gps_layers, hidden_channels, dropout)


    epochs = 50
    verbose = 5
    lr = 1e-4

    gname = f"GATSkip-{gat_layers}-res"
    tname = f"GTransformer-{gps_layers}-res"

    results_mlp = run_experiment(mlp, train_loader, test_loader, num_epochs=epochs, verbose=verbose, name="MLP", lr=lr, reg="l1", save=True)
    results_gat = run_experiment(gat, train_loader, test_loader, num_epochs=epochs, verbose=verbose, name=gname, lr=lr, reg="l1", save=True)
    results_gps = run_experiment(gps, train_loader, test_loader, num_epochs=epochs, verbose=verbose, name=tname, lr=lr, reg="l1", save=True)

    results_mlp_l1 = run_experiment(
        mlp_l1, train_loader, test_loader, num_epochs=epochs, verbose=verbose, name="MLP-l1", lr=lr, reg="l1", save=True
    )
    results_gat_l1 = run_experiment(
        gat_l1, train_loader, test_loader, num_epochs=epochs, verbose=verbose, name=gname + "-l1", lr=lr, reg="l1", save=True
    )
    results_gps_l1 = run_experiment(
        gps_l1, train_loader, test_loader, num_epochs=epochs, verbose=verbose, name=tname + "-l1", lr=lr, reg="l1", save=True
    )

    save_results(cfg.paths.reports, "geneva", results_gat, results_gps, results_mlp, results_mlp_l1, results_gat_l1, results_gps_l1)
