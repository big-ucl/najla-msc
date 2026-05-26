import json
import pickle
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
import torch_geometric as pyg
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.data.data import BaseData
from tqdm import tqdm

from activitygraphs.base import IS_HOME_COL_IDX
from activitygraphs.config import Config
from activitygraphs.utils import get_project_root


class GenevaDataset(pyg.data.InMemoryDataset):
    def __init__(self, dataset_path: Path):
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


class ActivityDataset(pyg.data.Dataset):
    def __init__(
        self,
        root: str,
        network_graph: pyg.data.Data | None = None,
        spatial_features: torch.Tensor | None = None,
        spatial_labels: torch.Tensor | None = None,
        demographics: torch.Tensor | None = None,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
    ):
        self._network_graph_in = network_graph
        self._spatial_features_in = spatial_features
        self._spatial_labels_in = spatial_labels
        self._demographics_in = demographics

        super().__init__(root, transform, pre_transform, pre_filter)

        processed_dir = Path(self.processed_dir)
        self.network_graph: pyg.data.Data = torch.load(processed_dir / "network_graph.pt", weights_only=False)
        self.spatial_features: torch.Tensor = torch.load(processed_dir / "spatial_features.pt", weights_only=True)
        self.spatial_labels: torch.Tensor = torch.load(processed_dir / "spatial_labels.pt", weights_only=True)
        self.demographics: torch.Tensor = torch.load(processed_dir / "demographics.pt", weights_only=True)

        num_nodes = self.network_graph.num_nodes
        if num_nodes is None:
            raise ValueError("Network graph has `None` number of nodes")

        self.num_individuals: int = self.demographics.size(0)
        self.num_nodes: int = num_nodes

    @property
    def raw_file_names(self) -> list[str]:
        return []

    @property
    def processed_file_names(self) -> list[str]:
        return [
            "network_graph.pt",
            "spatial_features.pt",
            "spatial_labels.pt",
            "demographics.pt",
        ]

    def download(self) -> None:
        pass

    def process(self):
        graph = self._network_graph_in
        sf = self._spatial_features_in
        sl = self._spatial_labels_in
        demo = self._demographics_in

        if graph is None or sf is None or sl is None or demo is None:
            raise ValueError("First-time construction requires the four component Tensors/Data to be passed to init.")

        num_sf_users, num_sf_nodes, _ = sf.shape
        num_sl_users, num_sl_nodes, num_sl_labels = sl.shape
        num_dm_users, _ = demo.shape

        if not (graph.num_nodes == num_sf_nodes == num_sl_nodes):
            raise ValueError(
                f"Node count mismatch: network has {graph.num_nodes}, spatial_features has {num_sf_nodes}, spatial_labels has {num_sl_nodes}."
            )
        if not (num_sf_users == num_sl_users == num_dm_users):
            raise ValueError(
                f"Individual count mismatch: spatial_features has {num_sf_users}, spatial_labels has {num_sl_users}, demographics has {num_dm_users}."
            )
        if num_sl_labels != 1:
            raise ValueError(f"spatial_labels last dim must be 1, got {num_sl_labels}.")

        # Make sure the network graph has node features, otherwise, fall back to a zero-width tensor.
        if graph.x is None:
            graph = graph.clone()
            graph.x = torch.empty((graph.num_nodes, 0), dtype=sf.dtype)

        if self.pre_transform is not None:
            graph = self.pre_transform(graph)

        processed_dir = Path(self.processed_dir)
        torch.save(graph, processed_dir / "network_graph.pt")
        torch.save(sf, processed_dir / "spatial_features.pt")
        torch.save(sl, processed_dir / "spatial_labels.pt")
        torch.save(demo, processed_dir / "demographics.pt")

    def len(self) -> int:
        return self.num_individuals

    def get(self, i: int) -> pyg.data.Data:
        if not 0 <= i < self.num_individuals:
            raise IndexError(f"Index {i} out of range for {self.num_individuals} individuals.")

        network_x = self.network_graph.x
        spatial_x = self.spatial_features[i]

        if network_x is None:
            network_x = torch.empty((self.num_nodes, 0), dtype=spatial_x.dtype)

        # Cast spatial features to same type as network graph features if mismatch
        if network_x.numel() > 0 and spatial_x.dtype != network_x.dtype:
            spatial_x = spatial_x.to(network_x.dtype)

        # Create new individual-annotated network graph
        full_x = torch.cat([network_x, spatial_x], dim=1)
        y = self.spatial_labels[i]
        graph_x = self.demographics[i]
        edge_index = self.network_graph.edge_index
        edge_attr = self.network_graph.edge_attr

        data = pyg.data.Data(x=full_x, edge_index=edge_index, edge_attr=edge_attr, y=y)

        # Unsqueeze so batching works
        data.graph_x = graph_x.unsqueeze(0)
        data.user_id = torch.tensor([i], dtype=torch.long)

        # Carry over any extra attributes from the network graph (pos, etc.)
        for key, value in self.network_graph:
            if key in ("x", "edge_index", "edge_attr", "y"):
                continue

            data[key] = value

        return data


@dataclass
class FittedScalers:
    network_features: StandardScaler | None
    network_edges: StandardScaler | None
    spatial: StandardScaler | None
    demographics: StandardScaler | None

    def save(self, path: Path) -> None:
        with path.open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "FittedScalers":
        with path.open("rb") as f:
            return pickle.load(f)


def load_or_build_dataset(
    cfg: Config,
    project_root: Path | None = None,
    **build_kwargs,
) -> ActivityDataset:
    project_root = get_project_root(project_root)
    root = project_root / cfg.data.paths.pyg_datasets
    return ActivityDataset(root=str(root), **build_kwargs)


def split_indices(
    dataset: ActivityDataset,
    test_size: float,
    seed: int,
    cache_path: Path | None = None,
) -> tuple[list[int], list[int]]:
    if cache_path is not None and cache_path.exists():
        with cache_path.open() as f:
            indices = json.load(f)

        return indices["train"], indices["test"]

    train_idx, test_idx = train_test_split(
        list(range(len(dataset))),
        test_size=test_size,
        random_state=seed,
    )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w") as f:
            json.dump({"train": train_idx, "test": test_idx, "seed": seed}, f)

    return train_idx, test_idx


def fit_scalers(
    dataset: ActivityDataset,
    train_idx: list[int],
    exclude_spatial_cols: list[int] | None = None,
    exclude_demographic_cols: list[int] | None = None,
) -> FittedScalers:
    network_features = dataset.network_graph.x
    edge_attr = dataset.network_graph.edge_attr

    # Network node features: no need to separate between train and test, the network is the same.
    network_scaler = None
    if network_features is not None and network_features.numel() > 0:
        network_scaler = StandardScaler()
        network_scaler.fit(network_features.numpy())

    # Network edge attributes: same as above
    edge_scaler = None
    if edge_attr is not None:
        edge_scaler = StandardScaler()
        edge_scaler.fit(edge_attr.numpy())

    # Spatial features: individual-specific, fit only on train
    spatial_train = dataset.spatial_features[train_idx]
    if exclude_spatial_cols:
        keep = [c for c in range(spatial_train.shape[-1]) if c not in exclude_spatial_cols]
        spatial_train = spatial_train[..., keep]
    spatial_flat = spatial_train.reshape(-1, spatial_train.shape[-1]).float().numpy()
    spatial_scaler = StandardScaler()
    spatial_scaler.fit(spatial_flat)

    # Demographics: individual-specific, fit only on train
    demo_train = dataset.demographics[train_idx]
    if exclude_demographic_cols:
        keep = [c for c in range(demo_train.shape[-1]) if c not in exclude_demographic_cols]
        demo_train = demo_train[..., keep]
    demo_scaler = StandardScaler()
    demo_scaler.fit(demo_train.float().numpy())

    return FittedScalers(
        network_features=network_scaler, network_edges=edge_scaler, spatial=spatial_scaler, demographics=demo_scaler
    )


def apply_scalers(dataset: ActivityDataset, scalers: FittedScalers) -> None:
    if scalers.network_features is not None:
        x = dataset.network_graph.x.numpy()
        dataset.network_graph.x = torch.from_numpy(scalers.network_features.transform(x)).float()

    if scalers.network_edges is not None:
        ea = dataset.network_graph.edge_attr.numpy()
        dataset.network_graph.edge_attr = torch.from_numpy(scalers.network_edges.transform(ea)).float()

    if scalers.spatial is not None:
        sf = dataset.spatial_features.float()
        flat = sf.reshape(-1, sf.shape[-1]).numpy()
        scaled = torch.from_numpy(scalers.spatial.transform(flat)).float()
        dataset.spatial_features = scaled.reshape(sf.shape)

    if scalers.demographics is not None:
        demo = dataset.demographics.float().numpy()
        dataset.demographics = torch.from_numpy(scalers.demographics.transform(demo)).float()


def load_dataset(
    cfg: Config,
    test_size: float,
    seed: int,
    project_root: Path | None = None,
    **build_kwargs,
) -> tuple[ActivityDataset, ActivityDataset, FittedScalers]:
    project_root = get_project_root(project_root)
    pyg_dir = project_root / cfg.data.paths.pyg_datasets
    splits_cache = pyg_dir / "splits.json"
    scalers_cache = pyg_dir / "scalers.pkl"

    dataset = load_or_build_dataset(cfg, project_root=project_root, **build_kwargs)
    train_idx, test_idx = split_indices(dataset, test_size, seed, cache_path=splits_cache)

    if scalers_cache.exists():
        scalers = FittedScalers.load(scalers_cache)
    else:
        scalers = fit_scalers(dataset, train_idx, exclude_spatial_cols=[IS_HOME_COL_IDX])
        scalers.save(scalers_cache)

    apply_scalers(dataset, scalers)

    return cast(ActivityDataset, dataset[train_idx]), cast(ActivityDataset, dataset[test_idx]), scalers


def load_gva_dataset(
    cfg: Config, test_size: float, seed: int, project_root: Path | None = None, graphs_name: str = "Graphs"
) -> tuple[GenevaDataset, GenevaDataset]:
    project_root: Path = get_project_root(project_root)

    pyg_path = project_root / cfg.data.paths.pyg_datasets
    graphs_path = pyg_path / f"{graphs_name}.pickle"

    train_path = pyg_path / "train.pickle"
    test_path = pyg_path / "test.pickle"

    if train_path.exists() and test_path.exists():
        print(f"Loading {pyg_path} dataset...")

        with open(train_path, "rb") as f:
            train_dataset: GenevaDataset = pickle.load(f)

        with open(test_path, "rb") as f:
            test_dataset: GenevaDataset = pickle.load(f)

        return train_dataset, test_dataset

    print(f"Loading {graphs_path} graph list and building dataset...")

    dataset = GenevaDataset(graphs_path)
    train_indices, test_indices = train_test_split(range(len(dataset)), test_size=test_size, random_state=seed)

    # noinspection PyTypeChecker
    train_dataset: GenevaDataset = dataset[train_indices]
    # noinspection PyTypeChecker
    test_dataset: GenevaDataset = dataset[test_indices]

    print(f"Train size: {len(train_dataset)}. Splitting dataset: ")

    num_features = train_dataset[0].x.shape[1]
    non_home_cols = [i for i in range(num_features) if i != IS_HOME_COL_IDX]

    train_x = torch.cat([g.x[:, non_home_cols] for g in train_dataset]).numpy()
    train_edge_attr = torch.cat([g.edge_attr for g in train_dataset]).numpy()

    x_scaler = StandardScaler()
    x_scaler.fit(train_x)

    e_scaler = StandardScaler()
    e_scaler.fit(train_edge_attr)

    print("Fitted scalers. Processing dataset:")

    # Transform both splits
    for g in tqdm(train_dataset):
        replace_scaled_features(g, e_scaler, x_scaler, non_home_cols)

    for g in tqdm(test_dataset):
        replace_scaled_features(g, e_scaler, x_scaler, non_home_cols)

    print("Writing processed datasets")

    with open(train_path, "wb") as f:
        pickle.dump(train_dataset, f)

    with open(test_path, "wb") as f:
        pickle.dump(test_dataset, f)

    print("Done.")

    return train_dataset, test_dataset


def replace_scaled_features(g: BaseData, e_scaler: StandardScaler, x_scaler: StandardScaler, non_home_cols: list[int]):
    g.x[:, non_home_cols] = torch.tensor(x_scaler.transform(g.x[:, non_home_cols].numpy()), dtype=torch.float)
    g.edge_attr = torch.tensor(e_scaler.transform(g.edge_attr.numpy()), dtype=torch.float)
