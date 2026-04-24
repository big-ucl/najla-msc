import pickle
from pathlib import Path

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


def load_dataset(cfg: Config, test_size: float, seed: int, project_root: Path | None = None,
                 graphs_name: str = "Graphs") -> tuple[GenevaDataset, GenevaDataset]:
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
