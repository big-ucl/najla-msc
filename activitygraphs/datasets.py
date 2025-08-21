from typing import Protocol

import networkx as nx
import polars as pl
import polars.selectors as cs
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.utils import from_networkx


class Graph(Protocol):
    """A protocol to represent an input graph such as a SyntheticGraph"""

    WEIGHT_NAME: str

    @property
    def G_full(self) -> nx.Graph:
        pass


class Schedules(Protocol):
    """A protocol to represent a schedules object (e.g. SyntheticSchedules)"""

    n_samples: int
    graph: Graph
    person_choices_df: pl.DataFrame
    schedule_df: pl.DataFrame
    trip_df: pl.DataFrame


class BasicLocationsDataset(InMemoryDataset):
    """Represents a PyG dataset that contains the training data for a graph ML model"""

    def __init__(
        self,
        person_ids: pl.Series,
        data: Data,
        X: torch.Tensor,
        y: torch.Tensor,
    ):
        """
        Args:
            person_ids (pl.Series):
                A series of length `S` of all person_ids in the dataset
            data (pyg.data.Data):
                The underlying graph without the schedules
            X (torch.Tensor):
                The node features of shape (S, N + 1), where `N` is the number of nodes. The first column must be the
                sequence number
            y (torch.Tensor):
                The node targets (binary indicators) of shape (S, N)
        """
        super().__init__()
        self.edge_index = data.edge_index
        self.edge_attr = data.edge_attr
        self.data = data

        self._person_ids = person_ids
        self._graph_x = X[:, 0].unsqueeze(1).float()
        self._X = X[:, 1:].unsqueeze(2).float()
        self._y = (y.unsqueeze(2).float() - self._X).squeeze().argmax(dim=1)

    def len(self):
        return len(self._X)

    @property
    def num_classes(self) -> int:
        return self._data.num_nodes

    def person_ids(self):
        return self._person_ids[self.indices()]

    def get(self, idx):
        return Data(
            x=self._X[idx],
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            y=self._y[idx],
            graph_x=self._graph_x[idx],
            person_id=self._person_ids[idx],
        )


def convert_to_pyg_dataset(schedules: Schedules) -> Dataset:
    """Converts a population schedule object into a PyG Dataset"""

    pyg_graph = from_networkx(schedules.graph.G_full, group_edge_attrs=schedules.graph.WEIGHT_NAME)

    features = (
        schedules.trip_df.group_by("person_id")
        .agg(pl.col("from_loc_id").unique(maintain_order=True))
        .explode("from_loc_id")
        .with_columns(pl.int_range(pl.len()).over("person_id").alias("sequence_num"))
        .to_dummies("from_loc_id")
        .with_columns(cs.starts_with("from_loc_id").cum_sum().over("person_id", order_by="sequence_num"))
    )

    targets = features.with_columns(pl.col("sequence_num") - 1)
    dataset_df = features.join(targets, on=["person_id", "sequence_num"], suffix="_target").sort(
        by=["person_id", "sequence_num"]
    )

    person_ids = dataset_df["person_id"]
    X = dataset_df.select(pl.col("sequence_num"), cs.starts_with("from_loc_id") & ~cs.ends_with("_target")).to_torch()
    y = dataset_df.select(cs.ends_with("_target")).to_torch()

    return BasicLocationsDataset(person_ids, pyg_graph, X, y)


def train_test_split(
    dataset: BasicLocationsDataset, test_size=0.15, random_state: int = None
) -> tuple[BasicLocationsDataset, BasicLocationsDataset]:
    """Performs a train-test split using the `person_id`s so that no person is split across the train and test set.

    Args:
        dataset (BasicLocationsDataset): the dataset to perform the split on
        test_size (float, optional): the proportion of samples in the test set. Defaults to 0.15.
        random_state (int, optional): random state for the RNG. Defaults to None.

    Returns:
        tuple[BasicLocationsDataset, BasicLocationsDataset]: A tuple of (train_set, test_set)
    """
    groups = dataset.person_ids()
    train_idx, test_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state).split(dataset, groups=groups)
    )

    train_set = dataset[train_idx]
    test_set = dataset[test_idx]

    return train_set, test_set
