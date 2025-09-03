import itertools
from typing import Generic, Protocol, TypeVar

import networkx as nx
import numpy as np
import polars as pl
import polars.selectors as cs
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.utils import from_networkx
from utils import check_shape


class Graph(Protocol):
    """A protocol to represent an input graph such as a SyntheticGraph"""

    WEIGHT_NAME: str
    nodes: np.ndarray[str]

    @property
    def G(self) -> nx.Graph:
        pass


class Schedules(Protocol):
    """A protocol to represent a schedules object (e.g. SyntheticSchedules)"""

    n_samples: int
    graph: Graph
    person_df: pl.DataFrame
    person_choices_df: pl.DataFrame
    schedule_df: pl.DataFrame
    trip_df: pl.DataFrame
    visit_types: list[str]


class BasicLocationsDataset(InMemoryDataset):
    """Represents a PyG dataset that contains the training data for a graph ML model"""

    def __init__(
        self,
        data: Data,
        person_ids: torch.Tensor,
        incomes: torch.Tensor,
        home_node_labels: torch.Tensor,
        node_labels: torch.Tensor,
    ):
        """
        Args:
            data (Data): The underlying graph without the schedules with V nodes
            person_ids (torch.Tensor): A tensor of all person_ids in the dataset, shape=(N, 1)
            incomes (torch.Tensor): A tensor of income by person in the dataset, shape=(N, 1)
            home_node_labels (torch.Tensor):
                A tensor of one-hot encoded node labels, with only home visits labeled, shape=(N, V, L)
            node_labels (torch.Tensor):
                A tensor of one-hot encoded node labels, with all visits labeled, shape=(N, V, L)
        """
        super().__init__()

        n_nodes = home_node_labels.shape[1]
        n_samples = person_ids.shape[0]
        n_features = data.num_features
        self._n_labels = home_node_labels.shape[2]

        if n_nodes != data.num_nodes:
            raise ValueError(f"Invalid data, got {data.num_nodes} nodes, expected {n_nodes}")

        self._data = data
        self._edge_index = data.edge_index
        self._edge_attr = data.edge_attr
        self._person_ids = check_shape(person_ids, (n_samples, 1))
        self._graph_x = check_shape(incomes, (n_samples, -1))
        self._x_features = check_shape(data.x, (n_nodes, n_features))
        self._x_labels = check_shape(home_node_labels, (n_samples, n_nodes, self._n_labels))
        self._y_labels = check_shape(node_labels, (n_samples, n_nodes, self._n_labels))

    def len(self):
        return len(self._person_ids)

    @property
    def num_classes(self) -> int:
        return self._n_labels

    def person_ids(self) -> list:
        return self._person_ids[self.indices()].squeeze().tolist()

    def get(self, idx):
        x = torch.concat([self._x_features, self._x_labels[idx]], dim=1)

        return Data(
            x=x,
            edge_index=self._edge_index,
            edge_attr=self._edge_attr,
            y=self._y_labels[idx],
            graph_x=self._graph_x[idx],
            x_labels=self._x_labels[idx],
            person_id=self._person_ids[idx],
        )


def _encode_node_visits(schedules: Schedules, only_encode_labels: list[str] | None = None) -> torch.Tensor:
    schedules_df = (
        schedules.schedule_df
        if only_encode_labels is None
        else schedules.schedule_df.filter(pl.col("type").is_in(only_encode_labels))
    )

    dummies = schedules_df.to_dummies("loc_id")
    node_labels_df = dummies.group_by("person_id").agg(cs.starts_with("loc_id").bitwise_or()).sort(by="person_id")

    # Convert to PyTorch tensors
    node_labels = node_labels_df.select(cs.starts_with("loc_id")).to_torch()
    node_labels = node_labels.reshape(schedules.n_samples, len(schedules.graph.nodes), 1)

    return node_labels


def _encode_node_labels(schedules: Schedules, only_encode_labels: list[str] | None = None) -> torch.Tensor:
    nodes = schedules.graph.nodes
    labels = schedules.visit_types
    only_encode_labels = labels if only_encode_labels is None else only_encode_labels

    expressions = []
    for node, label in itertools.product(nodes, labels):
        expression = (
            pl.when(pl.col("loc_id") == node, pl.col("type") == label).then(1).otherwise(0)
            if label in only_encode_labels
            else pl.lit(0)
        )

        expressions.append(expression.alias(f"loc_id_{node}_[{label}]"))

    dummies = schedules.schedule_df.select("person_id", *expressions)
    node_labels_df = dummies.group_by("person_id").agg(cs.starts_with("loc_id").bitwise_or()).sort(by="person_id")

    # Convert to PyTorch tensors
    node_labels = node_labels_df.drop("person_id").to_torch()
    node_labels = node_labels.reshape(schedules.n_samples, len(schedules.graph.nodes), len(schedules.visit_types))

    return node_labels


def convert_to_pyg_dataset(schedules: Schedules, label_reason_of_visit: bool = True) -> Dataset:
    """Converts a population schedule object into a PyG Dataset"""

    pyg_graph = from_networkx(
        schedules.graph.G,
        group_node_attrs=["is_home", "is_workplace", "is_shopping", "home_price"],
        group_edge_attrs=[schedules.graph.WEIGHT_NAME],
    )

    person_ids = schedules.person_df.sort(by="person_id")["person_id"].to_torch().unsqueeze(1)
    incomes = schedules.person_df.sort(by="person_id")["income"].to_torch().unsqueeze(1)

    encoder = _encode_node_labels if label_reason_of_visit else _encode_node_visits
    home_node_labels = encoder(schedules, only_encode_labels=["H"])
    node_labels = encoder(schedules)

    return BasicLocationsDataset(pyg_graph, person_ids, incomes, home_node_labels, node_labels)


class HasPersonIDs(Protocol):
    def person_ids(self) -> list:
        pass


T_HasPersonIDs = TypeVar("T_HasPersonIDs", Dataset, HasPersonIDs)


def train_test_split(
    dataset: Generic[T_HasPersonIDs], test_size=0.15, random_state: int = None
) -> tuple[T_HasPersonIDs, T_HasPersonIDs]:
    """Performs a train-test split using the `person_id`s so that no person is split across the train and test set.

    Args:
        dataset (T : Dataset, HasPersonIDs): the dataset to perform the split on, implements `HasPersonIDs`
        test_size (float, optional): the proportion of samples in the test set. Defaults to 0.15.
        random_state (int, optional): random state for the RNG. Defaults to None.

    Returns:
        tuple[T, T]: A tuple of (train_set, test_set)
    """
    groups = dataset.person_ids()
    train_idx, test_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state).split(dataset, groups=groups)
    )

    train_set = dataset[train_idx]
    test_set = dataset[test_idx]

    return train_set, test_set
