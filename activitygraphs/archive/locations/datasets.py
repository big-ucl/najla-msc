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
        """
        Description: Returns the primary (possibly pruned or simplified) NetworkX graph
        representing the transport network. Concrete implementations such as SyntheticGraph
        use this to expose the graph that is actually used for ML training and evaluation,
        as opposed to the fully-connected version accessible via G_full.

        Output:
          - (nx.Graph): The NetworkX graph used for model input, where nodes are locations
                and edges are connections between them (e.g. travel links with distance weights).
        """
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
        person_features: torch.Tensor,
        home_node_labels: torch.Tensor,
        node_labels: torch.Tensor,
    ):
        """
        Args:
            data (Data): The underlying graph without the schedules with V nodes
            person_ids (torch.Tensor): A tensor of all person_ids in the dataset, shape=(N, 1)
            incomes (torch.Tensor): A tensor of income by person in the dataset, shape=(N, F)
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
        self._graph_x = check_shape(person_features, (n_samples, -1))
        self._x_features = check_shape(data.x, (n_nodes, n_features))
        self._x_labels = check_shape(home_node_labels, (n_samples, n_nodes, self._n_labels))
        self._y_labels = check_shape(node_labels, (n_samples, n_nodes, self._n_labels))

    def len(self):
        """
        Description: Returns the total number of samples (persons) in this dataset.

        Output:
          - (int): The number of persons stored in the dataset.
        """
        return len(self._person_ids)

    @property
    def num_classes(self) -> int:
        """
        Description: Returns the number of label classes per node (e.g. number of visit types such as Home,
        Work, Shopping). This is the size of the last dimension of the node label tensors.

        Output:
          - (int): Number of distinct label classes used for node classification.
        """
        return self._n_labels

    @property
    def num_graph_features(self) -> int:
        """
        Description: Returns the number of graph-level (person-level) features, such as income or
        demographic attributes. This is the width of the person feature matrix stored in _graph_x.

        Output:
          - (int): Number of person-level feature dimensions.
        """
        return self._graph_x.shape[1]

    def person_ids(self) -> list:
        """
        Description: Returns the list of person IDs for the samples currently selected by this
        dataset view (e.g. after a train/test split). Uses PyG's `indices()` to respect any
        subset selection.

        Output:
          - (list): A flat Python list of integer person IDs.
        """
        return self._person_ids[self.indices()].squeeze().tolist()

    def class_weights(self):
        """
        Description: Computes per-class positive weights to handle class imbalance during training.
        For each label class, the weight is calculated as (number of negatives) / (number of positives),
        so that rare positive examples are upweighted in the loss function.

        Output:
          - (torch.Tensor): A 1D tensor of shape (num_classes,) with the positive weight for
            each label class. Used as `pos_weight` in binary cross-entropy loss functions.
        """
        y_labels = self._y_labels[self.indices()]  # Restrict to the current dataset split
        n_examples = y_labels.shape[0] * y_labels.shape[1]  # Total number of (sample, node) pairs
        n_pos_examples = y_labels.flatten(end_dim=1).sum(dim=0)  # Count positive labels per class
        n_neg_examples = n_examples - n_pos_examples  # Count negative labels per class
        return n_neg_examples / n_pos_examples  # Ratio: upweights rare positive classes

    def get(self, idx):
        """
        Description: Retrieves one data sample (graph) from the dataset for a given index.
        Constructs two versions of the node features:
          - x (used during training): node features concatenated with the FULL ground-truth labels
            (all visit types visible, so the model learns to complete them).
          - x_infer (used during inference): node features concatenated with only HOME labels
            (simulates prediction where only the home is known).

        Input:
          - idx (int): The integer index of the sample to retrieve.

        Output:
          - (torch_geometric.data.Data): A PyG Data object with the following fields:
              x          : node features + full labels (for training), shape=(V, F+L)
              x_infer    : node features + home-only labels (for inference), shape=(V, F+L)
              edge_index : graph connectivity, shape=(2, E)
              edge_attr  : edge weights/features, shape=(E, A)
              y          : ground-truth node labels, shape=(V, L)
              graph_x    : person-level (graph-level) features, shape=(1, P)
              x_features : raw node features without labels, shape=(V, F)
              x_labels   : home-only node labels (input), shape=(V, L)
              person_id  : the person ID for this sample, shape=(1,)
        """
        # Concatenate node features with all ground-truth labels — used as input during training
        x_train = torch.concat([self._x_features, self._y_labels[idx]], dim=1)
        # Concatenate node features with home-only labels — used as input during inference/prediction
        x_infer = torch.concat([self._x_features, self._x_labels[idx]], dim=1)

        return Data(
            x=x_train,
            x_infer=x_infer,
            edge_index=self._edge_index,
            edge_attr=self._edge_attr,
            y=self._y_labels[idx],
            graph_x=self._graph_x[[idx]],
            x_features=self._x_features,
            x_labels=self._x_labels[idx],
            person_id=self._person_ids[idx],
        )


def _encode_node_visits(schedules: Schedules, only_encode_labels: list[str] | None = None) -> torch.Tensor:
    """
    Description: Encodes which nodes (locations) were visited by each person as a binary tensor,
    WITHOUT distinguishing visit types (reason of visit). For each person and each node, the value
    is 1 if the person visited that node at least once (for any of the filtered visit types), and
    0 otherwise. This is the simpler encoding that ignores WHY a node was visited.

    Input:
      - schedules (Schedules): A schedules object containing schedule_df, graph, and n_samples.
      - only_encode_labels (list[str] | None): If provided, only visits with these activity types
        (e.g. ['H'] for home visits) will be encoded. If None, all visit types are encoded.

    Output:
      - (torch.Tensor): A binary tensor of shape (N, V, 1), where N is the number of persons,
        V is the number of nodes in the graph, and 1 is a single binary channel indicating whether
        the person visited that node.
    """
    # Filter the schedule to only include the relevant visit types if specified
    schedules_df = (
        schedules.schedule_df
        if only_encode_labels is None
        else schedules.schedule_df.filter(pl.col("type").is_in(only_encode_labels))
    )

    nodes = schedules.graph.nodes  # All nodes in the transport network graph
    included_nodes = schedules_df["loc_id"].unique()  # Nodes that appear in the (filtered) schedule

    # Column names expected in the one-hot encoded output (one per node)
    dummy_columns = [f"loc_id_{node}" for node in nodes]
    # For nodes with zero visits, add a column of zeros so all nodes are always represented
    missing_columns = [pl.lit(0).alias(f"loc_id_{node}") for node in nodes if node not in included_nodes]

    # One-hot encode the visited location column, then aggregate per person using bitwise OR (visited = 1)
    dummies = schedules_df.to_dummies("loc_id").with_columns(*missing_columns).select("person_id", *dummy_columns)
    node_labels_df = dummies.group_by("person_id").agg(cs.starts_with("loc_id").bitwise_or()).sort(by="person_id")

    # Convert to PyTorch tensors
    node_labels = node_labels_df.select(cs.starts_with("loc_id")).to_torch()
    # Reshape to (N_samples, N_nodes, 1) — one binary channel per node per person
    node_labels = node_labels.reshape(schedules.n_samples, len(schedules.graph.nodes), 1)

    return node_labels


def _encode_node_labels(schedules: Schedules, only_encode_labels: list[str] | None = None) -> torch.Tensor:
    """
    Description: Encodes which nodes (locations) were visited by each person as a multi-label tensor,
    WITH visit-type information. For each person, each node, and each visit type (e.g. Home, Work,
    Shopping), the value is 1 if the person visited that node for that purpose, and 0 otherwise.
    This is the richer encoding that captures WHY (for which activity type) each node was visited.

    Input:
      - schedules (Schedules): A schedules object containing schedule_df, graph, visit_types,
        and n_samples.
      - only_encode_labels (list[str] | None): If provided, only visits with these activity types
        will be encoded as 1 (others are forced to 0). If None, all visit types are encoded.

    Output:
      - (torch.Tensor): A binary tensor of shape (N, V, L), where N is the number of persons,
        V is the number of nodes in the graph, and L is the number of visit/label types.
        Each entry is 1 if person n visited node v for activity type l, else 0.
    """
    nodes = schedules.graph.nodes  # All nodes/locations in the graph
    labels = schedules.visit_types  # All activity types found in the schedule (e.g. H, W, S1, S2)
    # If no label filter is provided, encode all activity types
    only_encode_labels = labels if only_encode_labels is None else only_encode_labels

    # Build one expression per (node, label) pair that encodes whether the visit occurred
    expressions = []
    for node, label in itertools.product(nodes, labels):
        # The expression is 1 if the person visited this node for this activity type, 0 otherwise
        # If the label is NOT in only_encode_labels, the expression is always 0 (masked out)
        expression = (
            pl.when(pl.col("loc_id") == node, pl.col("type") == label).then(1).otherwise(0)
            if label in only_encode_labels
            else pl.lit(0)
        )

        expressions.append(expression.alias(f"loc_id_{node}_[{label}]"))

    # Compute the indicator for every row, then aggregate per person using bitwise OR
    dummies = schedules.schedule_df.select("person_id", *expressions)
    node_labels_df = dummies.group_by("person_id").agg(cs.starts_with("loc_id").bitwise_or()).sort(by="person_id")

    # Convert to PyTorch tensors
    node_labels = node_labels_df.drop("person_id").to_torch()
    # Reshape to (N_samples, N_nodes, N_labels) — one binary channel per node per activity type per person
    node_labels = node_labels.reshape(schedules.n_samples, len(schedules.graph.nodes), len(schedules.visit_types))

    return node_labels


def convert_to_pyg_dataset(
    schedules: Schedules, label_reason_of_visit: bool = True, categorical_person_features: list[str] = None
) -> Dataset:
    """Converts a population schedule object into a PyG Dataset"""

    categorical_person_features = [] if categorical_person_features is None else categorical_person_features

    pyg_graph = from_networkx(
        schedules.graph.G,
        group_node_attrs=["is_home", "is_workplace", "is_shopping", "home_price"],
        group_edge_attrs=[schedules.graph.WEIGHT_NAME],
    )

    person_df = schedules.person_df.sort(by="person_id")
    person_ids = person_df["person_id"].to_torch().unsqueeze(1)
    person_features_df = person_df.drop("person_id").to_dummies(categorical_person_features, drop_first=True)
    person_features = person_features_df.to_torch()

    encoder = _encode_node_labels if label_reason_of_visit else _encode_node_visits
    home_node_labels = encoder(schedules, only_encode_labels=["H"])
    node_labels = encoder(schedules)

    return BasicLocationsDataset(
        pyg_graph, person_ids.float(), person_features.float(), home_node_labels.float(), node_labels.float()
    )


class HasPersonIDs(Protocol):
    """
    Description: A structural Protocol (interface) that specifies any object exposing a
    `person_ids()` method returning a list. This is used as a type constraint so that
    functions like `train_test_split` can work with any dataset that tracks person IDs,
    regardless of the concrete class.
    """
    def person_ids(self) -> list:
        """
        Description: Returns the list of person IDs for all samples in this dataset.

        Output:
          - (list): A list of person identifiers (typically integers).
        """
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
