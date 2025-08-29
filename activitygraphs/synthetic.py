from functools import cached_property
from typing import Optional

import networkx as nx
import numpy as np
import polars as pl
import polars.selectors as cs


def _set_binary_graph_attribute(G: nx.Graph, included_nodes: np.ndarray, attr_name: str):
    nx.set_node_attributes(G, {node: node in included_nodes for node in G.nodes}, attr_name)


def _shifted(name, prefix="to_"):
    return pl.col(name).shift(-1).alias(prefix + name)


def _select_closest_from_choice(
    nodes: np.ndarray, distances: np.ndarray, choices_idx: np.ndarray, valid: np.ndarray, exclude_chosen: bool = False
):
    """Given a list of chosen nodes (1 per person in the sample), selects the closest node from the list of valid nodes

    Args:
        nodes (np.ndarray): list of nodes in the graph
        distances (np.ndarray): distance matrix between each pair of nodes
        choices_idx (np.ndarray): indices of the chosen node for each person in the sample
        valid (np.ndarray): list of nodes that can be selected
        exclude_chosen (bool, optional): if true, disallows selecting the chosen node itself. Defaults to False.

    Returns:
        np.ndarray: a list of the selected (closest) node for each chosen node
    """
    is_valid_mask = np.isin(nodes, valid)
    masked_distances = np.where(is_valid_mask, distances[choices_idx], np.inf)

    if exclude_chosen:
        all_rows = np.arange(masked_distances.shape[0])
        masked_distances[all_rows, choices_idx] = np.inf

    closest_nodes_idx = np.argmin(masked_distances, axis=1)
    return nodes[closest_nodes_idx]


class SyntheticGraph:
    """Represents a transport network, with physical locations as nodes and links between them as edges"""

    WEIGHT_NAME = "distance"

    def __init__(
        self,
        edges: dict[tuple[str, str], int] | list[tuple[str, str]],
        workplace_nodes: np.ndarray,
        shopping_nodes: np.ndarray,
        home_nodes: Optional[np.ndarray] = None,
    ):
        """
        Args:
            edges (dict[tuple[str, str], int] | list[tuple[str, str]]):
                Dictionary mapping edges to the distances between them or list of edges where distances
                are assumed to be 1.
            workplace_nodes (list[str]):
                List of nodes that are potential workplaces.
            shopping_nodes (list[str]):
                List of nodes that are potential shopping destinations.
            home_nodes (Optional[list[str]], optional):
                List of nodes that are available home locations. If not specified, all nodes are considered
                available. Defaults to None.
        """
        if isinstance(edges, list):
            edges = {edge: 1 for edge in edges}

        self._G = nx.Graph()
        self._G.add_weighted_edges_from(((u, v, w) for ((u, v), w) in edges.items()), weight=self.WEIGHT_NAME)

        self.edges = edges
        self.nodes = np.array(list(self.G.nodes()))

        if home_nodes is None:
            home_nodes = self.nodes

        self.home_nodes = np.array(home_nodes)
        self.workplace_nodes = np.array(workplace_nodes)
        self.shopping_nodes = np.array(shopping_nodes)

        _set_binary_graph_attribute(self._G, self.home_nodes, "is_home")
        _set_binary_graph_attribute(self._G, self.workplace_nodes, "is_workplace")
        _set_binary_graph_attribute(self._G, self.shopping_nodes, "is_shopping")

    def __repr__(self):
        return f"SyntheticGraph({len(self.nodes)} nodes, {len(self.edges)} edges)"

    def with_home_nodes(self, home_nodes: np.ndarray) -> "SyntheticGraph":
        """Returns a new SyntheticGraph with updated home nodes"""
        return SyntheticGraph(self.edges, self.workplace_nodes, self.shopping_nodes, home_nodes)

    def with_workplace_nodes(self, workplace_nodes: np.ndarray) -> "SyntheticGraph":
        """Returns a new SyntheticGraph with updated workplace nodes"""
        return SyntheticGraph(self.edges, workplace_nodes, self.shopping_nodes, self.home_nodes)

    def with_shopping_nodes(self, shopping_nodes: np.ndarray) -> "SyntheticGraph":
        """Returns a new SyntheticGraph with updated shopping nodes"""
        return SyntheticGraph(self.edges, self.workplace_nodes, shopping_nodes, self.home_nodes)

    @cached_property
    def G(self) -> nx.Graph:
        """Returns the NetworkX graph, with only existing links connected"""
        return self._G

    @cached_property
    def G_full(self) -> nx.Graph:
        """Returns the fully connected NetworkX graph.

        Any two nodes are connected with an edge whose distance is the shortest path distance between the two nodes.
        """
        G_full = nx.from_numpy_array(self.distance_matrix, edge_attr="distance", nodelist=self.nodes)
        nx.set_node_attributes(G_full, dict(self._G.nodes(data=True)))
        return G_full

    @cached_property
    def node_ordering(self):
        """Returns a consistent ordering for the nodes in the graph"""
        return dict((node, idx) for idx, node in enumerate(self.nodes))

    @cached_property
    def distance_matrix(self):
        """Returns an array of the distances between nodes in the graph"""
        matrix = np.empty((len(self.nodes), len(self.nodes)), dtype=np.float32)
        distances = nx.shortest_path_length(self._G, weight="distance")

        for node, distance in distances:
            idx = self.node_ordering[node]
            row_items = sorted(distance.items(), key=lambda x: self.node_ordering[x[0]])
            row = [dist for _, dist in row_items]

            matrix[idx, :] = row

        return matrix

    @cached_property
    def distance_matrix_df(self):
        """Returns a polars DataFrame of the distances between nodes in the graph, in long form"""
        return (
            pl.DataFrame(self.distance_matrix, schema=list(self.nodes))
            .with_columns(from_loc_id=self.nodes.tolist())
            .unpivot(index="from_loc_id", variable_name="to_loc_id", value_name="distance")
        )


class SyntheticSchedules:
    """A class that represents a synthetic dataset of a population of agents and their schedules."""

    def __init__(
        self, n_samples: int, graph: SyntheticGraph, person_choices_df: pl.DataFrame | None, schedule_df: pl.DataFrame
    ):
        self.n_samples = n_samples
        self.graph = graph
        self.person_choices_df = person_choices_df
        self.schedule_df = schedule_df

        distances_df = graph.distance_matrix_df

        trip_df = (
            self.schedule_df.sort(by=["person_id", "sequence_num"])
            .with_columns(_shifted("loc_id"), _shifted("type"), _shifted("person_id"))
            .filter(pl.col("person_id") == pl.col("to_person_id"))
            .drop("to_person_id")
            .rename({"type": "from_type", "loc_id": "from_loc_id"})
        )

        self.trip_df = trip_df.join(distances_df, on=["from_loc_id", "to_loc_id"]).sort(
            by=["person_id", "sequence_num"]
        )


class SyntheticGenerator:
    """A class for generating synthetic populations and schedules on a SyntheticGraph"""

    AVAILABLE_SCHEDULES = np.array([
        ["W", "-", "-"],
        ["S1", "-", "-"],
        ["S2", "-", "-"],
        ["W", "S1", "-"],
        ["W", "S2", "-"],
        ["S1", "W", "-"],
        ["S1", "S2", "-"],
        ["S2", "W", "-"],
        ["S2", "S1", "-"],
        # ["W", "S1", "S2"],
        ["W", "S2", "S1"],
        ["S1", "W", "S2"],
        ["S1", "S2", "W"],
        ["S2", "W", "S1"],
        # ["S2", "S1", "W"],
    ])

    def __init__(self, graph: SyntheticGraph, rng: np.random.Generator = None):
        """_summary_

        Args:
            graph (SyntheticGraph): the graph on which to generate the data.
            rng (np.random.Generator, optional): the numpy random number generator, creates a new
            `numpy.random.default_rng` if None. Defaults to None.
        """
        self._graph = graph
        self._rng = rng if rng is not None else np.random.default_rng()

        self._n_samples = None
        self._person_choices_df = None
        self._schedule_df = None

    @property
    def person_choices_df(self):
        """Returns a polars DataFrame of the population and chosen locations

        Raises:
            LookupError: If the population has not been generated yet
        """
        if self._person_choices_df is None:
            raise LookupError("The population has not been generated yet. Please call `generate_population`.")

        return self._person_choices_df

    @property
    def n_samples(self):
        """Returns the number of samples in the population

        Raises:
            LookupError: If the population has not been generated yet
        """
        if self._n_samples is None:
            raise LookupError("The population has not been generated yet. Please call `generate_population`.")

        return self._n_samples

    @property
    def schedule_df(self):
        """Returns a polars DataFrame of the schedules for each person in the population

        Raises:
            LookupError: If the schedules have not been generated yet
        """
        if self._schedule_df is None:
            raise LookupError("The schedules have not been generated yet. Please call `generate_schedules`.")

        return self._schedule_df

    def generate_population(self, n_samples: int, exclude_chosen_from_shopping: bool):
        """Generates a population that lives on the graph, with a home, a workplace, and two shopping destinations.

        Args:
            n_samples (int): the number of samples (people) to generate.
            exclude_chosen_from_shopping (bool): if true, disallows shopping at home or work
            `numpy.random.default_rng` if None. Defaults to None.
        """
        home_choice_idx = self._rng.integers(0, len(self._graph.home_nodes), size=n_samples)
        home_choice = np.array(self._graph.home_nodes)[home_choice_idx]

        work_choice_idx = self._rng.integers(0, len(self._graph.workplace_nodes), size=n_samples)
        work_choice = np.array(self._graph.workplace_nodes)[work_choice_idx]

        closest_home_shopping = _select_closest_from_choice(
            self._graph.nodes,
            self._graph.distance_matrix,
            choices_idx=home_choice_idx,
            valid=self._graph.shopping_nodes,
            exclude_chosen=exclude_chosen_from_shopping,
        )
        closest_work_shopping = _select_closest_from_choice(
            self._graph.nodes,
            self._graph.distance_matrix,
            choices_idx=work_choice_idx,
            valid=self._graph.shopping_nodes,
            exclude_chosen=exclude_chosen_from_shopping,
        )

        self._n_samples = n_samples
        self._person_choices_df = pl.concat([
            pl.DataFrame({
                "type": "H",
                "loc_id": home_choice,
            }).with_row_index("person_id"),
            pl.DataFrame({
                "type": "W",
                "loc_id": work_choice,
            }).with_row_index("person_id"),
            pl.DataFrame({
                "type": "S1",
                "loc_id": closest_home_shopping,
            }).with_row_index("person_id"),
            pl.DataFrame({
                "type": "S2",
                "loc_id": closest_work_shopping,
            }).with_row_index("person_id"),
        ]).sort(by="person_id")

    def generate_schedules(self):
        """Generates schedules for the population.

        Raises:
            LookupError: if the population has not been generated yet
        """
        n_samples = self.n_samples

        available_schedules = self.AVAILABLE_SCHEDULES

        chosen_schedules = self._rng.integers(low=0, high=len(available_schedules), size=n_samples)
        schedules = available_schedules[chosen_schedules]
        home_col = np.repeat("H", n_samples).reshape(n_samples, 1)
        schedules = np.hstack([home_col, schedules, home_col])

        self._schedule_df = (
            pl.DataFrame(schedules, schema=["1", "2", "3", "4", "5"])
            .with_row_index("person_id")
            .unpivot(index="person_id", variable_name="numpy_seq", value_name="type")
            .sort(by=["person_id", "numpy_seq"])
            .filter(pl.col("type") != "-")
            .with_columns(pl.int_range(pl.len()).over("person_id", order_by="numpy_seq").alias("sequence_num"))
            .drop("numpy_seq")
            .join(self.person_choices_df, on=["person_id", "type"])
            .select("person_id", "sequence_num", "type", "loc_id")
            .sort(by=["person_id", "sequence_num"])
        )

    def build(self) -> SyntheticSchedules:
        return SyntheticSchedules(self.n_samples, self._graph, self.person_choices_df, self.schedule_df)


def compute_all_possible_schedules(graph: SyntheticGraph, available_schedules: np.ndarray) -> pl.DataFrame:
    """From a SyntheticGraph & associated SyntheticGenerator, compute all possible schedules / trips that
    can form on the graph

    Args:
        graph (SyntheticGraph)
        available_schedules (np.ndarray)

    Returns:
        pl.DataFrame: the schedules
    """

    home_choice_idx = np.repeat(range(len(graph.home_nodes)), len(graph.workplace_nodes))
    work_choice_idx = np.tile(range(len(graph.workplace_nodes)), len(graph.home_nodes))

    home_choice = graph.home_nodes[home_choice_idx]
    work_choice = graph.workplace_nodes[work_choice_idx]

    closest_home_shopping = _select_closest_from_choice(
        graph.nodes,
        graph.distance_matrix,
        choices_idx=home_choice_idx,
        valid=graph.shopping_nodes,
        exclude_chosen=True,
    )
    closest_work_shopping = _select_closest_from_choice(
        graph.nodes,
        graph.distance_matrix,
        choices_idx=work_choice_idx,
        valid=graph.shopping_nodes,
        exclude_chosen=True,
    )

    person_choices = np.stack([
        home_choice,
        work_choice,
        closest_home_shopping,
        closest_work_shopping,
    ]).T

    choices_repeated = np.repeat(person_choices, len(available_schedules), axis=0)
    home_choice_repeated = choices_repeated[:, 0].reshape(-1, 1)
    scheds = np.tile(available_schedules.T, len(person_choices)).T

    for i, activity in enumerate(["H", "W", "S1", "S2"]):
        chosen_locs_repeated = choices_repeated[:, i].reshape(-1, 1)
        scheds = np.where(scheds == activity, chosen_locs_repeated, scheds)

    scheds = np.concat([home_choice_repeated, scheds, home_choice_repeated], axis=1)

    # Move all "-" to the left of the array, see https://stackoverflow.com/a/43011036
    valid_mask = (scheds != "-")
    flipped_mask = valid_mask.sum(axis=1, keepdims=1) > np.arange(scheds.shape[1] - 1, -1, -1)
    flipped_mask = flipped_mask[:, ::-1]

    scheds[flipped_mask] = scheds[valid_mask]
    scheds[~flipped_mask] = "-"

    schedule_df = (
        pl.DataFrame(scheds, schema=["1", "2", "3", "4", "5"], orient="row")
        .with_row_index("person_id")
        .unpivot(index="person_id", variable_name="numpy_seq", value_name="loc_id")
        .sort(by=["person_id", "numpy_seq"])
        .filter(pl.col("loc_id") != "-")
        .with_columns(
            pl.int_range(pl.len()).over("person_id", order_by="numpy_seq").alias("sequence_num"),
            pl.lit("-").alias("type"),
        )
        .drop("numpy_seq")
    )

    s = SyntheticSchedules(0, graph, None, schedule_df)
    features = (
        s.trip_df.group_by("person_id")
        .agg(pl.col("from_loc_id").unique(maintain_order=True))
        .explode("from_loc_id")
        .with_columns(pl.int_range(pl.len()).over("person_id").alias("sequence_num"))
        .to_dummies("from_loc_id")
        .with_columns(cs.starts_with("from_loc_id").cum_sum().over("person_id", order_by="sequence_num"))
    )
    targets = features.with_columns(pl.col("sequence_num") - 1).rename(lambda col: col.replace("from_", "to_"))
    dataset_df = features.join(targets, on=["person_id", "sequence_num"]).sort(by=["person_id", "sequence_num"])

    return dataset_df
