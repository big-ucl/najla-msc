from abc import ABC, abstractmethod
from functools import cached_property
from typing import Optional

import networkx as nx
import numpy as np
import polars as pl
import polars.selectors as cs


def _set_binary_graph_attribute(G: nx.Graph, included_nodes: np.ndarray, attr_name: str):
    """
    Description: Sets a binary (True/False) node attribute on every node in the graph.
    For each node, the attribute is True if the node is in `included_nodes`, and False otherwise.
    Used to mark nodes as home locations, workplaces, or shopping destinations.

    Input:
      - G (nx.Graph): The NetworkX graph on which to set the attribute.
      - included_nodes (np.ndarray): Array of node IDs that should receive the value True.
      - attr_name (str): The name of the node attribute to set (e.g. 'is_home', 'is_workplace').

    Output:
      - None. Modifies the graph G in place.
    """
    nx.set_node_attributes(G, {node: node in included_nodes for node in G.nodes}, attr_name)


def _shifted(name, prefix="to_"):
    """
    Description: Creates a Polars expression that shifts a column by -1 (one row backwards) and
    renames it with the given prefix. Used to create 'to_*' columns alongside 'from_*' columns
    when building a trip DataFrame from a sequence of schedule visits (e.g. 'to_loc_id' is the
    location that comes AFTER the current one in the schedule).

    Input:
      - name (str): The original column name (e.g. 'loc_id', 'type').
      - prefix (str): The prefix to prepend to the shifted column name. Defaults to 'to_'.

    Output:
      - (pl.Expr): A Polars expression representing the shifted column, renamed to '{prefix}{name}'.
    """
    return pl.col(name).shift(-1).alias(prefix + name)


def _select_closest_from_choice(
    nodes: np.ndarray,
    distances: np.ndarray,
    choices_idx: np.ndarray,
    valid: np.ndarray,
    exclude_chosen: bool = False,
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

    @classmethod
    def example(cls):
        """Returns an example synthetic graph"""
        edges = {
            ("A", "B"): 3,
            ("B", "C"): 6,
            ("C", "A"): 5,
            ("C", "D"): 15,
            ("D", "E"): 3,
            ("E", "F"): 7,
            ("E", "G"): 2,
            ("F", "G"): 4,
            ("F", "D"): 9,
            ("F", "H"): 10,
            ("H", "B"): 12,
            ("H", "C"): 16,
        }

        workplace_nodes = np.array(["A", "B", "C"])
        shopping_nodes = np.array(["B", "C", "D", "E"])

        return cls(edges, workplace_nodes, shopping_nodes)

    def __init__(
        self,
        edges: dict[tuple[str, str], int] | list[tuple[str, str]],
        workplace_nodes: np.ndarray,
        shopping_nodes: np.ndarray,
        home_nodes: Optional[np.ndarray] = None,
        home_prices: Optional[dict[str, float]] = None,
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
            home_nodes (list[str], optional):
                List of nodes that are available home locations. If not specified, all nodes are considered
                available. Defaults to None.
            home_prices (dict[str, float], optional):
                Prices of home nodes. If not specified, prices are 0 to N in alphabetical order. Defaults to None.
        """
        if isinstance(edges, list):
            edges = {edge: 1 for edge in edges}

        self._G = nx.Graph()
        self._G.add_weighted_edges_from(((u, v, w) for ((u, v), w) in edges.items()), weight=self.WEIGHT_NAME)

        self.edges = edges
        self.nodes = np.array(list(self.G.nodes()))

        if home_nodes is None:
            home_nodes = self.nodes

        if home_prices is None:
            home_prices = {u: p for u, p in zip(sorted(home_nodes), range(len(home_nodes)))}

        if set(home_prices.keys()) != set(home_nodes):
            raise ValueError(f"Home prices do not match home nodes. Got {home_prices.keys()}, expected {home_nodes}")

        self.home_nodes = np.array(home_nodes)
        self.workplace_nodes = np.array(workplace_nodes)
        self.shopping_nodes = np.array(shopping_nodes)
        self.home_prices = np.array([home_prices[u] for u in self.home_nodes])

        _set_binary_graph_attribute(self._G, self.home_nodes, "is_home")
        _set_binary_graph_attribute(self._G, self.workplace_nodes, "is_workplace")
        _set_binary_graph_attribute(self._G, self.shopping_nodes, "is_shopping")

        nx.set_node_attributes(self._G, home_prices, "home_price")

    def __repr__(self):
        """
        Description: Returns a concise human-readable representation of the SyntheticGraph,
        showing the number of nodes and edges.

        Output:
          - (str): A string like 'SyntheticGraph(8 nodes, 12 edges)'.
        """
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
        # TODO Change to full being the per-activity node-duplicated version
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
            pl
            .DataFrame(self.distance_matrix, schema=list(self.nodes))
            .with_columns(from_loc_id=pl.Series(self.nodes))
            .unpivot(index="from_loc_id", variable_name="to_loc_id", value_name="distance")
        )


class SyntheticSchedules:
    """A class that represents a synthetic dataset of a population of agents and their schedules."""

    def __init__(
        self,
        n_samples: int,
        graph: SyntheticGraph,
        person_df: pl.DataFrame | None,
        person_choices_df: pl.DataFrame | None,
        schedule_df: pl.DataFrame,
    ):
        """
        Description: Initialises the SyntheticSchedules object from pre-generated population and
        schedule data. Automatically constructs a trip DataFrame from consecutive schedule entries
        by pairing each visit with the next one (from_loc_id -> to_loc_id), and joins travel
        distances from the graph's distance matrix.

        Input:
          - n_samples (int): The number of persons (agents) in the population.
          - graph (SyntheticGraph): The transport network graph the population lives on.
          - person_df (pl.DataFrame | None): DataFrame of person attributes (e.g. income),
            one row per person. May be None if not yet generated.
          - person_choices_df (pl.DataFrame | None): DataFrame of each person's chosen locations
            (home, work, shopping), one row per (person, type). May be None if not yet generated.
          - schedule_df (pl.DataFrame): DataFrame of each person's activity sequence.
            Must contain columns: 'person_id', 'sequence_num', 'type', 'loc_id'.
        """
        self.n_samples = n_samples  # Total number of persons in the synthetic population
        self.graph = graph  # The transport network on which agents are placed
        self.person_df = person_df  # Personal attributes (e.g. income) per person
        self.person_choices_df = person_choices_df  # Chosen activity locations per person
        self.schedule_df = schedule_df  # Activity schedule: who visits where in what order
        # All unique activity types present in this schedule (e.g. ['H', 'W', 'S1', 'S2'])
        self.visit_types = schedule_df["type"].unique().to_list()

        distances_df = graph.distance_matrix_df  # Pairwise distances between all nodes (long format)

        # Build trip DataFrame: for each consecutive pair of visits, create one row with
        # from_loc_id, to_loc_id, from_type, to_type, and the distance travelled
        trip_df = (
            self.schedule_df
            .sort(by=["person_id", "sequence_num"])
            # Shift columns by -1 to get the NEXT visit in sequence alongside the current one
            .with_columns(_shifted("loc_id"), _shifted("type"), _shifted("person_id"))
            # Remove the last row for each person (shift introduces None at the boundary)
            .filter(pl.col("person_id") == pl.col("to_person_id"))
            .drop("to_person_id")
            .rename({"type": "from_type", "loc_id": "from_loc_id"})
        )

        # Join trip distance from the graph's distance matrix
        self.trip_df = trip_df.join(distances_df, on=["from_loc_id", "to_loc_id"]).sort(
            by=["person_id", "sequence_num"]
        )


class SyntheticGenerator(ABC):
    """
    Description: Abstract base class (ABC) for generating synthetic populations and activity
    schedules on a SyntheticGraph. Subclasses implement `generate_population` and
    `generate_schedules` with different behavioural models (probabilistic, deterministic, etc.).
    Provides shared internal methods for setting person choices and schedules. After calling
    `generate_population()` and `generate_schedules()`, call `build()` to get a SyntheticSchedules.
    """
    # All valid daily schedule patterns (as sequences of activity types, '-' means no activity).
    # Each row is one possible ordered schedule (e.g. ['W', '-', '-'] = just work, no shopping).
    # Schedules are later expanded by replacing activity codes with chosen location IDs.
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

    def __init__(self, graph: SyntheticGraph, rng: np.random.Generator):
        """
        Description: Initialises the generator with the transport network and a random number
        generator. All data attributes are initially None; they are populated by calling
        `generate_population()` and `generate_schedules()`.

        Input:
          - graph (SyntheticGraph): The transport network on which to generate the population.
          - rng (np.random.Generator): A numpy random number generator for reproducibility.
            If None, creates a new `np.random.default_rng()`.
        """
        self._graph = graph  # The transport network graph (nodes = locations, edges = distances)
        # Use the provided RNG or create a fresh one with a random seed
        self._rng = rng if rng is not None else np.random.default_rng()

        # These are populated by calling generate_population() and generate_schedules()
        self._n_samples = None  # Will hold the number of persons in the population
        self._person_df = None  # Will hold a DataFrame of person attributes (e.g. income)
        self._person_choices_df = None  # Will hold a DataFrame of each person's chosen locations
        self._schedule_df = None  # Will hold the full activity sequence for each person

    @property
    def person_df(self) -> pl.DataFrame:
        """Returns a polars DataFrame of the population and its characteristics"""
        if self._person_df is None:
            raise LookupError("The population has not yet been generated yet Please call `generate_population`.")

        return self._person_df

    @property
    def person_choices_df(self) -> pl.DataFrame:
        """Returns a polars DataFrame of the chosen locations for each person in the population

        Raises:
            LookupError: If the population has not been generated yet
        """
        if self._person_choices_df is None:
            raise LookupError("The population has not been generated yet. Please call `generate_population`.")

        return self._person_choices_df

    @property
    def n_samples(self) -> int:
        """Returns the number of samples in the population

        Raises:
            LookupError: If the population has not been generated yet
        """
        if self._n_samples is None:
            raise LookupError("The population has not been generated yet. Please call `generate_population`.")

        return self._n_samples

    @property
    def schedule_df(self) -> pl.DataFrame:
        """Returns a polars DataFrame of the schedules for each person in the population

        Raises:
            LookupError: If the schedules have not been generated yet
        """
        if self._schedule_df is None:
            raise LookupError("The schedules have not been generated yet. Please call `generate_schedules`.")

        return self._schedule_df

    @abstractmethod
    def generate_population(self, n_samples: int):
        """Generates a population that lives on the graph, with a home, a workplace, and two shopping destinations.

        Args:
            n_samples (int): the number of samples (people) to generate.
        """
        pass

    @abstractmethod
    def generate_schedules(self):
        """Generates schedules for the population.

        Raises:
            LookupError: if the population has not been generated yet
        """
        pass

    def build(self) -> SyntheticSchedules:
        """
        Description: Assembles and returns a SyntheticSchedules object from the generated
        population and schedule data. Must be called AFTER `generate_population()` and
        `generate_schedules()` have been called (or those methods will raise LookupError).

        Output:
          - (SyntheticSchedules): A complete dataset object containing the graph, all person
            attributes, location choices, and activity sequences.
        """
        return SyntheticSchedules(
            self.n_samples,
            self._graph,
            self.person_df,
            self.person_choices_df,
            self.schedule_df,
        )

    def _set_person_choices(self, homes, workplaces, home_shopping, work_shopping):
        """
        Description: Internal method that constructs and stores the person_choices_df DataFrame.
        This DataFrame records the chosen home, workplace, home-side shopping, and work-side shopping
        location for each person, labelled with their activity type ('H', 'W', 'S1', 'S2').

        Input:
          - homes (np.ndarray): Array of home location IDs, one per person.
          - workplaces (np.ndarray): Array of workplace location IDs, one per person.
          - home_shopping (np.ndarray): Array of home-side shopping location IDs, one per person.
          - work_shopping (np.ndarray): Array of work-side shopping location IDs, one per person.

        Output:
          - None. Stores the result in self._person_choices_df.
        """
        self._person_choices_df = pl.concat([
            pl.DataFrame({
                "type": "H",
                "loc_id": homes,
            }).with_row_index("person_id"),
            pl.DataFrame({
                "type": "W",
                "loc_id": workplaces,
            }).with_row_index("person_id"),
            pl.DataFrame({
                "type": "S1",
                "loc_id": home_shopping,
            }).with_row_index("person_id"),
            pl.DataFrame({
                "type": "S2",
                "loc_id": work_shopping,
            }).with_row_index("person_id"),
        ]).sort(by="person_id")

    def _set_schedules(self, chosen_schedules):
        """
        Description: Internal method that builds and stores the schedule_df DataFrame from an
        array of schedule indices. Each schedule is a sequence of activity codes (e.g. 'W', 'S1')
        that is bookended by home visits ('H') and expanded to include the actual chosen location
        IDs from person_choices_df. Placeholder '-' entries are filtered out.

        Input:
          - chosen_schedules (np.ndarray): Integer array of shape (n_samples,) where each value
            is an index into AVAILABLE_SCHEDULES, selecting which daily schedule pattern to use.

        Output:
          - None. Stores the result in self._schedule_df.
        """
        # Look up the actual schedule pattern rows from the AVAILABLE_SCHEDULES table
        schedules = self.AVAILABLE_SCHEDULES[chosen_schedules]
        # Prepend and append 'H' to every schedule: all persons start and end at home
        home_col = np.repeat("H", self.n_samples).reshape(self.n_samples, 1)
        schedules = np.hstack([home_col, schedules, home_col])

        self._schedule_df = (
            pl
            .DataFrame(schedules, schema=["1", "2", "3", "4", "5"])
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


class ProbabilisticSyntheticGenerator(SyntheticGenerator):
    """A class for generating synthetic populations and schedules on a SyntheticGraph"""

    def __init__(
        self,
        graph: SyntheticGraph,
        rng: np.random.Generator = None,
        distance_scale: float = 3.0,
        income_mu: float = 0,
        income_sigma: float = 0.8,
    ):
        """
        Description: Initialises the probabilistic generator with behavioural model parameters.
        This generator creates agents with lognormally-distributed incomes and selects their
        activity locations using probabilistic models: home and workplace are chosen based on
        income sensitivity (richer agents prefer more expensive/prestigious locations), while
        shopping locations are chosen based on inverse-distance probability (closer is more likely).

        Args:
            graph (SyntheticGraph): the graph on which to generate the data.
            rng (np.random.Generator, optional):
                the numpy random number generator, creates a new
                numpy.random.default_rng` if None. Defaults to None.
            distance_scale (float, optional):
                the scaling factor weighting distance from home/work in shopping
                probability calculation. Defaults to 3.0.
            income_mu (float, optional):
                the mu parameter of the lognormal income distribution. Defaults to 0.
            income_sigma (float, optional):
                the sigma parameter of the lognormal income distribution. Defaults to 0.8.
        """
        super().__init__(graph, rng)

        self._distance_scale = distance_scale  # Controls how strongly distance influences shopping choice
        self._income_mu = income_mu  # Log-mean of the lognormal income distribution
        self._income_sigma = income_sigma  # Log-std of the lognormal income distribution
        self._income_scale = 10  # Multiplicative scaling factor applied to raw income samples

    def _choice_with_incomes(
        self,
        nodes: np.ndarray,
        prices: np.ndarray,
        incomes: np.ndarray,
        n_samples: int,
        inverse: bool = True,
    ):
        """
        Description: Samples a location choice for each person using a logit model where
        income acts as the price sensitivity parameter. The probability of choosing a node
        is proportional to exp(-price / income): richer agents are less sensitive to high prices
        and are therefore more likely to choose expensive nodes. If inverse=True, the prices
        array is reversed so that higher indices are more expensive.

        Input:
          - nodes (np.ndarray): Array of candidate node IDs to choose from.
          - prices (np.ndarray): Array of prices for each node, same length as nodes.
          - incomes (np.ndarray): Array of income values, one per person (n_samples,).
          - n_samples (int): Number of persons to sample for.
          - inverse (bool): If True, reverse the price order (higher rank = more expensive).
            Defaults to True.

        Output:
          - (tuple[np.ndarray, np.ndarray]): A tuple of (choices_idx, choices) where:
              choices_idx : integer array of chosen node indices, shape=(n_samples,)
              choices     : array of chosen node IDs, shape=(n_samples,)
        """
        # Optionally reverse the prices so that the last node is the most expensive
        prices = prices[::-1] if inverse else prices
        # Compute logits: lower income → more negative logits for expensive nodes (more price sensitive)
        logits = (-1 / incomes.reshape(-1, 1)) * prices
        # Softmax to convert logits to probabilities (one distribution per person)
        probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

        choices_idx = np.empty(n_samples, dtype=int)  # Will store the index of the chosen node per person
        choices = np.empty(n_samples, dtype=str)  # Will store the node ID of the chosen node per person

        for i, p in enumerate(probabilities):
            choice_idx = self._rng.choice(range(len(nodes)), p=p)  # Sample one node index per person
            choices_idx[i] = choice_idx
            choices[i] = nodes[choice_idx]

        return choices_idx, choices

    def _choice_with_distance_probs(self, targets_idx: np.ndarray, valid: np.ndarray):
        """Given a list of target nodes (1 per person in the sample), samples a valid node based on inverse distance.

        Args:
            targets_idx (np.ndarray):
                indices of the target node from which to compute distance prob for each person in
                the sample
            valid (np.ndarray): list of nodes that can be selected

        Returns:
            np.ndarray: a list of the selected (closest) node for each chosen node
        """
        nodes = self._graph.nodes
        targets, counts = np.unique(targets_idx, return_counts=True)

        choices = np.empty(len(targets_idx), dtype=str)
        for target, count in zip(targets, counts):
            is_valid_node = np.ones(len(nodes), dtype=np.bool)
            is_valid_node = is_valid_node & np.isin(nodes, valid)
            is_valid_node[target] = False

            distance = self._graph.distance_matrix[target]
            distance[~is_valid_node] = 0.01  # To avoid zero division error

            inverse_distance = np.where(is_valid_node, 1 / distance, -np.inf)
            inverse_distance = inverse_distance * self._distance_scale

            exp = np.exp(inverse_distance)
            probabilities = exp / np.sum(exp, keepdims=True)

            choice = self._rng.choice(nodes, p=probabilities, size=count)
            choices[targets_idx == target] = choice

        return choices

    def generate_population(self, n_samples: int):
        """Generates a population that lives on the graph, with a home, a workplace, and two shopping destinations.

        Args:
            n_samples (int): the number of samples (people) to generate.
        """
        self._schedule_df = None  # Reset any previously generated schedules

        # Draw incomes from a lognormal distribution (scaled to realistic range)
        incomes = self._rng.lognormal(self._income_mu, self._income_sigma, size=n_samples) * self._income_scale
        # Use simple rank ordering [0, 1, 2, ...] as workplace "prices" (proxy for prestige)
        work_prices = np.arange(len(self._graph.workplace_nodes))

        # Choose home location: income-sensitive (richer agents choose more expensive homes)
        homes_idx, homes = self._choice_with_incomes(
            self._graph.home_nodes, self._graph.home_prices, incomes, n_samples
        )
        # Choose workplace: income-sensitive (richer agents choose more prestigious workplaces)
        workplaces_idx, workplaces = self._choice_with_incomes(
            self._graph.workplace_nodes, work_prices, incomes, n_samples
        )

        # Choose home-side shopping: distance-based (closer to home is more likely)
        home_shopping = self._choice_with_distance_probs(homes_idx, self._graph.shopping_nodes)
        # Choose work-side shopping: distance-based (closer to work is more likely)
        work_shopping = self._choice_with_distance_probs(workplaces_idx, self._graph.shopping_nodes)

        self._n_samples = n_samples  # Store the population size

        # Build and store the person attributes DataFrame (one row per person, with person_id index)
        self._person_df = pl.DataFrame({
            "income": incomes,
        }).with_row_index("person_id")

        self._set_person_choices(homes, workplaces, home_shopping, work_shopping)

    def generate_schedules(self):
        """Generates schedules for the population.

        Raises:
            LookupError: if the population has not been generated yet
        """
        chosen_schedules = self._rng.integers(low=0, high=len(self.AVAILABLE_SCHEDULES), size=self.n_samples)
        self._set_schedules(chosen_schedules)


class DeterministicSyntheticGenerator(SyntheticGenerator):
    """
    Description: A SyntheticGenerator that creates agents with deterministic binary behavioural
    rules rather than continuous probabilistic models. Each agent is randomly labelled as
    'rich' or 'not rich' and as 'shops first' or 'does not shop first', and their location
    choices are deterministically assigned based on these labels. This creates a simpler,
    more interpretable synthetic dataset where the ground-truth decision rule is known exactly.
    Useful for testing whether a model can learn these binary patterns.
    """
    def __init__(
        self, graph: SyntheticGraph, rng: np.random.Generator = None, p_is_rich: bool = 0.3, p_shop_first: bool = 0.2
    ):
        """
        Description: Initialises the deterministic generator with the two key behavioural
        probabilities controlling what fraction of agents are labelled as 'rich' or 'shops first'.

        Input:
          - graph (SyntheticGraph): The transport network on which to generate the population.
          - rng (np.random.Generator | None): Random number generator for reproducibility.
            Defaults to None (creates new default generator).
          - p_is_rich (float): Probability that any given agent is 'rich' (affects home and
            workplace choices). Defaults to 0.3.
          - p_shop_first (float): Probability that any given agent 'shops first' (affects which
            shopping nodes they choose). Defaults to 0.2.
        """
        super().__init__(graph, rng)

        self.p_is_rich = p_is_rich  # Fraction of population labelled as 'rich'
        self.p_shop_first = p_shop_first  # Fraction of population that shops before work

    def generate_population(self, n_samples: int):
        """
        Description: Generates a deterministic synthetic population. Each person is assigned:
          - is_rich: True with probability p_is_rich → chooses the last home node and first workplace.
          - shop_first: True with probability p_shop_first → chooses the last two shopping nodes.
          - chosen_schedule: A randomly selected schedule index from AVAILABLE_SCHEDULES.

        Input:
          - n_samples (int): Number of persons to generate.

        Output:
          - None. Stores person_df and person_choices_df in self.
        """
        p_is_rich = self.p_is_rich  # Local copy for clarity
        p_shop_first = self.p_shop_first  # Local copy for clarity
        n_schedules = len(self.AVAILABLE_SCHEDULES)  # Total number of available schedule types

        # Randomly assign binary wealth and shopping preference labels per person
        is_rich = self._rng.choice([True, False], size=n_samples, p=[p_is_rich, 1 - p_is_rich])
        shop_first = self._rng.choice([True, False], size=n_samples, p=[p_shop_first, 1 - p_shop_first])
        # Randomly assign a schedule type to each person
        chosen_schedule = self._rng.integers(low=0, high=n_schedules, size=n_samples)

        # Rich agents choose the last (most expensive) home node, non-rich choose the second-to-last
        homes = np.where(is_rich, self._graph.home_nodes[-1], self._graph.home_nodes[-2])
        # Rich agents choose the first (most prestigious) workplace, non-rich choose the last
        workplaces = np.where(is_rich, self._graph.workplace_nodes[0], self._graph.workplace_nodes[-1])
        # 'Shops first' agents choose the last shopping node near home, others choose second-to-last
        home_shopping = np.where(shop_first, self._graph.shopping_nodes[-1], self._graph.shopping_nodes[-2])
        # 'Shops first' agents choose the first shopping node near work, others choose the second
        work_shopping = np.where(shop_first, self._graph.shopping_nodes[0], self._graph.shopping_nodes[1])

        self._n_samples = n_samples  # Store population size
        # Build person attributes DataFrame with binary feature columns and a person_id index
        self._person_df = pl.DataFrame({
            "is_rich": is_rich,
            "shop_first": shop_first,
            "chosen_schedule": chosen_schedule,
        }).with_row_index("person_id")
        self._set_person_choices(homes, workplaces, home_shopping, work_shopping)

    def generate_schedules(self):
        """
        Description: Generates schedules for the population using the pre-assigned schedule
        indices stored in person_df. Must be called after `generate_population()`.

        Output:
          - None. Stores the schedule in self._schedule_df.
        """
        # Retrieve the pre-assigned schedule indices (one per person) from the person DataFrame
        chosen_schedules = self.person_df["chosen_schedule"].to_numpy()
        self._set_schedules(chosen_schedules)


def make_generator(kind: str = "probabilistic", *generator_args, **generator_kwargs) -> SyntheticGenerator:
    """
    Description: Factory function that creates and returns the correct SyntheticGenerator subclass
    based on a string identifier. This allows experiment code to select a generator type without
    importing the concrete classes directly.

    Input:
      - kind (str): Which generator type to create. Options are:
          'probabilistic' → ProbabilisticSyntheticGenerator (income + distance-based choices)
          'deterministic' → DeterministicSyntheticGenerator (binary rich/shop-first rules)
        Defaults to 'probabilistic'.
      - *generator_args: Positional arguments forwarded to the generator's __init__.
      - **generator_kwargs: Keyword arguments forwarded to the generator's __init__.

    Output:
      - (SyntheticGenerator): An instance of the requested generator subclass.
    """
    if kind == "probabilistic":
        return ProbabilisticSyntheticGenerator(*generator_args, **generator_kwargs)
    elif kind == "deterministic":
        return DeterministicSyntheticGenerator(*generator_args, **generator_kwargs)

    raise ValueError(f"Unknown generator kind: {kind}. Must be 'probabilistic' or 'deterministic'")


def compute_all_possible_schedules(graph: SyntheticGraph, available_schedules: np.ndarray) -> pl.DataFrame:
    """From a SyntheticGraph & associated SyntheticGenerator, compute all possible schedules / trips that
    can form on the graph

    Args:
        graph (SyntheticGraph)
        available_schedules (np.ndarray)

    Returns:
        pl.DataFrame: the schedules
    """

    # Build all possible (home, workplace) combinations using a cartesian product
    # home_choice_idx[i] and work_choice_idx[i] give the indices of one combination
    home_choice_idx = np.repeat(range(len(graph.home_nodes)), len(graph.workplace_nodes))
    work_choice_idx = np.tile(range(len(graph.workplace_nodes)), len(graph.home_nodes))

    # Actual node IDs for each (home, workplace) combination
    home_choice = graph.home_nodes[home_choice_idx]
    work_choice = graph.workplace_nodes[work_choice_idx]

    # For each home node, find the closest valid shopping node (excluding the home itself)
    closest_home_shopping = _select_closest_from_choice(
        graph.nodes,
        graph.distance_matrix,
        choices_idx=home_choice_idx,
        valid=graph.shopping_nodes,
        exclude_chosen=True,
    )
    # For each workplace node, find the closest valid shopping node (excluding the workplace itself)
    closest_work_shopping = _select_closest_from_choice(
        graph.nodes,
        graph.distance_matrix,
        choices_idx=work_choice_idx,
        valid=graph.shopping_nodes,
        exclude_chosen=True,
    )

    # Stack all 4 location choices into a (n_combinations, 4) array: [home, work, home_shop, work_shop]
    person_choices = np.stack([
        home_choice,
        work_choice,
        closest_home_shopping,
        closest_work_shopping,
    ]).T

    # Repeat each (home, work, shop, shop) combination once per available schedule pattern
    choices_repeated = np.repeat(person_choices, len(available_schedules), axis=0)
    # Extract the home node for each row (used to bookend each schedule with H...H)
    home_choice_repeated = choices_repeated[:, 0].reshape(-1, 1)
    # Tile schedule patterns across all location combinations
    scheds = np.tile(available_schedules.T, len(person_choices)).T

    # Replace activity type codes ('H', 'W', 'S1', 'S2') with the actual chosen location IDs
    for i, activity in enumerate(["H", "W", "S1", "S2"]):
        chosen_locs_repeated = choices_repeated[:, i].reshape(-1, 1)
        scheds = np.where(scheds == activity, chosen_locs_repeated, scheds)

    # Bookend every schedule with the home location at the start and end (H → ... → H)
    scheds = np.concat([home_choice_repeated, scheds, home_choice_repeated], axis=1)

    # Move all "-" to the left of the array, see https://stackoverflow.com/a/43011036
    # This left-justifies the schedule so all valid entries come first and "-" pad the right
    valid_mask = scheds != "-"
    flipped_mask = valid_mask.sum(axis=1, keepdims=1) > np.arange(scheds.shape[1] - 1, -1, -1)
    flipped_mask = flipped_mask[:, ::-1]

    scheds[flipped_mask] = scheds[valid_mask]
    scheds[~flipped_mask] = "-"

    # Convert the raw schedule array to a Polars DataFrame in long format
    schedule_df = (
        pl
        .DataFrame(scheds, schema=["1", "2", "3", "4", "5"], orient="row")
        .with_row_index("person_id")
        .unpivot(index="person_id", variable_name="numpy_seq", value_name="loc_id")
        .sort(by=["person_id", "numpy_seq"])
        .filter(pl.col("loc_id") != "-")  # Remove padding entries
        .with_columns(
            pl.int_range(pl.len()).over("person_id", order_by="numpy_seq").alias("sequence_num"),
            pl.lit("-").alias("type"),  # Placeholder type column (not used for this lookup)
        )
        .drop("numpy_seq")
    )

    # Build a temporary SyntheticSchedules to access the trip DataFrame (for sequence features)
    s = SyntheticSchedules(0, graph, None, None, schedule_df)
    # Build cumulative indicator features: for each step, which nodes have been visited so far?
    features = (
        s.trip_df
        .group_by("person_id")
        .agg(pl.col("from_loc_id").unique(maintain_order=True))
        .explode("from_loc_id")
        .with_columns(pl.int_range(pl.len()).over("person_id").alias("sequence_num"))
        .to_dummies("from_loc_id")  # One-hot encode the current location
        # Cumulative sum gives the set of all locations visited up to this step
        .with_columns(cs.starts_with("from_loc_id").cum_sum().over("person_id", order_by="sequence_num"))
    )
    # Shift features by -1 to get the NEXT state (used as prediction targets)
    targets = features.with_columns(pl.col("sequence_num") - 1).rename(lambda col: col.replace("from_", "to_"))
    # Join current state features with next state targets for each (person_id, sequence_num) pair
    dataset_df = features.join(targets, on=["person_id", "sequence_num"]).sort(by=["person_id", "sequence_num"])

    return dataset_df
