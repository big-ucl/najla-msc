"""
Module: activitygraphs/archive/exploration/metrics.py

Description:
    Defines a hierarchy of graph metric classes and a Metrics runner for computing
    structural properties of household activity graphs.

    The module follows a three-tier class hierarchy:
      - Metric (abstract base): holds name and category.
          - GraphMetric: metrics that require a NetworkX graph (e.g. radius, diameter).
              Subclasses override _compute_metric(G).
          - PolarsMetric: metrics computed directly on Polars DataFrames using
              Polars expressions (no NetworkX needed — fast, vectorised).
              - NodeMetric: Polars expression applied to the node table.
              - EdgeMetric: Polars expression applied to the edge/trip table.

    Concrete metric classes (Order, Size, Radius, Diameter, …) each inherit from
    the appropriate base class and implement either _expr() or _compute_metric().

    The Metrics class is the main entry point for users: it accepts lists of node,
    edge, and graph metric objects and computes all of them together, joining the
    results into a single per-household DataFrame.

    AllMetrics is a pre-configured Metrics instance with all standard metrics enabled.
"""

import itertools
import warnings
from abc import ABC, abstractmethod
from enum import Enum
from typing import Iterator

import networkx as nx
import polars as pl
from archive.exploration.graphs import ActivityGraph
from tqdm.auto import tqdm

# ── Enumerations ──────────────────────────────────────────────────────────────


class Category(Enum):
    """
    Description:
        Categories that group graph metrics by the structural property they measure.
        Used for labelling and filtering results.

      - NA: uncategorised or general metrics.
      - CONNECTEDNESS: metrics about how well-connected the graph is (e.g. density,
            reciprocity, centralisation).
      - QUANTITY: counts of graph elements (e.g. number of nodes, number of edges,
            total distance).
      - EXTENT: metrics about the spatial or temporal span of activities (e.g. max
            or mean trip distance).
      - CLUSTERING: metrics related to clustering structure (e.g. assortativity,
            Katz centralisation).
    """
    NA = "N/A"                      # Uncategorised / general
    CONNECTEDNESS = "connectedness" # How well nodes are connected to each other
    QUANTITY = "quantity"           # Simple counts or totals
    EXTENT = "extent"               # Spatial/temporal spread of activities
    CLUSTERING = "clustering"       # Clustering and grouping structure


class WeightColumn(Enum):
    """
    Description:
        Selects which numeric edge attribute is used as a weight when computing
        weighted graph metrics (e.g. weighted radius, weighted assortativity).

      - NA: no weighting — the metric is computed on the unweighted graph.
      - DISTANCE: use the `distance` column (trip distance in km or metres).
      - DURATION: use the `duration` column (trip duration in minutes).
    """
    NA = "N/A"            # Unweighted variant of the metric
    DISTANCE = "distance" # Weight by trip distance
    DURATION = "duration" # Weight by trip duration


# ── Abstract base classes ─────────────────────────────────────────────────────


class Metric(ABC):
    """
    Description:
        Abstract base class for all graph metrics. Stores the metric's name and
        category. Concrete subclasses must either implement _compute_metric() (for
        NetworkX-based metrics) or _expr() (for Polars expression-based metrics).

    Attributes:
      - name (str): the column name this metric will use in result DataFrames.
      - category (Category): the structural category this metric belongs to.
    """
    name: str           # Column name for this metric in result DataFrames
    category: Category  # Structural category (e.g. CONNECTEDNESS, QUANTITY)

    def __init__(self, name: str, category: Category):
        """
        Description:
            Stores the metric name and category. Called by all subclass constructors.

        Input:
          - name (str): column name for this metric in result DataFrames.
          - category (Category): which structural category this metric belongs to.
        """
        self.name = name
        self.category = category


class GraphMetric(Metric):
    """
    Description:
        Abstract base class for metrics that require a materialised NetworkX graph.
        Subclasses implement _compute_metric(G) to return a single scalar per graph.

        GraphMetric objects are callable: metric(G) is equivalent to
        metric._compute_metric(G).

    Attributes:
      - weight_col (WeightColumn): which edge attribute to use as weight (or NA for
            unweighted computation).
      - return_dtype (pl.DataType): the Polars column type for the result (default
            Float64; use Int64 for integer-valued metrics).
    """
    weight_col: WeightColumn   # Edge attribute used as edge weight (or NA for unweighted)
    return_dtype: pl.DataType  # Polars dtype for the result column in the output DataFrame

    def __init__(self, name, category: Category, weight_col: WeightColumn, return_dtype: pl.DataType = pl.Float64):
        """
        Description:
            Initialises a GraphMetric with its weight column and Polars return type.

        Input:
          - name (str): column name for this metric's results.
          - category (Category): structural category of this metric.
          - weight_col (WeightColumn): which edge attribute to use as weight.
          - return_dtype (pl.DataType): Polars dtype for the output column (default Float64).
        """
        self.weight_col = weight_col    # Store the weight column selector
        self.return_dtype = return_dtype  # Store the expected output dtype
        super().__init__(name, category)

    def __call__(self, G: nx.MultiDiGraph) -> int | float:
        """
        Description:
            Makes GraphMetric instances callable, so they can be used as metric(G)
            rather than metric._compute_metric(G). Delegates to _compute_metric.

        Input:
          - G (nx.MultiDiGraph): the household activity graph to compute the metric on.

        Output:
          - (int | float | None): the scalar metric value, or None if the metric is
                undefined for this graph (e.g. not strongly connected for radius).
        """
        return self._compute_metric(G)

    @abstractmethod
    def _compute_metric(self, G: nx.MultiDiGraph) -> int | float: ...


class PolarsMetric(Metric):
    """
    Description:
        Abstract base class for metrics computed via Polars expressions.
        Subclasses implement _expr() which returns a Polars aggregation expression.
        The expression is wrapped with .alias(self.name) so the result column is
        named correctly.

        PolarsMetric objects are not callable; instead use polars_expr() to get
        the aliased expression for use in a DataFrame.group_by().agg() call.
    """

    def __init__(self, name: str, category: Category):
        """
        Description:
            Initialises a PolarsMetric (delegates to Metric.__init__).

        Input:
          - name (str): column name for this metric's results in the output DataFrame.
          - category (Category): structural category of this metric.
        """
        super().__init__(name, category)

    def polars_expr(self) -> pl.Expr:
        """
        Description:
            Returns the Polars aggregation expression for this metric, aliased to
            the metric's name. Passed to DataFrame.group_by().agg() during computation.

        Output:
          - (pl.Expr): the aggregation expression aliased as self.name.
        """
        # Wrap the raw expression with .alias so the output column is named correctly
        return self._expr().alias(self.name)

    @abstractmethod
    def _expr(self) -> pl.Expr: ...


class NodeMetric(PolarsMetric):
    """
    Description:
        Marker subclass of PolarsMetric for metrics computed on the node (location)
        table. The Metrics runner will group node_df by grouping_col and apply these
        expressions. No additional behaviour beyond PolarsMetric.
    """
    pass


class EdgeMetric(PolarsMetric):
    """
    Description:
        Subclass of PolarsMetric for metrics computed on the edge (trip) table.
        Can optionally reference a weight column (e.g. distance or duration) via
        the weight_col attribute.

    Attributes:
      - weight_col (WeightColumn): which trip column to use as a weight
            (NA means unweighted / count-based).
    """
    weight_col: WeightColumn  # Which trip attribute column this metric uses for weighting

    def __init__(self, name: str, category: Category, weight_col: WeightColumn = WeightColumn.NA):
        """
        Description:
            Initialises an EdgeMetric with an optional weight column.

        Input:
          - name (str): column name for this metric's results.
          - category (Category): structural category of this metric.
          - weight_col (WeightColumn): which trip column to use (default: NA / no weight).
        """
        self.weight_col = weight_col  # Store the weight column selector
        super().__init__(name, category)


# ── Concrete metric implementations ──────────────────────────────────────────


class Order(NodeMetric):
    """
    Description:
        Counts the number of unique visited locations (graph nodes) for each household.
        A higher order means the household visited more distinct places.
        This is a NodeMetric — computed on the node table via a Polars expression.
    """
    def __init__(self):
        """
        Description: Initialises the Order metric with a fixed name ("order") and
        the QUANTITY category. No parameters are needed because order is always
        defined the same way: count of location nodes per household.

        Output:
          - (Order): A fully configured Order metric instance, ready to be passed
                to a Metrics runner.
        """
        super().__init__(name="order", category=Category.QUANTITY)

    def _expr(self):
        """
        Output:
          - (pl.Expr): counts the number of loc_id entries (= number of visited locations).
        """
        return pl.col("loc_id").len()


class Size(EdgeMetric):
    """
    Description:
        Counts the number of trips (graph edges) made by each household.
        A higher size means more trips were recorded for that household.
        This is an EdgeMetric — computed on the edge/trip table.
    """
    def __init__(self, weight_col: WeightColumn = WeightColumn.NA):
        """
        Description: Initialises the Size metric with a fixed name ("size") and the QUANTITY
        category. The weight_col parameter is accepted for interface compatibility with other
        EdgeMetric subclasses, but Size always counts rows (ignores the weight column value).

        Input:
          - weight_col (WeightColumn): Accepted for compatibility but not used in the count
                expression. Defaults to WeightColumn.NA (unweighted).

        Output:
          - (Size): A fully configured Size metric instance, ready to be passed to a Metrics runner.
        """
        super().__init__(name="size", category=Category.QUANTITY, weight_col=weight_col)

    def _expr(self):
        """
        Output:
          - (pl.Expr): counts all rows in the grouped edge table (= number of trips).
        """
        return pl.len()


class DistanceMax(EdgeMetric):
    """
    Description:
        Returns the maximum value of the weight column (distance or duration)
        across all trips for each household. Captures the longest single trip.
    """
    def __init__(self, weight_col: WeightColumn):
        """
        Input:
          - weight_col (WeightColumn): DISTANCE or DURATION (selects which column to take max of).
        """
        super().__init__(name=f"{weight_col.value}_max", category=Category.EXTENT, weight_col=weight_col)

    def _expr(self):
        """
        Output:
          - (pl.Expr): maximum of the weight column across all trips for a household.
        """
        return pl.col(self.weight_col.value).max()


class Radius(GraphMetric):
    """
    Description:
        Computes the graph radius — the minimum over all nodes of the maximum
        shortest-path distance to any other node (the eccentricity of the
        "most central" node). Only defined for strongly connected graphs.

        Returns None if the graph is not strongly connected (e.g. isolated nodes).
    """
    def __init__(self, weight_col: WeightColumn):
        """
        Input:
          - weight_col (WeightColumn): edge attribute used as path weight (NA = hop count).
        """
        super().__init__(name=f"radius_({weight_col.value})", category=Category.NA, weight_col=weight_col)

    def _compute_metric(self, G: nx.MultiDiGraph):
        """
        Input:
          - G (nx.MultiDiGraph): the household activity graph.

        Output:
          - (float | None): graph radius, or None if G is not strongly connected.
        """
        # Radius is undefined for disconnected graphs; return None to indicate missing
        if not nx.is_strongly_connected(G):
            return None

        return nx.radius(G, weight=self.weight_col.value)


class Diameter(GraphMetric):
    """
    Description:
        Computes the graph diameter — the maximum shortest-path distance between
        any pair of nodes (the eccentricity of the "outermost" node).
        Only defined for strongly connected graphs; returns None otherwise.
    """
    def __init__(self, weight_col: WeightColumn):
        """
        Input:
          - weight_col (WeightColumn): edge attribute used as path weight (NA = hop count).
        """
        super().__init__(name=f"diameter_({weight_col.value})", category=Category.NA, weight_col=weight_col)

    def _compute_metric(self, G: nx.MultiDiGraph):
        """
        Input:
          - G (nx.MultiDiGraph): the household activity graph.

        Output:
          - (float | None): graph diameter, or None if G is not strongly connected.
        """
        if not nx.is_strongly_connected(G):
            return None

        return nx.diameter(G, weight=self.weight_col.value)


class DistanceMean(EdgeMetric):
    """
    Description:
        Computes the mean value of the weight column (distance or duration) across
        all trips for each household. Represents the average trip length or time.
    """
    def __init__(self, weight_col: WeightColumn):
        """
        Input:
          - weight_col (WeightColumn): DISTANCE or DURATION.
        """
        super().__init__(name=f"{weight_col.value}_mean", category=Category.EXTENT, weight_col=weight_col)

    def _expr(self):
        """
        Output:
          - (pl.Expr): mean of the weight column across all trips for a household.
        """
        return pl.col(self.weight_col.value).mean()


class DistanceSum(EdgeMetric):
    """
    Description:
        Computes the total (sum) of the weight column across all trips for each
        household. For distance this is total distance travelled; for duration it
        is total time spent travelling.
    """
    def __init__(self, weight_col: WeightColumn):
        """
        Input:
          - weight_col (WeightColumn): DISTANCE or DURATION.
        """
        super().__init__(name=f"{weight_col.value}_sum", category=Category.QUANTITY, weight_col=weight_col)

    def _expr(self):
        """
        Output:
          - (pl.Expr): sum of the weight column across all trips for a household.
        """
        return pl.col(self.weight_col.value).sum()


class EdgeDensity(GraphMetric):
    """
    Description:
        Computes the edge density of the graph: the ratio of actual edges to the
        maximum possible edges (n*(n-1) for a directed graph on n nodes).
        Values range from 0 (no edges) to 1 (complete graph).
        A high density means the household visits many location pairs.
    """
    def __init__(self, weight_col: WeightColumn = WeightColumn.NA):
        """
        Description: Initialises the EdgeDensity metric with a fixed name ("edge_density")
        and the CONNECTEDNESS category. The weight_col is accepted for interface compatibility
        but EdgeDensity uses NetworkX's nx.density() which does not use edge weights.

        Input:
          - weight_col (WeightColumn): Accepted for interface compatibility but ignored in the
                density computation. Defaults to WeightColumn.NA.

        Output:
          - (EdgeDensity): A fully configured EdgeDensity metric instance.
        """
        super().__init__(name="edge_density", category=Category.CONNECTEDNESS, weight_col=weight_col)

    def _compute_metric(self, G: nx.MultiDiGraph):
        """
        Input:
          - G (nx.MultiDiGraph): the household activity graph.

        Output:
          - (float): edge density in [0, 1].
        """
        return nx.density(G)


class Assortativity(GraphMetric):
    """
    Description:
        Computes the degree assortativity coefficient — the Pearson correlation
        between the degrees of connected node pairs. Positive values mean high-degree
        nodes tend to connect to other high-degree nodes (assortative mixing);
        negative values mean high-degree nodes connect to low-degree nodes.

        Known issue: raises RuntimeWarning (division by zero) for graphs with exactly
        2 nodes or some isolated components; these return None.
    """
    def __init__(self, weight_col: WeightColumn):
        """
        Input:
          - weight_col (WeightColumn): edge attribute used as weight for the
                weighted degree computation.
        """
        super().__init__(name=f"assortativity-{weight_col.value}", category=Category.CLUSTERING, weight_col=weight_col)

    def _compute_metric(self, G):
        """
        Input:
          - G (nx.MultiDiGraph): the household activity graph.

        Output:
          - (float | None): assortativity coefficient in [-1, 1], or None on error.
        """
        try:
            # FIXME Division by zero when graph has 2 nodes or with some lone componenents
            # Temporarily convert RuntimeWarnings to exceptions so we can catch them
            warnings.filterwarnings("error")
            return nx.degree_assortativity_coefficient(G, weight=self.weight_col.value)
        except RuntimeWarning:
            # Return None for degenerate cases instead of crashing
            return None
        finally:
            # Always restore the default warning behaviour after this call
            warnings.resetwarnings()


class Reciprocity(GraphMetric):
    """
    Description:
        Computes graph reciprocity — the fraction of edges that have a corresponding
        reverse edge. A value of 1.0 means every trip A->B also has a return trip B->A;
        0.0 means no return trips at all.

        Note: computed on nx.DiGraph (collapsing multi-edges) rather than the raw
        MultiDiGraph, since multiple parallel edges would inflate the count.
    """
    def __init__(self, weight_col: WeightColumn = WeightColumn.NA):
        """
        Description: Initialises the Reciprocity metric with a fixed name ("reciprocity") and
        the CONNECTEDNESS category. The weight_col is accepted for interface compatibility but
        NetworkX's reciprocity() function does not use edge weights.

        Input:
          - weight_col (WeightColumn): Accepted for interface compatibility but not used in
                the reciprocity calculation. Defaults to WeightColumn.NA.

        Output:
          - (Reciprocity): A fully configured Reciprocity metric instance.
        """
        super().__init__("reciprocity", Category.CONNECTEDNESS, weight_col=weight_col)

    def _compute_metric(self, G):
        """
        Input:
          - G (nx.MultiDiGraph): the household activity graph.

        Output:
          - (float): reciprocity in [0, 1].
        """
        # Convert to simple DiGraph first to avoid double-counting parallel edges
        return nx.reciprocity(nx.DiGraph(G))


# ── Centralisation metrics ────────────────────────────────────────────────────


def _compute_freeman_centralisation(centralities: dict[str, float]) -> float:
    """
    Description:
        Computes the Freeman graph centralisation index from a dictionary of
        per-node centrality scores.

        Freeman centralisation measures how much one node dominates over all others
        compared to a star graph (the most centralised structure). It normalises the
        sum of differences between the maximum centrality and each node's centrality
        by the theoretical maximum for a graph of that size.

        Returns None for graphs with 2 or fewer nodes because the normalisation
        formula requires at least 3 nodes to be meaningful.

    Input:
      - centralities (dict[str, float]): mapping from node ID string to its
            centrality score (e.g. from nx.degree_centrality or nx.betweenness_centrality).

    Output:
      - (float | None): Freeman centralisation index in [0, 1], where 0 means all
            nodes have equal centrality and 1 means one node dominates completely
            (star topology). Returns None for graphs with <= 2 nodes.
    """
    # Freeman centralisation is undefined for tiny graphs (would divide by zero)
    if len(centralities) <= 2:
        return None

    # The node with the highest centrality score determines the "star centre"
    max_centrality = centralities[max(centralities, key=centralities.get)]
    # Normalisation factor for a directed graph: (n-1)*(n-2)
    normalisation = (len(centralities) - 1) * (len(centralities) - 2)

    # Sum of differences from the maximum, divided by the theoretical maximum
    return sum(max_centrality - c for c in centralities.values()) / normalisation


class DegreeCentralisation(GraphMetric):
    """
    Description:
        Computes the Freeman degree centralisation — how much one location dominates
        the activity pattern in terms of how many other locations it connects to.
        A value near 1 means one location (e.g. home) is visited far more than others.
    """
    def __init__(self, weight_col: WeightColumn = WeightColumn.NA):
        """
        Description: Initialises the DegreeCentralisation metric with a fixed name
        ("degree_central") and the CONNECTEDNESS category. The weight_col parameter is
        accepted for interface compatibility with AllMetrics but is always overridden
        to WeightColumn.NA internally, because nx.degree_centrality() does not use
        edge weights.

        Input:
          - weight_col (WeightColumn): Accepted for interface compatibility with the AllMetrics
                constructor pattern, but the internal computation always uses WeightColumn.NA.
                Defaults to WeightColumn.NA.

        Output:
          - (DegreeCentralisation): A fully configured DegreeCentralisation metric instance.
        """
        super().__init__(name="degree_central", category=Category.CONNECTEDNESS, weight_col=WeightColumn.NA)

    def _compute_metric(self, G):
        """
        Input:
          - G (nx.MultiDiGraph): the household activity graph.

        Output:
          - (float | None): Freeman degree centralisation in [0, 1], or None for tiny graphs.
        """
        centralities = nx.degree_centrality(G)  # Per-node degree centrality scores
        return _compute_freeman_centralisation(centralities)


class BetweenessCentralisation(GraphMetric):
    """
    Description:
        Computes the Freeman betweenness centralisation — how much one location lies
        on the shortest paths between other locations. A high value means trips are
        "funnelled" through one key hub location.
    """
    def __init__(self, weight_col: WeightColumn):
        """
        Input:
          - weight_col (WeightColumn): edge attribute used as path cost for shortest paths.
        """
        super().__init__(name=f"betweeness_central_({weight_col.value})", category=Category.NA, weight_col=weight_col)

    def _compute_metric(self, G):
        """
        Input:
          - G (nx.MultiDiGraph): the household activity graph.

        Output:
          - (float | None): Freeman betweenness centralisation in [0, 1], or None.
        """
        # Compute per-node betweenness centrality scores
        centralities = nx.betweenness_centrality(G, weight=self.weight_col)
        return _compute_freeman_centralisation(centralities)


class ClosenessCentralisation(GraphMetric):
    """
    Description:
        Computes Freeman closeness centralisation — how much one location is closer
        (in path-distance terms) to all other locations than the average.
    """
    def __init__(self, weight_col: WeightColumn):
        """
        Input:
          - weight_col (WeightColumn): edge attribute used as path distance.
        """
        super().__init__(name=f"closeness_central_({weight_col.value})", category=Category.NA, weight_col=weight_col)

    def _compute_metric(self, G):
        """
        Input:
          - G (nx.MultiDiGraph): the household activity graph.

        Output:
          - (float | None): Freeman closeness centralisation in [0, 1], or None.
        """
        # Compute per-node closeness centrality scores using the specified distance attribute
        centralities = nx.closeness_centrality(G, distance=self.weight_col)
        return _compute_freeman_centralisation(centralities)


class KatzCentralisation(GraphMetric):
    """
    Description:
        Computes the Freeman centralisation of Katz centrality scores.
        Katz centrality counts all paths (not just shortest paths) between nodes,
        with longer paths penalised by an attenuation factor. Captures global
        influence in the activity pattern.

        Returns None for graphs with 2 or fewer nodes (Katz is ill-defined there).
    """
    def __init__(self, weight_col: WeightColumn):
        """
        Input:
          - weight_col (WeightColumn): edge attribute used as weight in Katz computation.
        """
        super().__init__(name=f"katz_central_({weight_col.value})", category=Category.CLUSTERING, weight_col=weight_col)

    def _compute_metric(self, G):
        """
        Input:
          - G (nx.MultiDiGraph): the household activity graph.

        Output:
          - (float | None): Freeman Katz centralisation in [0, 1], or None for tiny graphs.
        """
        # Katz centrality is numerically unstable for very small graphs
        if G.number_of_nodes() <= 2:
            return None

        # Use the numpy-based solver for stability; collapse to DiGraph to avoid multi-edge issues
        centralities = nx.katz_centrality_numpy(nx.DiGraph(G), weight=self.weight_col)
        return _compute_freeman_centralisation(centralities)


# ── Metric runners and computers ──────────────────────────────────────────────


def _polars_exprs(polars_metrics: list[PolarsMetric]) -> list[pl.Expr]:
    """
    Description:
        Converts a list of PolarsMetric objects into a list of aliased Polars
        aggregation expressions ready to pass to DataFrame.group_by().agg().

    Input:
      - polars_metrics (list[PolarsMetric]): node or edge metric objects to convert.

    Output:
      - (list[pl.Expr]): list of Polars expressions, one per metric.
    """
    return [metric.polars_expr() for metric in polars_metrics]


class Metrics:
    """
    Description:
        Orchestrates the computation of multiple graph metrics across an ActivityGraph
        and returns the results as a single per-household Polars DataFrame.

        Accepts three separate lists of metric types:
          - node_metrics: computed on the node table (fast, vectorised).
          - edge_metrics: computed on the edge table (fast, vectorised).
          - graph_metrics: computed on NetworkX graphs (slow, requires materialisation).

        The three result DataFrames are joined on grouping_col (default "hh_id") to
        produce one wide row per household with all metric columns.

    Attributes:
      - grouping_col (str): column used to group results (one row per unique value).
      - node_metrics (list[NodeMetric]): metrics computed on the node table.
      - edge_metrics (list[EdgeMetric]): metrics computed on the edge table.
      - graph_metrics (list[GraphMetric]): metrics computed on NetworkX graphs.
    """

    def __init__(
        self,
        node_metrics: list[NodeMetric],
        edge_metrics: list[EdgeMetric],
        graph_metrics: list[GraphMetric],
        grouping_col: str = "hh_id",
    ):
        """
        Description:
            Initialises the Metrics runner with its three lists of metrics and
            builds the output schema for the graph-metrics DataFrame.

        Input:
          - node_metrics (list[NodeMetric]): metrics to compute on the node table.
          - edge_metrics (list[EdgeMetric]): metrics to compute on the edge table.
          - graph_metrics (list[GraphMetric]): metrics to compute via NetworkX.
          - grouping_col (str): column to group by (default "hh_id" for per-household results).
        """
        self.grouping_col = grouping_col  # Column name used to group and join results
        self.node_metrics = node_metrics  # List of NodeMetric objects
        self.edge_metrics = edge_metrics  # List of EdgeMetric objects
        self.graph_metrics = graph_metrics  # List of GraphMetric objects (most expensive)

        # Build the schema for the graph-metrics output: one string ID column + one column per metric
        metric_dict = {metric.name: metric.return_dtype for metric in graph_metrics}
        # The leading grouping_col column must be first in the schema
        self._graph_schema = pl.Schema({grouping_col: pl.String} | metric_dict)

    @property
    def metrics(self) -> list[Metric]:
        """
        Description:
            Returns all metrics across all three lists in a flat list.
            Useful for iterating over all metrics regardless of type.

        Output:
          - (list[Metric]): combined list of node, edge, and graph metrics.
        """
        return self.node_metrics + self.edge_metrics + self.graph_metrics

    def names(self) -> list[str]:
        """
        Description:
            Returns the output column names for all metrics, in the same order
            as self.metrics. Useful for inspecting or selecting metric columns.

        Output:
          - (list[str]): list of metric name strings.
        """
        return [metric.name for metric in self.metrics]

    def compute(
        self,
        graph: ActivityGraph,
        verbose: bool = False,
        max_iter: int | None = None,
        how: str = "inner",
    ) -> pl.DataFrame:
        """
        Description:
            Computes all three metric groups and joins their results into a single
            wide DataFrame with one row per household.

        Input:
          - graph (ActivityGraph): the dataset to compute metrics on.
          - verbose (bool): if True, shows a tqdm progress bar for graph metrics.
          - max_iter (int | None): if set, limits computation to the first max_iter
                households (useful for quick testing).
          - how (str): join strategy for combining the three result tables.
                "inner" (default) keeps only households present in all three tables.

        Output:
          - (pl.DataFrame): one row per household with all metric columns, joined
                on grouping_col.
        """
        # Compute the three metric groups separately, then join them
        node_metrics = self._compute_node_metrics(graph)
        edge_metrics = self._compute_edge_metrics(graph)
        graph_metrics = self._compute_graph_metrics(graph, verbose=verbose, max_iter=max_iter)

        # Join all three on the grouping column to produce one wide result table
        return node_metrics.join(edge_metrics, on=self.grouping_col, how=how).join(
            graph_metrics, on=self.grouping_col, how=how
        )

    def _compute_node_metrics(self, graph: ActivityGraph) -> pl.DataFrame:
        """
        Description:
            Groups the node DataFrame by grouping_col and applies all node metric
            Polars expressions. Fast vectorised computation, no NetworkX needed.

        Input:
          - graph (ActivityGraph): source of the node_df to aggregate.

        Output:
          - (pl.DataFrame): one row per household with one column per NodeMetric.
        """
        metrics = _polars_exprs(self.node_metrics)  # Convert metric objects to Polars expressions
        grouped_nodes = graph.node_df.group_by(self.grouping_col)
        return grouped_nodes.agg(*metrics)

    def _compute_edge_metrics(self, graph: ActivityGraph) -> pl.DataFrame:
        """
        Description:
            Groups the edge DataFrame by grouping_col and applies all edge metric
            Polars expressions. Fast vectorised computation, no NetworkX needed.

        Input:
          - graph (ActivityGraph): source of the edge_df to aggregate.

        Output:
          - (pl.DataFrame): one row per household with one column per EdgeMetric.
        """
        metrics = _polars_exprs(self.edge_metrics)  # Convert metric objects to Polars expressions
        grouped_nodes = graph.edge_df.group_by(self.grouping_col)
        return grouped_nodes.agg(*metrics)

    def _compute_graph_metrics(self, graph: ActivityGraph, verbose: bool, max_iter: int | None) -> pl.DataFrame:
        """
        Description:
            Materialises NetworkX graphs one household at a time and computes all
            GraphMetric objects on each. Collects results into a Polars DataFrame.

        Input:
          - graph (ActivityGraph): source of NetworkX graphs (via graph.to_nxs()).
          - verbose (bool): show tqdm progress bar if True.
          - max_iter (int | None): limit to the first max_iter households if set.

        Output:
          - (pl.DataFrame): one row per household with one column per GraphMetric,
                typed according to self._graph_schema.
        """
        results_generator = self._metrics_generator(graph, verbose, max_iter)

        # Optionally cap the number of households processed (for testing/debugging)
        if max_iter is not None:
            results_generator = itertools.islice(results_generator, max_iter)

        # Materialise the generator into a DataFrame with the pre-defined schema
        return pl.DataFrame(data=results_generator, schema=self._graph_schema)

    def _metrics_generator(
        self, graph: ActivityGraph, verbose: bool, max_iter: int | None
    ) -> Iterator[dict[str, int | float]]:
        """
        Description:
            Generator that lazily yields one result dictionary per household.
            For each household, materialises its NetworkX graph and computes all
            GraphMetric objects.

        Input:
          - graph (ActivityGraph): provides the to_nxs() iterator of NetworkX graphs.
          - verbose (bool): if True, wraps the iterator with a tqdm progress bar.
          - max_iter (int | None): total count for the progress bar (or None to use
                graph.n_subgraphs).

        Output:
          - (Iterator[dict[str, int | float]]): yields dicts with keys
                {grouping_col: hh_id, metric1_name: value, ...} for each household.
        """
        # Use max_iter as the progress bar total if provided, otherwise use full dataset size
        total = max_iter if max_iter is not None else graph.n_subgraphs

        for hh_id, nx_graph in tqdm(graph.to_nxs(), total=total, disable=not verbose):
            # Compute every GraphMetric for this household's NetworkX graph
            metrics_dict = {metric.name: metric(nx_graph) for metric in self.graph_metrics}
            # Yield a row dict: {grouping_col: hh_id, "radius_(distance)": 3.2, ...}
            yield {self.grouping_col: hh_id} | metrics_dict


class AllMetrics(Metrics):
    """
    Description:
        A pre-configured Metrics instance that includes all standard graph metrics
        used in the activity-pattern analysis. Instantiate this instead of building
        a Metrics object manually when you want the full set of metrics.

        Included metrics:
          Node:  Order (number of locations visited)
          Edge:  Size, DistanceMax, DistanceMean, DistanceSum (trip counts/lengths)
          Graph: Radius, Diameter, EdgeDensity, Assortativity, Reciprocity,
                 DegreeCentralisation, BetweenessCentralisation,
                 ClosenessCentralisation, KatzCentralisation
    """
    def __init__(self, weight_col: WeightColumn, grouping_col: str = "hh_id"):
        """
        Description:
            Builds all standard metric lists and passes them to the Metrics parent.

        Input:
          - weight_col (WeightColumn): which edge attribute (DISTANCE or DURATION)
                to use for all weighted metrics.
          - grouping_col (str): column to group results by (default "hh_id").
        """
        # Node metrics: only Order (number of distinct locations visited)
        node_metrics = [Order()]

        # Edge metric classes that accept a weight_col argument
        edge_metrics_cls = [Size, DistanceMax, DistanceMean, DistanceSum]

        # Graph metric classes that accept a weight_col argument
        graph_metrics_cls = [
            Radius,
            Diameter,
            EdgeDensity,
            Assortativity,
            Reciprocity,
            DegreeCentralisation,
            BetweenessCentralisation,
            ClosenessCentralisation,
            KatzCentralisation,
        ]

        # Instantiate each metric class with the chosen weight column
        edge_metrics = [metric_cls(weight_col) for metric_cls in edge_metrics_cls]
        graph_metrics = [metric_cls(weight_col) for metric_cls in graph_metrics_cls]

        super().__init__(node_metrics, edge_metrics, graph_metrics, grouping_col)
