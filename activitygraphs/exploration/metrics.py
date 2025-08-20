import itertools
import warnings
from abc import ABC, abstractmethod
from enum import Enum
from typing import Iterator

import networkx as nx
import polars as pl
from exploration.graphs import ActivityGraph
from tqdm.auto import tqdm

# Enums


class Category(Enum):
    NA = "N/A"
    CONNECTEDNESS = "connectedness"
    QUANTITY = "quantity"
    EXTENT = "extent"
    CLUSTERING = "clustering"


class WeightColumn(Enum):
    NA = "N/A"
    DISTANCE = "distance"
    DURATION = "duration"


# Abstract base classes


class Metric(ABC):
    name: str
    category: Category

    def __init__(self, name: str, category: Category):
        self.name = name
        self.category = category


class GraphMetric(Metric):
    weight_col: WeightColumn
    return_dtype: pl.DataType

    def __init__(self, name, category: Category, weight_col: WeightColumn, return_dtype: pl.DataType = pl.Float64):
        self.weight_col = weight_col
        self.return_dtype = return_dtype
        super().__init__(name, category)

    def __call__(self, G: nx.MultiDiGraph) -> int | float:
        return self._compute_metric(G)

    @abstractmethod
    def _compute_metric(self, G: nx.MultiDiGraph) -> int | float: ...


class PolarsMetric(Metric):
    def __init__(self, name: str, category: Category):
        super().__init__(name, category)

    def polars_expr(self) -> pl.Expr:
        return self._expr().alias(self.name)

    @abstractmethod
    def _expr(self) -> pl.Expr: ...


class NodeMetric(PolarsMetric):
    pass


class EdgeMetric(PolarsMetric):
    weight_col: WeightColumn

    def __init__(self, name: str, category: Category, weight_col: WeightColumn = WeightColumn.NA):
        self.weight_col = weight_col
        super().__init__(name, category)


# Concrete metrics


class Order(NodeMetric):
    def __init__(self):
        super().__init__(name="order", category=Category.QUANTITY)

    def _expr(self):
        return pl.col("loc_id").len()


class Size(EdgeMetric):
    def __init__(self, weight_col: WeightColumn = WeightColumn.NA):
        super().__init__(name="size", category=Category.QUANTITY, weight_col=weight_col)

    def _expr(self):
        return pl.len()


class DistanceMax(EdgeMetric):
    def __init__(self, weight_col: WeightColumn):
        super().__init__(name=f"{weight_col.value}_max", category=Category.EXTENT, weight_col=weight_col)

    def _expr(self):
        return pl.col(self.weight_col.value).max()


class Radius(GraphMetric):
    def __init__(self, weight_col: WeightColumn):
        super().__init__(name=f"radius_({weight_col.value})", category=Category.NA, weight_col=weight_col)

    def _compute_metric(self, G: nx.MultiDiGraph):
        if not nx.is_strongly_connected(G):
            return None

        return nx.radius(G, weight=self.weight_col.value)


class Diameter(GraphMetric):
    def __init__(self, weight_col: WeightColumn):
        super().__init__(name=f"diameter_({weight_col.value})", category=Category.NA, weight_col=weight_col)

    def _compute_metric(self, G: nx.MultiDiGraph):
        if not nx.is_strongly_connected(G):
            return None

        return nx.diameter(G, weight=self.weight_col.value)


class DistanceMean(EdgeMetric):
    def __init__(self, weight_col: WeightColumn):
        super().__init__(name=f"{weight_col.value}_mean", category=Category.EXTENT, weight_col=weight_col)

    def _expr(self):
        return pl.col(self.weight_col.value).mean()


class DistanceSum(EdgeMetric):
    def __init__(self, weight_col: WeightColumn):
        super().__init__(name=f"{weight_col.value}_sum", category=Category.QUANTITY, weight_col=weight_col)

    def _expr(self):
        return pl.col(self.weight_col.value).sum()


class EdgeDensity(GraphMetric):
    def __init__(self, weight_col: WeightColumn = WeightColumn.NA):
        super().__init__(name="edge_density", category=Category.CONNECTEDNESS, weight_col=weight_col)

    def _compute_metric(self, G: nx.MultiDiGraph):
        return nx.density(G)


class Assortativity(GraphMetric):
    def __init__(self, weight_col: WeightColumn):
        super().__init__(name=f"assortativity-{weight_col.value}", category=Category.CLUSTERING, weight_col=weight_col)

    def _compute_metric(self, G):
        try:
            # FIXME Division by zero when graph has 2 nodes or with some lone componenents
            warnings.filterwarnings("error")
            return nx.degree_assortativity_coefficient(G, weight=self.weight_col.value)
        except RuntimeWarning:
            return None
        finally:
            warnings.resetwarnings()


class Reciprocity(GraphMetric):
    def __init__(self, weight_col: WeightColumn = WeightColumn.NA):
        super().__init__("reciprocity", Category.CONNECTEDNESS, weight_col=weight_col)

    def _compute_metric(self, G):
        return nx.reciprocity(nx.DiGraph(G))


# Centralisation metrics


def _compute_freeman_centralisation(centralities: dict[str, float]) -> float:
    if len(centralities) <= 2:
        return None

    max_centrality = centralities[max(centralities, key=centralities.get)]
    normalisation = (len(centralities) - 1) * (len(centralities) - 2)

    return sum(max_centrality - c for c in centralities.values()) / normalisation


class DegreeCentralisation(GraphMetric):
    def __init__(self, weight_col: WeightColumn = WeightColumn.NA):
        super().__init__(name="degree_central", category=Category.CONNECTEDNESS, weight_col=WeightColumn.NA)

    def _compute_metric(self, G):
        centralities = nx.degree_centrality(G)
        return _compute_freeman_centralisation(centralities)


class BetweenessCentralisation(GraphMetric):
    def __init__(self, weight_col: WeightColumn):
        super().__init__(name=f"betweeness_central_({weight_col.value})", category=Category.NA, weight_col=weight_col)

    def _compute_metric(self, G):
        centralities = nx.betweenness_centrality(G, weight=self.weight_col)
        return _compute_freeman_centralisation(centralities)


class ClosenessCentralisation(GraphMetric):
    def __init__(self, weight_col: WeightColumn):
        super().__init__(name=f"closeness_central_({weight_col.value})", category=Category.NA, weight_col=weight_col)

    def _compute_metric(self, G):
        centralities = nx.closeness_centrality(G, distance=self.weight_col)
        return _compute_freeman_centralisation(centralities)


class KatzCentralisation(GraphMetric):
    def __init__(self, weight_col: WeightColumn):
        super().__init__(name=f"katz_central_({weight_col.value})", category=Category.CLUSTERING, weight_col=weight_col)

    def _compute_metric(self, G):
        if G.number_of_nodes() <= 2:
            return None

        centralities = nx.katz_centrality_numpy(nx.DiGraph(G), weight=self.weight_col)
        return _compute_freeman_centralisation(centralities)


# Metric runners and computers


def _polars_exprs(polars_metrics: list[PolarsMetric]) -> list[pl.Expr]:
    return [metric.polars_expr() for metric in polars_metrics]


class Metrics:
    def __init__(
        self,
        node_metrics: list[NodeMetric],
        edge_metrics: list[EdgeMetric],
        graph_metrics: list[GraphMetric],
        grouping_col: str = "hh_id",
    ):
        self.grouping_col = grouping_col
        self.node_metrics = node_metrics
        self.edge_metrics = edge_metrics
        self.graph_metrics = graph_metrics

        metric_dict = {metric.name: metric.return_dtype for metric in graph_metrics}

        self._graph_schema = pl.Schema({grouping_col: pl.String} | metric_dict)

    @property
    def metrics(self) -> list[Metric]:
        return self.node_metrics + self.edge_metrics + self.graph_metrics

    def names(self) -> list[str]:
        return [metric.name for metric in self.metrics]

    def compute(
        self,
        graph: ActivityGraph,
        verbose: bool = False,
        max_iter: int | None = None,
        how: str = "inner",
    ) -> pl.DataFrame:
        node_metrics = self._compute_node_metrics(graph)
        edge_metrics = self._compute_edge_metrics(graph)
        graph_metrics = self._compute_graph_metrics(graph, verbose=verbose, max_iter=max_iter)

        return node_metrics.join(edge_metrics, on=self.grouping_col, how=how).join(
            graph_metrics, on=self.grouping_col, how=how
        )

    def _compute_node_metrics(self, graph: ActivityGraph) -> pl.DataFrame:
        metrics = _polars_exprs(self.node_metrics)
        grouped_nodes = graph.node_df.group_by(self.grouping_col)
        return grouped_nodes.agg(*metrics)

    def _compute_edge_metrics(self, graph: ActivityGraph) -> pl.DataFrame:
        metrics = _polars_exprs(self.edge_metrics)
        grouped_nodes = graph.edge_df.group_by(self.grouping_col)
        return grouped_nodes.agg(*metrics)

    def _compute_graph_metrics(self, graph: ActivityGraph, verbose: bool, max_iter: int | None) -> pl.DataFrame:
        results_generator = self._metrics_generator(graph, verbose, max_iter)

        if max_iter is not None:
            results_generator = itertools.islice(results_generator, max_iter)

        return pl.DataFrame(data=results_generator, schema=self._graph_schema)

    def _metrics_generator(
        self, graph: ActivityGraph, verbose: bool, max_iter: int | None
    ) -> Iterator[dict[str, int | float]]:
        total = max_iter if max_iter is not None else graph.n_subgraphs

        for hh_id, nx_graph in tqdm(graph.to_nxs(), total=total, disable=not verbose):
            metrics_dict = {metric.name: metric(nx_graph) for metric in self.graph_metrics}
            yield {self.grouping_col: hh_id} | metrics_dict


class AllMetrics(Metrics):
    def __init__(self, weight_col: WeightColumn, grouping_col: str = "hh_id"):
        node_metrics = [Order()]
        edge_metrics_cls = [Size, DistanceMax, DistanceMean, DistanceSum]
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

        edge_metrics = [metric_cls(weight_col) for metric_cls in edge_metrics_cls]
        graph_metrics = [metric_cls(weight_col) for metric_cls in graph_metrics_cls]

        super().__init__(node_metrics, edge_metrics, graph_metrics, grouping_col)
