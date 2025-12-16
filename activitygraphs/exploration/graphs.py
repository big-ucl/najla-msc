from typing import Iterator, Self

import exploration.dataprocessing as dp
import networkx as nx
import polars as pl

import activitygraphs.base

NODELIST_SCHEMA = pl.Schema({
    "hh_id": pl.String,
    "loc_id": pl.String,
    "is_home": pl.Boolean,
    "is_work": pl.Boolean,
    "land_uses": dp.LandUse.polars_enum(),
    "purposes": dp.Purpose.polars_enum(),
    "lat": pl.Float64(),
    "lon": pl.Float64(),
})

EDGELIST_SCHEMA = dp.TRIP_SCHEMA


class ActivityGraph:
    parent: Self | None
    node_df: pl.DataFrame
    edge_df: pl.DataFrame

    n_subgraphs: int

    def __init__(
        self,
        node_df: pl.DataFrame,
        edge_df: pl.DataFrame,
        parent: Self | None = None,
        name: str = None,
    ):
        self.parent = parent
        self.name = name
        self.n_subgraphs = edge_df.n_unique(subset="hh_id")

        self.node_df = dp.check_schema(node_df, NODELIST_SCHEMA)
        self.edge_df = dp.check_schema(edge_df, EDGELIST_SCHEMA)

    @classmethod
    def from_dataset(cls, dataset: dp.ActivityDataset, parent: Self | None = None):
        nodes, edges = _generate_node_and_edgelist(dataset.hh_person_df, dataset.trip_df)

        return cls(nodes, edges, name=dataset.name)

    def hh_ids(self) -> pl.Series:
        return pl.concat([self.edge_df["hh_id"], self.node_df["hh_id"]]).unique()

    def hh_graph(self, hh_id: str) -> Self:
        if self.parent:
            raise ValueError(
                f"Cannot create a nested household graph for {hh_id=} on an existing household graph name={self._dataset.name}"
            )

        name = self.name + f"-{hh_id}"
        node_df = self.node_df.filter(pl.col("hh_id") == hh_id)
        edge_df = self.edge_df.filter(pl.col("hh_id") == hh_id)

        return ActivityGraph(node_df, edge_df, parent=self, name=name)

    def partition_by_hh_id(self, n_chunks: int) -> list[Self]:
        chunk_length = len(self.hh_ids()) // n_chunks

        chunks_by_hh = self.hh_ids().to_frame().with_row_index("chunk").with_columns(pl.col("chunk") // chunk_length)

        node_dfs = _partition_df(self.node_df, chunks_by_hh, "hh_id")
        edge_dfs = _partition_df(self.edge_df, chunks_by_hh, "hh_id")

        if node_dfs.keys() != edge_dfs.keys():
            raise RuntimeError(
                "Number of paritions in `node_df` and `edge_df` is not equal. This is a bug, there might be too many nodes w/o edges"
            )

        graphs = []
        for chunk in [str(c) for c in range(n_chunks)]:
            node_df = node_dfs[chunk]
            edge_df = edge_dfs[chunk]

            name = self.name + f"-p{chunk}"
            graph = ActivityGraph(node_df, edge_df, name=name)
            graphs.append(graph)

        return graphs

    def to_nx(self, hh_id: str) -> nx.MultiDiGraph:
        _, G = next(self.hh_graph(hh_id).to_nxs())
        return G

    def to_nxs(self) -> Iterator[tuple[str, nx.MultiDiGraph]]:
        for (hh_id,), hh_edges in self.edge_df.group_by("hh_id"):
            hh_nodes = self.node_df.filter(pl.col("hh_id") == hh_id)
            yield (hh_id, _generate_hh_graph(hh_nodes, hh_edges))

    def __repr__(self):
        return f"ActivityGraph(dataset={self.name})"


def _partition_df(df: pl.DataFrame, chunks_by_id: pl.DataFrame, id_col: str, chunk_col: str = "chunk"):
    partitions = df.with_columns(
        pl.col(id_col).replace(old=chunks_by_id[id_col], new=chunks_by_id[chunk_col]).alias(chunk_col)
    ).partition_by(chunk_col, as_dict=True)

    return {chunk: part.drop(chunk_col) for (chunk,), part in partitions.items()}


def _generate_node_attribute_df(hh_person_df: pl.DataFrame, trip_df: pl.DataFrame) -> pl.DataFrame:
    origin_trip_nodes = trip_df.select(
        hh_id="hh_id",
        loc_id="loc_origin_loc_id",
        is_home=False,
        is_work=False,
        land_use="land_use",
        purpose="purpose",
        lat="loc_origin_lat",
        lon="loc_origin_lon",
    ).unique()

    dest_trip_nodes = trip_df.select(
        hh_id="hh_id",
        loc_id="loc_dest_loc_id",
        is_home=False,
        is_work=False,
        land_use=None,
        purpose="purpose_dest",
        lat="loc_destination_lat",
        lon="loc_destination_lon",
    ).unique()

    home_nodes = hh_person_df.select(
        hh_id="hh_id",
        loc_id="loc_home_loc_id",
        is_home=True,
        is_work=False,
        land_use=None,
        purpose=pl.lit(dp.Purpose.HOME, dtype=dp.Purpose.polars_enum()),
        lat="loc_home_lat",
        lon="loc_home_lon",
    ).unique()

    work_nodes = (
        hh_person_df.filter(pl.col("loc_work_loc_id") != "-1")
        .select(
            hh_id="hh_id",
            loc_id="loc_work_loc_id",
            is_home=False,
            is_work=True,
            land_use=None,
            purpose=pl.lit(dp.Purpose.WORK, dtype=dp.Purpose.polars_enum()),
            lat="loc_work_lat",
            lon="loc_work_lon",
        )
        .unique()
    )

    nodes = pl.concat([
        origin_trip_nodes,
        dest_trip_nodes,
        home_nodes,
        work_nodes,
    ])

    node_attributes_df = nodes.group_by(["hh_id", "loc_id"]).agg(
        pl.col("is_home").any(),
        pl.col("is_work").any(),
        pl.col("land_use").drop_nulls().bitwise_or().alias("land_uses"),
        pl.col("purpose").drop_nulls().bitwise_or().alias("purposes"),
        pl.col("lat").drop_nulls().first().alias("lat"),
        pl.col("lon").drop_nulls().first().alias("lon"),
    )

    return dp.check_schema(node_attributes_df, NODELIST_SCHEMA)


def _generate_node_and_edgelist(hh_person_df: pl.DataFrame, trip_df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    node_attribute_df = _generate_node_attribute_df(hh_person_df, trip_df)

    node_attribute_df = node_attribute_df.sort("hh_id", "loc_id")
    trip_df = trip_df.sort("hh_id", "person_id", "trip_id", "trip_number")

    return node_attribute_df, trip_df


def _generate_hh_graph(nodelist_df: pl.DataFrame, edgelist_df: pl.DataFrame) -> nx.MultiDiGraph:
    edgelist_df = edgelist_df.with_columns(
        pl.col("mode").map_elements(activitygraphs.mode.Mode, return_dtype=pl.Object),
    )

    G = nx.from_pandas_edgelist(
        edgelist_df,
        source="loc_origin_loc_id",
        target="loc_dest_loc_id",
        edge_key="trip_id",
        create_using=nx.MultiDiGraph,
        edge_attr=[
            "person_id",
            "mode",
            "duration",
            "distance",
            "start_time",
            "end_time",
        ],
    )

    node_attribute_dict = nodelist_df.with_columns(
        pl.col("purposes").map_elements(dp.Purpose, return_dtype=pl.Object),
        pl.col("land_uses").map_elements(dp.LandUse, return_dtype=pl.Object),
    ).rows_by_key(key="loc_id", named=True, unique=True)
    nx.set_node_attributes(G, node_attribute_dict)
    return G
