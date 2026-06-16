"""
Module: activitygraphs/archive/exploration/graphs.py

Description:
    Constructs activity graphs from processed travel-survey data.

    An "activity graph" represents the movement patterns of one or more households
    as a directed graph where:
      - Nodes are unique visited locations (home, work, trip origins/destinations).
      - Directed edges are individual trip legs connecting origin to destination.
      - Node attributes include whether the location is home/work, combined land-use
        flags, combined purpose flags, and geographic coordinates.
      - Edge attributes are the full trip record (mode, duration, distance, times).

    The module provides:
      - Schema constants (NODELIST_SCHEMA, EDGELIST_SCHEMA) defining expected columns.
      - ActivityGraph: a class that holds the aggregate node and edge DataFrames for
            an entire dataset and can produce individual per-household subgraphs
            or NetworkX MultiDiGraph objects for graph-metric computation.
      - Private helper functions that build the node and edge DataFrames and convert
            them to NetworkX format.
"""

from typing import Iterator, Self

from archive import exploration as dp
import networkx as nx
import polars as pl

import activitygraphs.base

# Schema for the node (location) table: one row per unique (household, location) pair.
# land_uses and purposes are bitwise-OR combined flags (UInt32) so that a location
# visited for multiple purposes has all relevant bits set.
NODELIST_SCHEMA = pl.Schema({
    "hh_id": pl.String,                     # Household identifier this node belongs to
    "loc_id": pl.String,                    # Unique location ID (zone, stop, address)
    "is_home": pl.Boolean,                  # True if this location is the household's home
    "is_work": pl.Boolean,                  # True if this location is the person's usual workplace
    "land_uses": dp.LandUse.polars_enum(),  # Bitwise-OR of all LandUse flags seen at this location
    "purposes": dp.Purpose.polars_enum(),   # Bitwise-OR of all Purpose flags seen at this location
    "lat": pl.Float64(),                    # WGS84 latitude of this location
    "lon": pl.Float64(),                    # WGS84 longitude of this location
})

# The edge table has the same schema as the trip table: each trip record is one edge
EDGELIST_SCHEMA = dp.TRIP_SCHEMA


class ActivityGraph:
    """
    Description:
        Holds the aggregate activity-graph data for one or more households.

        Internally, the graph is stored as two Polars DataFrames (node_df and edge_df)
        rather than a NetworkX object so that vectorised operations (filtering, grouping,
        schema validation) are fast even for large datasets.  NetworkX objects are only
        materialised on demand via to_nx() / to_nxs() for metric computation.

        An ActivityGraph can represent:
          - A full dataset (all households).
          - A single household's subgraph (created by hh_graph()).
          - A chunk/partition of households (created by partition_by_hh_id()).

        The `parent` attribute is set when this graph is a per-household subgraph,
        preventing further nesting (which would not make sense).

    Attributes:
      - parent (ActivityGraph | None): the parent graph this was sliced from, or None
            if this is a top-level full-dataset graph.
      - node_df (pl.DataFrame): validated node table (schema: NODELIST_SCHEMA).
      - edge_df (pl.DataFrame): validated edge/trip table (schema: EDGELIST_SCHEMA).
      - n_subgraphs (int): number of distinct households in this graph, i.e. the
            number of individual NetworkX graphs that to_nxs() will produce.
      - name (str | None): human-readable identifier, e.g. "ltds_2019" or
            "ltds_2019-HH001".
    """
    parent: Self | None   # Parent ActivityGraph if this is a household-level subgraph
    node_df: pl.DataFrame # Node table: one row per (household, location) pair
    edge_df: pl.DataFrame # Edge table: one row per trip leg

    n_subgraphs: int      # Count of distinct households (= number of NetworkX graphs)

    def __init__(
        self,
        node_df: pl.DataFrame,
        edge_df: pl.DataFrame,
        parent: Self | None = None,
        name: str = None,
    ):
        """
        Description:
            Creates an ActivityGraph from pre-built node and edge DataFrames,
            validating both against their expected schemas.

        Input:
          - node_df (pl.DataFrame): node table conforming to NODELIST_SCHEMA.
          - edge_df (pl.DataFrame): edge/trip table conforming to EDGELIST_SCHEMA.
          - parent (ActivityGraph | None): parent graph if this is a subgraph slice;
                None for top-level (full-dataset) graphs.
          - name (str | None): optional human-readable identifier for this graph.

        Output:
          - (ActivityGraph): the new ActivityGraph instance.
        """
        self.parent = parent  # Store reference to parent (None for full-dataset graphs)
        self.name = name      # Human-readable identifier (e.g. dataset name or "dataset-HH001")
        # Count distinct households in the edge table — each will become one NetworkX graph
        self.n_subgraphs = edge_df.n_unique(subset="hh_id")

        # Validate both tables against their schemas before storing
        self.node_df = dp.check_schema(node_df, NODELIST_SCHEMA)
        self.edge_df = dp.check_schema(edge_df, EDGELIST_SCHEMA)

    @classmethod
    def from_dataset(cls, dataset: dp.ActivityDataset, parent: Self | None = None):
        """
        Description:
            Convenience constructor that builds an ActivityGraph directly from an
            ActivityDataset by generating the node and edge DataFrames automatically.

        Input:
          - dataset (ActivityDataset): the processed travel-survey dataset to build
                the graph from. Uses dataset.hh_person_df and dataset.trip_df.
          - parent (ActivityGraph | None): optional parent graph (rarely needed here).

        Output:
          - (ActivityGraph): a new top-level ActivityGraph for the entire dataset,
                named after the dataset.
        """
        # Build node and edge DataFrames from the household-person and trip tables
        nodes, edges = _generate_node_and_edgelist(dataset.hh_person_df, dataset.trip_df)

        return cls(nodes, edges, name=dataset.name)

    def hh_ids(self) -> pl.Series:
        """
        Description:
            Returns all unique household IDs present in this graph, combining both
            the node and edge tables to catch households that have nodes but no trips
            or vice versa.

        Output:
          - (pl.Series): a Series of unique household ID strings.
        """
        # Union the hh_id columns from both tables to avoid missing any households
        return pl.concat([self.edge_df["hh_id"], self.node_df["hh_id"]]).unique()

    def hh_graph(self, hh_id: str) -> Self:
        """
        Description:
            Extracts a single household's activity graph as a new ActivityGraph
            containing only the nodes and edges belonging to that household.

        Input:
          - hh_id (str): the household identifier to filter on.

        Output:
          - (ActivityGraph): a new ActivityGraph for the single household, with
                `parent` set to this graph.

        Raises:
          - ValueError: if this graph already has a parent (i.e. it is already a
                household-level subgraph — nesting is not allowed).
        """
        if self.parent:
            raise ValueError(
                f"Cannot create a nested household graph for {hh_id=} on an existing household graph name={self._dataset.name}"
            )

        # Append the household ID to the name so the subgraph is identifiable
        name = self.name + f"-{hh_id}"
        # Filter both tables to only include rows for this household
        node_df = self.node_df.filter(pl.col("hh_id") == hh_id)
        edge_df = self.edge_df.filter(pl.col("hh_id") == hh_id)

        return ActivityGraph(node_df, edge_df, parent=self, name=name)

    def partition_by_hh_id(self, n_chunks: int) -> list[Self]:
        """
        Description:
            Splits this ActivityGraph into n_chunks roughly equal-sized ActivityGraph
            objects by dividing the household IDs into chunks. This is used by
            parallel_to_nx() in multi.py to distribute work across CPU cores.

        Input:
          - n_chunks (int): number of partitions to create. Typically set to the
                number of available CPU cores (os.cpu_count()).

        Output:
          - (list[ActivityGraph]): a list of n_chunks ActivityGraph objects, each
                containing approximately 1/n_chunks of the households.

        Raises:
          - RuntimeError: if the node and edge partitioning produce different sets
                of chunk keys (indicates a data integrity bug).
        """
        # Determine how many households each chunk should contain
        chunk_length = len(self.hh_ids()) // n_chunks

        # Assign a chunk index (0 .. n_chunks-1) to each household ID using integer division
        chunks_by_hh = self.hh_ids().to_frame().with_row_index("chunk").with_columns(pl.col("chunk") // chunk_length)

        # Partition both DataFrames by the chunk index assigned to each household
        node_dfs = _partition_df(self.node_df, chunks_by_hh, "hh_id")
        edge_dfs = _partition_df(self.edge_df, chunks_by_hh, "hh_id")

        if node_dfs.keys() != edge_dfs.keys():
            raise RuntimeError(
                "Number of paritions in `node_df` and `edge_df` is not equal. This is a bug, there might be too many nodes w/o edges"
            )

        graphs = []  # Accumulator for the resulting ActivityGraph objects
        for chunk in [str(c) for c in range(n_chunks)]:
            node_df = node_dfs[chunk]  # Node rows for this chunk's households
            edge_df = edge_dfs[chunk]  # Edge rows for this chunk's households

            name = self.name + f"-p{chunk}"  # e.g. "ltds_2019-p0", "ltds_2019-p1"
            graph = ActivityGraph(node_df, edge_df, name=name)
            graphs.append(graph)

        return graphs

    def to_nx(self, hh_id: str) -> nx.MultiDiGraph:
        """
        Description:
            Materialises a single household's activity graph as a NetworkX
            MultiDiGraph. Convenience wrapper around to_nxs() for single lookups.

        Input:
          - hh_id (str): the household ID whose graph to convert.

        Output:
          - (nx.MultiDiGraph): directed multigraph where nodes are location IDs
                and each edge is one trip leg with attributes (mode, duration, etc.).
        """
        # to_nxs() is a generator; next() gets the first (and only) result
        _, G = next(self.hh_graph(hh_id).to_nxs())
        return G

    def to_nxs(self) -> Iterator[tuple[str, nx.MultiDiGraph]]:
        """
        Description:
            Generator that yields (hh_id, NetworkX graph) pairs for every household
            in this ActivityGraph. Use this to iterate over all households without
            materialising all graphs at once (memory-efficient).

        Output:
          - (Iterator[tuple[str, nx.MultiDiGraph]]): yields (hh_id, G) tuples where
                hh_id is the household string identifier and G is the NetworkX
                MultiDiGraph for that household.
        """
        # Group edges by household and build one NetworkX graph per group
        for (hh_id,), hh_edges in self.edge_df.group_by("hh_id"):
            hh_nodes = self.node_df.filter(pl.col("hh_id") == hh_id)  # Matching node rows
            yield (hh_id, _generate_hh_graph(hh_nodes, hh_edges))

    def __repr__(self):
        """
        Description: Returns a concise human-readable string representation of this
        ActivityGraph. This is what Python displays when you print the object or inspect
        it in a REPL. The string includes the dataset name so you can quickly identify
        which dataset this graph corresponds to.

        Output:
          - (str): A string of the form "ActivityGraph(dataset=<name>)", e.g.
                "ActivityGraph(dataset=ltds_2019)".
        """
        return f"ActivityGraph(dataset={self.name})"


def _partition_df(df: pl.DataFrame, chunks_by_id: pl.DataFrame, id_col: str, chunk_col: str = "chunk"):
    """
    Description:
        Splits a DataFrame into a dictionary of sub-DataFrames according to a
        pre-computed chunk assignment table. Each row of `df` is assigned to a
        chunk based on the value of its `id_col` column.

    Input:
      - df (pl.DataFrame): the DataFrame to partition (e.g. node_df or edge_df).
      - chunks_by_id (pl.DataFrame): a two-column DataFrame with columns [id_col,
            chunk_col] mapping each unique ID to its chunk number. Produced by
            partition_by_hh_id() using with_row_index().
      - id_col (str): name of the column in `df` that holds the grouping ID
            (e.g. "hh_id").
      - chunk_col (str): name of the chunk-index column in chunks_by_id (default "chunk").

    Output:
      - (dict[str, pl.DataFrame]): mapping from chunk index string (e.g. "0", "1")
            to the subset of `df` whose `id_col` values belong to that chunk.
            The temporary chunk column is dropped from each sub-DataFrame.
    """
    # Temporarily join the chunk assignment onto each row, then split by chunk value
    partitions = df.with_columns(
        pl.col(id_col).replace(old=chunks_by_id[id_col], new=chunks_by_id[chunk_col]).alias(chunk_col)
    ).partition_by(chunk_col, as_dict=True)

    # Drop the temporary chunk column from each partition before returning
    return {chunk: part.drop(chunk_col) for (chunk,), part in partitions.items()}


def _generate_node_attribute_df(hh_person_df: pl.DataFrame, trip_df: pl.DataFrame) -> pl.DataFrame:
    """
    Description:
        Builds the node (location) attribute DataFrame that forms the vertex set
        of the activity graph.

        Nodes are collected from four sources:
          1. Trip origins  — all unique (hh_id, origin loc_id) pairs from trips.
          2. Trip destinations — all unique (hh_id, dest loc_id) pairs from trips.
          3. Home locations — from the household/person table.
          4. Work locations — from the household/person table (skipping persons
             with no fixed workplace, indicated by loc_work_loc_id == "-1").

        Because the same location may appear multiple times for the same household
        (e.g. as both an origin and a destination, or visited for multiple purposes),
        the rows are grouped by (hh_id, loc_id) and the attributes are aggregated:
          - is_home / is_work: True if any source marks it as home/work.
          - land_uses / purposes: bitwise-OR of all flag values seen.
          - lat / lon: first non-null coordinate.

    Input:
      - hh_person_df (pl.DataFrame): household/person table (schema: HH_PERSON_SCHEMA).
      - trip_df (pl.DataFrame): trip-records table (schema: TRIP_SCHEMA).

    Output:
      - (pl.DataFrame): node attribute table conforming to NODELIST_SCHEMA, with one
            row per unique (household, location) pair.
    """
    # ── Source 1: trip origin locations ──────────────────────────────────────
    # Each origin has a land_use and a purpose recorded at departure
    origin_trip_nodes = trip_df.select(
        hh_id="hh_id",
        loc_id="loc_origin_loc_id",
        is_home=False,       # Trip origins are not flagged as home by default
        is_work=False,       # Trip origins are not flagged as work by default
        land_use="land_use", # Land-use category at the origin
        purpose="purpose",   # Activity purpose at the origin
        lat="loc_origin_lat",
        lon="loc_origin_lon",
    ).unique()  # Remove exact duplicate rows

    # ── Source 2: trip destination locations ─────────────────────────────────
    # Destinations have a purpose_dest but no land_use in the survey
    dest_trip_nodes = trip_df.select(
        hh_id="hh_id",
        loc_id="loc_dest_loc_id",
        is_home=False,
        is_work=False,
        land_use=None,              # Land use not recorded for destinations
        purpose="purpose_dest",     # Activity purpose at the destination
        lat="loc_destination_lat",
        lon="loc_destination_lon",
    ).unique()

    # ── Source 3: home locations from household table ────────────────────────
    home_nodes = hh_person_df.select(
        hh_id="hh_id",
        loc_id="loc_home_loc_id",
        is_home=True,               # Flag this location as the household's home
        is_work=False,
        land_use=None,              # Land use not recorded in HH table
        purpose=pl.lit(dp.Purpose.HOME, dtype=dp.Purpose.polars_enum()),  # Fixed HOME flag
        lat="loc_home_lat",
        lon="loc_home_lon",
    ).unique()

    # ── Source 4: work locations from household table (skip persons w/o workplace) ──
    work_nodes = (
        hh_person_df
        .filter(pl.col("loc_work_loc_id") != "-1")  # "-1" encodes "no workplace"
        .select(
            hh_id="hh_id",
            loc_id="loc_work_loc_id",
            is_home=False,
            is_work=True,           # Flag this location as the person's workplace
            land_use=None,
            purpose=pl.lit(dp.Purpose.WORK, dtype=dp.Purpose.polars_enum()),  # Fixed WORK flag
            lat="loc_work_lat",
            lon="loc_work_lon",
        )
        .unique()
    )

    # ── Combine all four sources into one long table ──────────────────────────
    nodes = pl.concat([
        origin_trip_nodes,
        dest_trip_nodes,
        home_nodes,
        work_nodes,
    ])

    # ── Aggregate: one row per (household, location) ─────────────────────────
    # For each unique (hh_id, loc_id) pair, merge the per-source attribute rows:
    node_attributes_df = nodes.group_by(["hh_id", "loc_id"]).agg(
        pl.col("is_home").any(),                                    # True if any source says it's home
        pl.col("is_work").any(),                                    # True if any source says it's work
        pl.col("land_use").drop_nulls().bitwise_or().alias("land_uses"),  # OR all non-null land-use flags
        pl.col("purpose").drop_nulls().bitwise_or().alias("purposes"),    # OR all non-null purpose flags
        pl.col("lat").drop_nulls().first().alias("lat"),            # Use first available latitude
        pl.col("lon").drop_nulls().first().alias("lon"),            # Use first available longitude
    )

    return dp.check_schema(node_attributes_df, NODELIST_SCHEMA)


def _generate_node_and_edgelist(hh_person_df: pl.DataFrame, trip_df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Description:
        Top-level factory function that produces the sorted node and edge DataFrames
        needed to construct an ActivityGraph from raw survey tables.

        Delegates node-attribute computation to _generate_node_attribute_df(), then
        sorts both tables for deterministic ordering (important for reproducibility
        when comparing graph structures across runs).

    Input:
      - hh_person_df (pl.DataFrame): household/person table (schema: HH_PERSON_SCHEMA).
      - trip_df (pl.DataFrame): trip-records table (schema: TRIP_SCHEMA).

    Output:
      - (tuple[pl.DataFrame, pl.DataFrame]): a pair (node_df, edge_df) where
            node_df conforms to NODELIST_SCHEMA and edge_df conforms to EDGELIST_SCHEMA,
            both sorted for deterministic ordering.
    """
    # Build the aggregated node attributes table from both input tables
    node_attribute_df = _generate_node_attribute_df(hh_person_df, trip_df)

    # Sort both tables for deterministic ordering across runs
    node_attribute_df = node_attribute_df.sort("hh_id", "loc_id")
    trip_df = trip_df.sort("hh_id", "person_id", "trip_id", "trip_number")

    return node_attribute_df, trip_df


def _generate_hh_graph(nodelist_df: pl.DataFrame, edgelist_df: pl.DataFrame) -> nx.MultiDiGraph:
    """
    Description:
        Converts a single household's node and edge DataFrames into a NetworkX
        MultiDiGraph ready for graph-metric computation.

        Steps:
          1. Convert the integer `mode` column back to the Mode enum object (so
             that NetworkX attributes are typed enum values, not raw integers).
          2. Build the directed multigraph from the edge list using NetworkX's
             from_pandas_edgelist, keyed on trip_id to support multiple trips
             between the same pair of locations.
          3. Convert the integer `purposes` and `land_uses` columns back to enum
             objects, then attach all node attributes to the graph.

    Input:
      - nodelist_df (pl.DataFrame): node attribute table for one household
            (conforms to NODELIST_SCHEMA).
      - edgelist_df (pl.DataFrame): trip edge table for one household
            (conforms to EDGELIST_SCHEMA / TRIP_SCHEMA).

    Output:
      - (nx.MultiDiGraph): a directed multigraph where:
            - Each node is a location ID string with attributes from nodelist_df.
            - Each edge is a trip leg keyed by trip_id, with attributes
              (person_id, mode, duration, distance, start_time, end_time).
    """
    # Convert the integer mode column back to Mode enum objects for richer attribute types
    edgelist_df = edgelist_df.with_columns(
        pl.col("mode").map_elements(activitygraphs.mode.Mode, return_dtype=pl.Object),
    )

    # Build the MultiDiGraph from the edge list; each trip becomes one directed edge
    G = nx.from_pandas_edgelist(
        edgelist_df,          # Polars DataFrame (converted internally to pandas)
        source="loc_origin_loc_id",   # Column used as the edge source node ID
        target="loc_dest_loc_id",     # Column used as the edge target node ID
        edge_key="trip_id",           # Unique key to distinguish parallel edges
        create_using=nx.MultiDiGraph, # Allow multiple directed edges between same pair
        edge_attr=[
            "person_id",   # Which person in the household made this trip
            "mode",        # Transport mode (Mode enum)
            "duration",    # Trip duration in minutes
            "distance",    # Trip distance
            "start_time",  # Departure time
            "end_time",    # Arrival time
        ],
    )

    # Convert integer purpose/land_use flags back to enum objects for richer node attributes
    node_attribute_dict = nodelist_df.with_columns(
        pl.col("purposes").map_elements(dp.Purpose, return_dtype=pl.Object),
        pl.col("land_uses").map_elements(dp.LandUse, return_dtype=pl.Object),
    ).rows_by_key(key="loc_id", named=True, unique=True)  # Dict keyed by loc_id for nx.set_node_attributes

    # Attach node attributes (is_home, is_work, purposes, land_uses, lat, lon) to the graph
    nx.set_node_attributes(G, node_attribute_dict)
    return G
