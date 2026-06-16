import math
from collections.abc import Generator, Hashable, Iterable, Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, Self, Type

import geopandas as gpd
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.data.storage import BaseStorage, EdgeStorage, NodeStorage
from tqdm import tqdm

from activitygraphs.base import USER_JOURNEY_SCHEMA, Mode
from activitygraphs.config import DataConfig
from activitygraphs.data.gtfs import PARENT_STOP_ROUTE_ID
from archive.network import LayerType, Network
from activitygraphs.utils import check_schema

type Encoder = Callable[[pd.Series | pl.Series], torch.Tensor]

type LocID = str | tuple[str, str]
type NodeMapping = Mapping[LocID, int]
type LocIDMapping = Mapping[LocID, tuple[LayerType, int]]

type NodeProcessor = Callable[
    [Network, Sequence[str], Mapping[str, Encoder] | None], tuple[NodeMapping, torch.Tensor | None]
]
type EdgeProcessor = Callable[
    [Network, Sequence[str], NodeMapping, Mapping[str, Encoder] | None],
    dict[str, tuple[torch.Tensor, torch.Tensor]],
]

# Default dimensionality for learned embeddings (e.g. transport mode, location type, layer name)
EMBEDDING_DIM = 3
# Column name added to layer location GeoDataFrames to identify which network layer each row belongs to
LAYER_NAME_COL = "layer_name"


class ActivityGraphBuilder:
    """
    Description: Builds annotated PyG HeteroData graph objects for each user in the dataset.
    Starting from a pre-built base graph (created by `network_to_pyg`), this class adds per-user
    node features (home location indicator) and per-user node labels (visited locations) to
    create individualised graph instances. Iterating via `annotated_graphs()` yields one HeteroData
    graph per user, ready for use in model training or inference.
    """
    def __init__(self, base_data: HeteroData, user_journeys_df: pl.DataFrame, separate_na_source_sink: bool):
        """
        Description: Initialises the ActivityGraphBuilder, validates the user journeys schema,
        and pre-processes user attributes and visited locations for later graph annotation.

        Input:
          - base_data (HeteroData): The pre-built heterogeneous graph (from `network_to_pyg`)
            containing the transport network structure (nodes, edges, encodings). Will be cloned
            and used as the template for each user's annotated graph.
          - user_journeys_df (pl.DataFrame): A DataFrame of user journey records conforming to
            USER_JOURNEY_SCHEMA. Contains origin, destination, purpose, and mode information.
          - separate_na_source_sink (bool): If True, treats 'NA' origin locations as 'NA_SOURCE'
            and 'NA' destination locations as 'NA_SINK' (separates unknown sources/sinks).
        """
        check_schema(user_journeys_df, USER_JOURNEY_SCHEMA)

        self.base_data = base_data.clone()  # Clone to avoid modifying the shared base graph

        # Extract two lookup dictionaries from the graph's location ID mapping:
        loc_id_mapping: LocIDMapping = base_data.graph_metadata["loc_id_mapping"]
        # Maps each location ID (string) to its layer type (e.g. LayerType.PLANAR)
        self._loc_id_to_layer_type = {k: t for k, (t, _) in loc_id_mapping.items()}
        # Maps each location ID (string) to its integer node index within its layer
        self._loc_id_to_node_index = {k: i for k, (_, i) in loc_id_mapping.items()}

        # Build user-level attributes (home, work, study locations) from journey records
        self._user_attributes = _build_user_attributes(user_journeys_df, separate_na_source_sink)
        # Build per-user visited location sequences from journey records
        self._visited_locations = _build_visited_locations(user_journeys_df, separate_na_source_sink)
        # Keep only users for which we have both attributes AND visited locations
        self._visited_locations = self._visited_locations.filter(
            pl.col("user_id").is_in(self._user_attributes["user_id"].implode())
        )

        # Unique user IDs for which we can build annotated graphs
        self._user_ids: list[str] = self._visited_locations["user_id"].unique().to_list()

    @property
    def user_ids(self) -> list[str]:
        """
        Description: Returns a copy of the list of user IDs for which annotated graphs can
        be generated. Returns a copy to prevent external modification.

        Output:
          - (list[str]): A list of unique user ID strings.
        """
        return self._user_ids.copy()

    def num_users(self):
        """
        Description: Returns the number of users in this builder (i.e. the number of annotated
        graphs that will be generated).

        Output:
          - (int): Number of distinct users.
        """
        return len(self._user_ids)

    def __len__(self) -> int:
        """
        Description: Returns the number of users in this builder. Allows use of Python's
        built-in `len()` function on the builder object.

        Output:
          - (int): Number of distinct users.
        """
        return self.num_users()

    def annotated_graphs(self) -> Generator[HeteroData]:
        """
        Description: A generator that yields one annotated HeteroData graph per user. For each
        user, the base graph is cloned and two sets of annotations are added:
          1. Home location indicator features (which nodes are marked as the user's home).
          2. Visited location labels (which nodes were actually visited by the user).
        These annotated graphs are used as training/inference samples for the GNN model.

        Output:
          - (Generator[HeteroData]): Yields one PyG HeteroData per user, with user-specific
            features and labels added on top of the shared base graph structure.
        """
        for (user_id,), user_visited_locations in self._visited_locations.group_by("user_id"):
            user_attrs = self._user_attributes.filter(user_id=user_id)

            data = self.base_data.clone()
            data.graph_metadata["user_id"] = user_id

            _add_location_indicator_features(
                data, user_attrs, "home_loc_id", self._loc_id_to_layer_type, self._loc_id_to_node_index
            )
            _add_visited_location_labels(
                data, user_visited_locations, self._loc_id_to_layer_type, self._loc_id_to_node_index
            )

            yield data


class ActivityDataset(Dataset):
    """
    Description: A PyG on-disk Dataset that stores one annotated HeteroData graph per user.
    Graphs are serialised to disk during `process()` and loaded on demand via `get()`. This allows
    working with large datasets that don't fit in memory. Provides two construction paths:
      - `from_builder`: processes raw graphs from an ActivityGraphBuilder and saves them.
      - `from_files`: reloads previously processed graphs from an existing directory.
    """
    def __init__(
        self,
        user_ids: list[str],
        graphs: Optional[Iterable[HeteroData]] = None,
        root: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        log: bool = True,
        force_reload: bool = False,
    ) -> None:
        """
        Description: Initialises the ActivityDataset. Sets up internal mappings from user IDs
        to file names, then delegates to PyG's Dataset __init__ (which calls process() if needed).

        Input:
          - user_ids (list[str]): List of user IDs to include in this dataset.
          - graphs (Iterable[HeteroData] | None): An iterable of HeteroData graphs to process and
            save. Required when creating a new dataset; set to None when loading from files.
          - root (str | None): Directory path where processed files are stored. PyG uses this to
            manage 'raw' and 'processed' subdirectories.
          - transform (Callable | None): Optional transform applied to graphs when loading.
          - pre_transform (Callable | None): Optional transform applied during processing/saving.
          - log (bool): If True, prints progress information. Defaults to True.
          - force_reload (bool): If True, reprocesses even if processed files already exist.
            Defaults to False.
        """
        self._set_user_ids(user_ids)
        self._graphs = graphs

        super().__init__(root, transform, pre_transform, pre_filter=None, log=log, force_reload=force_reload)

    def get(self, idx: int) -> HeteroData:
        """
        Description: Loads and returns the HeteroData graph for the given integer index by
        deserialising it from the corresponding `.pt` file on disk. Uses PyTorch's safe globals
        mechanism to allow deserialisation of custom types (LayerType, BaseStorage, etc.).

        Input:
          - idx (int): The integer index of the sample to load.

        Output:
          - (HeteroData): The annotated heterogeneous graph for the user at this index.
        """
        path = Path(self.processed_dir) / self._index_filename(idx)

        with torch.serialization.safe_globals([BaseStorage, EdgeStorage, NodeStorage, LayerType]):
            return torch.load(path)

    def len(self) -> int:
        """
        Description: Returns the total number of processed graphs (users) in this dataset.

        Output:
          - (int): Number of graphs stored in the dataset.
        """
        return len(self._processed_file_names)

    def process(self) -> None:
        """
        Description: Processes and saves all graphs in self._graphs to disk. Called automatically
        by PyG's Dataset if processed files are not found. Iterates through all annotated graphs,
        applies any pre_transform, and saves each one as a `.pt` file named by user ID.
        Updates the internal user ID list once processing is complete.

        Output:
          - None. Saves .pt files to self.processed_dir and updates self._user_ids.
        """
        user_ids = []  # Will collect user IDs in the order they are processed

        if self._graphs is None:
            raise ValueError("No iterable of Graphs provided, cannot process graphs.")

        for data in tqdm(self._graphs, total=self.len()):
            user_id = data.graph_metadata["user_id"]  # Retrieve the user ID stored in graph metadata

            if self.pre_transform is not None:
                data = self.pre_transform(data)  # Apply optional pre-processing transform

            # Save the graph as a PyTorch .pt file, named after the user ID
            path = Path(self.processed_dir) / self._user_filename(user_id)
            torch.save(data, path)
            user_ids.append(user_id)

        self._set_user_ids(user_ids)  # Update internal user ID list and file name mappings

    @property
    def processed_file_names(self) -> str | list[str] | tuple[str, ...]:
        """
        Description: Returns the list of expected processed file names. PyG uses this to check
        whether processing has already been done (if all files exist, process() is skipped).

        Output:
          - (list[str]): List of processed file names (e.g. ['data_user123.pt', ...]).
        """
        return self._processed_file_names

    def _set_user_ids(self, user_ids: list[str]) -> None:
        """
        Description: Internal method that updates the dataset's user ID list and rebuilds the
        derived index-to-user and file-name mappings. Called during __init__ and after process().

        Input:
          - user_ids (list[str]): The list of user IDs to store.

        Output:
          - None. Updates self._user_ids, self._user_indices, and self._processed_file_names.
        """
        self._user_ids = user_ids  # Ordered list of user IDs
        # Maps integer index → user ID, used to retrieve files in get(idx)
        self._user_indices = {i: user_id for i, user_id in enumerate(self._user_ids)}
        # List of .pt file names corresponding to each user, in the same order as _user_ids
        self._processed_file_names = [self._user_filename(user_id) for user_id in self._user_ids]

    @classmethod
    def from_builder(
        cls,
        graph_builder: ActivityGraphBuilder,
        cfg: DataConfig,
        project_root: Path | None = None,
        name: str | None = None,
        transform: Optional[Callable] = None,
        log: bool = True,
    ) -> Self:
        """
        Description: Class method that creates an ActivityDataset from an ActivityGraphBuilder,
        processing and saving all graphs to disk. Use this to build the dataset for the first time.
        The on-disk location is determined by cfg.paths.processed and the optional name suffix.

        Input:
          - graph_builder (ActivityGraphBuilder): Builder providing the annotated graphs to process.
          - cfg (DataConfig): Data configuration object specifying file paths.
          - project_root (Path | None): Root of the project directory. Defaults to '../..'.
          - name (str | None): Optional suffix appended to the dataset directory name.
          - transform (Callable | None): Optional transform applied when loading graphs.
          - log (bool): Whether to log progress. Defaults to True.

        Output:
          - (Self): A new ActivityDataset with all graphs processed and saved to disk.
        """
        root = cls._dir(cfg, project_root, name)

        return cls(
            graph_builder.user_ids,
            graph_builder.annotated_graphs(),
            root=str(root),
            transform=transform,
            pre_transform=None,
            log=log,
            force_reload=False,
        )

    @classmethod
    def from_files(
        cls,
        cfg: DataConfig,
        project_root: Path | None = None,
        name: str | None = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        log: bool = True,
        force_reload: bool = False,
    ) -> Self:
        """
        Description: Class method that loads an existing ActivityDataset from previously processed
        files on disk, without needing to reprocess. Scans the processed directory for .pt files,
        infers user IDs from filenames, and initialises the dataset pointing to those files.
        Use this after the dataset has already been created with `from_builder`.

        Input:
          - cfg (DataConfig): Data configuration object specifying file paths.
          - project_root (Path | None): Root of the project directory. Defaults to '../..'.
          - name (str | None): Optional suffix that was used when building the dataset.
          - transform (Callable | None): Optional transform applied when loading graphs.
          - pre_transform (Callable | None): Optional pre-transform (usually None for loaded data).
          - log (bool): Whether to log progress. Defaults to True.
          - force_reload (bool): If True, forces re-processing even if files exist. Defaults to False.

        Output:
          - (Self): An ActivityDataset backed by existing processed .pt files on disk.
        """
        root_dir = cls._dir(cfg, project_root, name)
        processed_graphs_dir = root_dir / "processed"

        non_user_ids = ["pre_filter.pt", "pre_transform.pt"]

        filenames = (file.name for file in processed_graphs_dir.iterdir())
        filtered_names = (f for f in filenames if f not in non_user_ids)
        user_ids = [cls._filename_to_user_id(f) for f in filtered_names]

        return cls(user_ids, None, str(root_dir), transform, pre_transform, log, force_reload)

    @classmethod
    def _dir(cls, cfg: DataConfig, project_root: Path | None = None, name: str | None = None):
        """
        Description: Internal class method that constructs the full directory path for the dataset.
        The directory is named after the class (e.g. 'ActivityDataset') with an optional suffix,
        nested under cfg.paths.processed within the project root.

        Input:
          - cfg (DataConfig): Configuration object providing the processed data directory path.
          - project_root (Path | None): Project root path. Defaults to Path('../..').
          - name (str | None): Optional name suffix for the directory (e.g. 'train', 'test').

        Output:
          - (Path): The full directory path for this dataset's files.
        """
        project_root = project_root if project_root is not None else Path("../..")
        suffix = "" if name is None else f"-{name}"  # Append '-name' suffix if provided
        return project_root / cfg.paths.processed / f"{cls.__name__}{suffix}"

    @staticmethod
    def _filename_to_user_id(filename: str) -> str:
        """
        Description: Extracts the user ID from a processed file name by stripping the 'data_'
        prefix and '.pt' suffix. This is the inverse of `_user_filename`.

        Input:
          - filename (str): A processed file name, e.g. 'data_user123.pt'.

        Output:
          - (str): The user ID string, e.g. 'user123'.
        """
        return filename.removeprefix("data_").removesuffix(".pt")

    @staticmethod
    def _user_filename(user_id: str) -> str:
        """
        Description: Constructs the .pt file name for a given user ID by wrapping it with
        the 'data_' prefix and '.pt' suffix.

        Input:
          - user_id (str): A user identifier string.

        Output:
          - (str): The file name, e.g. 'data_user123.pt'.
        """
        return f"data_{user_id}.pt"

    def _index_filename(self, idx: int) -> str:
        """
        Description: Returns the file name for the graph at integer index `idx`, by looking up
        the corresponding user ID and calling `_user_filename`.

        Input:
          - idx (int): The integer index of the sample.

        Output:
          - (str): The .pt file name for the graph at this index.
        """
        return f"data_{self._user_indices[idx]}.pt"


class EnumEncoder:
    """
    Description: An encoder that converts categorical values (enum instances or hashable objects)
    into dense embedding vectors using a learnable nn.Embedding table. Each unique category is
    assigned an integer index, then looked up in the embedding matrix to produce a dense vector
    of size embedding_dim. This allows the model to learn meaningful representations of categories
    like transport mode, location type, or layer name during training.
    """
    def __init__(self, enum_cls: Type[Enum] | list[Hashable], embedding_dim: int):
        """
        Description: Initialises the EnumEncoder with a set of categories and embedding size.

        Input:
          - enum_cls (Type[Enum] | list[Hashable]): An Enum class or any iterable of hashable
            values representing the categories to encode (e.g. Mode enum, list of layer names).
          - embedding_dim (int): The dimensionality of the output embedding vectors.
        """
        self._enum_cls = enum_cls  # The source of categories (used for iteration)
        # Map each category to a unique integer index (0, 1, 2, ...)
        self.mapping = {enum: index for index, enum in enumerate(enum_cls)}

        self.num_embeddings = len(self.mapping)  # Total number of distinct categories
        self.embedding_dim = embedding_dim  # Output vector size per category

        # Learnable embedding table: shape=(num_embeddings, embedding_dim)
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)

    def __call__(self, enum_series: pd.Series | pl.Series) -> torch.Tensor:
        """
        Description: Encodes a Series of categorical values into embedding vectors. Maps each
        value to its integer index using self.mapping, then looks it up in the embedding table.

        Input:
          - enum_series (pd.Series | pl.Series): A series of categorical values (e.g. Mode.WALK,
            Mode.TRANSIT, ...) that must all be present in the mapping.

        Output:
          - (torch.Tensor): Dense embedding vectors, shape=(len(enum_series), embedding_dim).
        """
        # Convert polars series to pandas for .map() compatibility
        enum_series = enum_series if isinstance(enum_series, pd.Series) else enum_series.to_pandas()

        # Map each category value to its integer index
        indices = enum_series.map(self.mapping).to_numpy()
        if indices.dtype != "int":
            raise ValueError("Conversion to indices failed, some values not in mapping ")

        # Move indices to the same device as the embedding weights (CPU or GPU)
        device = self.embedding.weight.device
        indices = torch.tensor(indices, device=device, dtype=torch.long)
        return self.embedding(indices)  # Look up embedding vectors for each index


class PTStopEncoder:
    """
    Description: Placeholder encoder class for public transport stop features. Not yet implemented.
    Reserved for future work encoding PT stop-specific attributes (e.g. number of routes, stop type).
    """
    pass


class RouteEncoder:
    """
    Description: Placeholder encoder class for transit route features. Not yet implemented.
    Reserved for future work encoding route-specific attributes (e.g. frequency, capacity, mode).
    """
    pass


class TimeOfDayEncoder:
    """
    Description: Encodes a time-of-day value (datetime) into a 2D cyclic representation using
    sine and cosine of the normalised time. This preserves the circular nature of time
    (e.g. 23:59 and 00:01 are close together). Produces two features per time value:
    sin(2π * t / 86400) and cos(2π * t / 86400), where t is total seconds since midnight.
    This is a standard technique for encoding periodic features in machine learning.
    """
    # noinspection PyUnresolvedReferences
    def __call__(self, tod_series: pd.Series | pl.Series) -> torch.Tensor:
        """
        Description: Encodes a series of datetime values as cyclic time-of-day features.
        Extracts hours, minutes, and seconds, converts to total seconds since midnight,
        then applies sine and cosine transformations to produce a 2D periodic representation.

        Input:
          - tod_series (pd.Series | pl.Series): A series of datetime values representing
            departure or arrival times.

        Output:
          - (torch.Tensor): A 2D tensor of shape=(len(tod_series), 2) where each row contains
            [sin(2π*t/86400), cos(2π*t/86400)] for the time t in seconds since midnight.
        """
        # Convert to polars Series for consistent datetime arithmetic
        tod_series = tod_series if isinstance(tod_series, pl.Series) else pl.from_pandas(tod_series)
        # Convert datetime to total seconds since midnight
        total_seconds = (
            3600 * tod_series.dt.hour().cast(pl.UInt32)
            + 60 * tod_series.dt.minute().cast(pl.UInt32)
            + tod_series.dt.second().cast(pl.UInt32)
        )

        seconds_in_day = 24 * 60 * 60  # Total seconds in a 24-hour day (86400)
        # Compute cyclic sine and cosine features (captures periodicity of time)
        sines = (2 * math.pi * total_seconds / seconds_in_day).sin()
        cosines = (2 * math.pi * total_seconds / seconds_in_day).cos()

        # Stack into a (N, 2) tensor: column 0 = sine, column 1 = cosine
        return torch.stack([sines.to_torch(), cosines.to_torch()], dim=1)


class GeometryEncoder:
    """
    Description: Encodes geographic geometry objects (polygon/point shapes from GeoPandas) into
    a numeric area feature. Reprojects geometries to the appropriate UTM coordinate reference
    system for accurate planar area computation, then extracts the area in square metres.
    Currently produces a single feature per geometry: the area. Useful for encoding zone or
    building sizes as node features in the graph.
    """
    def __call__(self, geometry_series: pd.Series | pl.Series) -> torch.Tensor:
        """
        Description: Encodes a series of geometry objects by computing their area in square metres.
        Uses an automatic UTM projection for accurate (non-degree-based) area calculation.

        Input:
          - geometry_series (pd.Series): A pandas Series with dtype 'geometry' (from GeoPandas).
            Must NOT be a polars Series — geometry types are only supported in pandas/geopandas.

        Output:
          - (torch.Tensor): A 2D tensor of shape=(len(geometry_series), 1) where each value is
            the area of the corresponding geometry in square metres.
        """
        if not isinstance(geometry_series, pd.Series) or geometry_series.dtype != "geometry":
            raise ValueError(
                f"Geometry must be of type gpd.Series with dtype=geometry,"
                f"found {type(geometry_series)} with dtype={geometry_series.dtype}"
            )

        geometry_series = gpd.GeoSeries(geometry_series)  # Wrap in GeoSeries for spatial operations
        planar_crs = geometry_series.estimate_utm_crs()  # Find the best UTM CRS for this region
        # Reproject to UTM and compute area in square metres
        area = torch.tensor(geometry_series.to_crs(planar_crs).area.to_numpy())

        return torch.stack([area], dim=1)  # Return as (N, 1) tensor


def network_to_pyg(network: Network, embedding_dim=EMBEDDING_DIM) -> HeteroData:
    """
    Description: Converts a multi-layer transport Network into a PyG HeteroData graph.
    Each layer type in the network becomes a distinct node type in the heterogeneous graph.
    Node features are computed using encoders for location type, layer name, and geometry.
    Edge features are computed for travel time, transport mode, and departure times.
    Inter-layer links (connections between layers) are also processed and added.
    The result is a static base graph that can be annotated per-user via ActivityGraphBuilder.

    Input:
      - network (Network): A multi-layer transport Network object containing PLANAR and
        PUBLIC_TRANSPORT layers, with nodes (locations) and edges (travel connections).
      - embedding_dim (int): Dimensionality for all learned category embeddings. Defaults to EMBEDDING_DIM (3).

    Output:
      - (HeteroData): A PyG heterogeneous graph with:
          - Node features (x) per layer type.
          - Edge indices and attributes per edge type and relation.
          - graph_metadata: dict with 'loc_id_mapping' and 'full_loc_id_mapping' for
            mapping location IDs to (layer_type, node_index) pairs.
    """
    data = HeteroData()

    layer_name_encoder = EnumEncoder(network.layers.keys(), embedding_dim)
    loc_type_encoder = EnumEncoder(network.location_types, embedding_dim)
    geometry_encoder = GeometryEncoder()
    node_encoders = {"type": loc_type_encoder, LAYER_NAME_COL: layer_name_encoder, "geometry": geometry_encoder}

    route_mode_encoder = EnumEncoder(Mode, embedding_dim)
    time_of_day_encoder = TimeOfDayEncoder()
    edge_encoders = {
        "route_mode": route_mode_encoder,
        "first_departure_time": time_of_day_encoder,
        "last_departure_time": time_of_day_encoder,
    }

    node_mappings: dict[LayerType, NodeMapping] = {}
    for layer_type in set(network.layers.values()):
        layer_mapping = _process_layers_by_type(data, network, layer_type, node_encoders, edge_encoders)
        node_mappings[layer_type] = layer_mapping

    link_types = set((network[l1].type, network[l2].type) for l1, l2 in network.links)
    for lower_type, upper_type in link_types:
        relations = _process_layer_links(network, lower_type, upper_type, node_mappings)

        for (lower, relation_name, upper), (edge_index, edge_attr) in relations.items():
            data[lower, relation_name, upper].edge_index = edge_index
            data[lower, relation_name, upper].edge_attr = edge_attr

    data.graph_metadata = _create_graph_metadata(node_mappings)

    return data


def _create_graph_metadata(
    node_mappings: dict[LayerType, NodeMapping],
) -> dict[str, LocIDMapping]:
    """
    Description: Builds the graph metadata dictionary from per-layer node mappings. Constructs
    two mappings:
      - 'full_loc_id_mapping': maps ALL location IDs (including PT (stop, route) tuples) to
        (layer_type, node_index) pairs.
      - 'loc_id_mapping': the subset where keys are plain strings (usable as simple loc_id lookups).
    For PT parent stop nodes, adds a direct string loc_id → node mapping so that stops can be
    looked up without needing the route dimension.

    Input:
      - node_mappings (dict[LayerType, NodeMapping]): A dictionary from layer type to its
        node mapping (loc_id or (loc_id, route_id) → integer index).

    Output:
      - (dict[str, LocIDMapping]): A dict with two keys:
          'full_loc_id_mapping': complete mapping including tuple keys.
          'loc_id_mapping': string-key-only subset for simple lookup.
    """
    full_loc_id_mapping = {}

    for layer_type, mapping in node_mappings.items():
        for loc_id, node_index in mapping.items():
            full_loc_id_mapping[loc_id] = (layer_type, node_index)

            # If PT `parent` node, add a direct loc_id -> node mapping
            if isinstance(loc_id, tuple) and loc_id[1] == PARENT_STOP_ROUTE_ID:
                full_loc_id_mapping[loc_id[0]] = (layer_type, node_index)

    loc_id_mapping = {k: v for k, v in full_loc_id_mapping.items() if isinstance(k, str)}

    return {"full_loc_id_mapping": full_loc_id_mapping, "loc_id_mapping": loc_id_mapping}


def add_labels_to_pyg(
    data: HeteroData, user_journeys_df: pl.DataFrame, separate_na_source_sink: bool
) -> Generator[HeteroData]:
    """
    Description: Generator function that adds per-user node features and labels to a base
    HeteroData graph, yielding one annotated graph per user. This is a functional alternative
    to ActivityGraphBuilder — useful when you want a one-shot annotation without building
    a full builder object. For large-scale usage, prefer ActivityGraphBuilder.

    Input:
      - data (HeteroData): The base graph created by `network_to_pyg`. Will be cloned per user.
      - user_journeys_df (pl.DataFrame): Journey records conforming to USER_JOURNEY_SCHEMA.
      - separate_na_source_sink (bool): If True, treats 'NA' origins as 'NA_SOURCE' and 'NA'
        destinations as 'NA_SINK'.

    Output:
      - (Generator[HeteroData]): Yields one annotated HeteroData graph per user, with home
        indicator features and visited location labels added.
    """
    check_schema(user_journeys_df, USER_JOURNEY_SCHEMA)

    loc_id_mapping: LocIDMapping = data.graph_metadata["loc_id_mapping"]
    loc_id_to_layer_type = {k: t for k, (t, _) in loc_id_mapping.items()}
    loc_id_to_node_index = {k: i for k, (_, i) in loc_id_mapping.items()}

    user_attributes = _build_user_attributes(user_journeys_df, separate_na_source_sink)
    visited_locations = _build_visited_locations(user_journeys_df, separate_na_source_sink)
    visited_locations = visited_locations.filter(pl.col("user_id").is_in(user_attributes["user_id"].implode()))

    for (user_id,), user_visited_locations in visited_locations.group_by("user_id"):
        user_attrs = user_attributes.filter(user_id=user_id)

        data = data.clone()
        data.graph_metadata["user_id"] = user_id

        _add_location_indicator_features(data, user_attrs, "home_loc_id", loc_id_to_layer_type, loc_id_to_node_index)
        _add_visited_location_labels(data, user_visited_locations, loc_id_to_layer_type, loc_id_to_node_index)

        yield data


def _build_user_attributes(user_journeys_df: pl.DataFrame, separate_na_source_sink: bool) -> pl.DataFrame:
    """
    Description: Builds a DataFrame of user-level location attributes by extracting each user's
    home location, work locations, and study locations from their journey records. Joins these
    together so each user has one row with their home_loc_id, work_loc_ids (list), and
    study_loc_ids (list). Users without a valid home location are excluded (inner join).

    Input:
      - user_journeys_df (pl.DataFrame): Journey records conforming to USER_JOURNEY_SCHEMA.
      - separate_na_source_sink (bool): If True, 'NA' origins become 'NA_SOURCE' and 'NA'
        destinations become 'NA_SINK' before extraction.

    Output:
      - (pl.DataFrame): One row per user with columns:
          user_id, home_loc_id (str), work_loc_ids (list), study_loc_ids (list).
    """
    home_locations = _extract_locations(
        user_journeys_df, "od_lieu_domicile", separate_na_source_sink, is_unique=True, loc_col="home_loc_id"
    )
    work_locations = _extract_locations(
        user_journeys_df, "od_lieu_travail", separate_na_source_sink, is_unique=False, loc_col="work_loc_ids"
    )
    study_locations = _extract_locations(
        user_journeys_df, "od_lieu_etude", separate_na_source_sink, is_unique=False, loc_col="study_loc_ids"
    )

    return (
        user_journeys_df
        .select(pl.col("user_id").unique())
        .join(home_locations, on="user_id", how="inner")
        .join(work_locations, on="user_id", how="left")
        .join(study_locations, on="user_id", how="left")
    )


def _extract_locations(
    user_journeys_df: pl.DataFrame,
    purpose: str,
    separate_na_source_sink: bool,
    is_unique: bool = True,
    loc_col: str = "loc_id",
) -> pl.DataFrame:
    """
    Description: Extracts the location(s) associated with a specific activity purpose for each
    user, by looking at the departure and arrival locations where that purpose was recorded.
    Departure locations are taken from the first leg of a journey, arrival locations from the
    last leg. If is_unique=True, only users with exactly one distinct location are kept (e.g.
    for home, which should be unique). If is_unique=False, all locations are collected as a list.

    Input:
      - user_journeys_df (pl.DataFrame): Journey records conforming to USER_JOURNEY_SCHEMA.
      - purpose (str): The activity purpose code to filter on (e.g. 'od_lieu_domicile' for home).
      - separate_na_source_sink (bool): If True, replaces 'NA' in origins with 'NA_SOURCE' and
        in destinations with 'NA_SINK'.
      - is_unique (bool): If True, only keep users with exactly one distinct location for this
        purpose. Defaults to True.
      - loc_col (str): The output column name for the location ID(s). Defaults to 'loc_id'.

    Output:
      - (pl.DataFrame): A DataFrame with columns ['user_id', loc_col] where loc_col is either
        a single string (if is_unique=True) or a list of strings (if is_unique=False).
    """
    departure_locs = user_journeys_df.filter(
        pl.col("dep_purpose") == purpose, pl.col("leg_id") == pl.col("leg_id").min().over("journey_id")
    ).select("user_id", pl.col("dep_loc_id").alias(loc_col))
    arrival_locs = user_journeys_df.filter(
        pl.col("arr_purpose") == purpose, pl.col("leg_id") == pl.col("leg_id").max().over("journey_id")
    ).select("user_id", pl.col("arr_loc_id").alias(loc_col))

    if separate_na_source_sink:
        departure_locs = departure_locs.with_columns(pl.col(loc_col).replace({"NA": "NA_SOURCE"}))
        arrival_locs = arrival_locs.with_columns(pl.col(loc_col).replace({"NA": "NA_SINK"}))

    locations = pl.concat([departure_locs, arrival_locs]).group_by("user_id").agg(pl.col(loc_col).unique())

    if is_unique:
        locations = locations.filter(pl.col(loc_col).list.len() <= 1).with_columns(pl.col(loc_col).list.item())

    return locations


def _build_visited_locations(user_journeys_df: pl.DataFrame, separate_na_source_sink: bool) -> pl.DataFrame:
    """
    Description: Builds a DataFrame of all locations visited by each user across all their
    journeys. Combines origin locations (from the first leg of each journey) with final
    destination locations. Deduplicates and sorts the result by user, journey, and sequence.
    Each row represents one location visit event with flags indicating whether it is an
    origin, destination, and/or endpoint of a journey.

    Input:
      - user_journeys_df (pl.DataFrame): Journey records conforming to USER_JOURNEY_SCHEMA.
      - separate_na_source_sink (bool): If True, 'NA' origins become 'NA_SOURCE' and 'NA'
        destinations become 'NA_SINK'.

    Output:
      - (pl.DataFrame): A DataFrame with columns:
          user_id, journey_id, seq_num, loc_id, is_origin, is_destination, is_endpoint.
    """
    def replace_na_with(col: str, value: str):
        """
        Description: Helper closure that returns a Polars expression replacing 'NA' in the
        given column with the given value, but ONLY if separate_na_source_sink is True.
        If False, the column is returned unchanged.

        Input:
          - col (str): The column name to apply the replacement to.
          - value (str): The replacement value for 'NA' entries (e.g. 'NA_SOURCE' or 'NA_SINK').

        Output:
          - (pl.Expr): A Polars expression for the (possibly modified) column.
        """
        return pl.col(col).replace({"NA": value}) if separate_na_source_sink else pl.col(col)

    origin_and_intermediate = user_journeys_df.select(
        "user_id",
        "journey_id",
        "leg_id",
        loc_id=replace_na_with("dep_loc_id", "NA_SOURCE"),
        is_origin=pl.col("leg_id") == 0,
        is_destination=False,
        is_endpoint=pl.col("leg_id") == 0,
    )

    destinations = (
        user_journeys_df
        .group_by("user_id", "journey_id")
        .agg(pl.all().last())
        .select(
            "user_id",
            "journey_id",
            "leg_id",
            loc_id=replace_na_with("arr_loc_id", "NA_SINK"),
            is_origin=False,
            is_destination=True,
            is_endpoint=True,
        )
    )

    visited_locations = (
        pl
        .concat([origin_and_intermediate, destinations])
        .unique(maintain_order=True)
        .sort("user_id", "journey_id", "leg_id", "is_destination", ~pl.col("is_origin"))
        .with_columns(pl.col("leg_id").rank(method="ordinal").over("user_id", "journey_id") - 1)
        .rename({"leg_id": "seq_num"})
    )

    return visited_locations


def _add_location_indicator_features(
    data: HeteroData,
    user_attributes: pl.DataFrame,
    loc_id_column: str,
    loc_id_to_layer_type: dict[str, LayerType],
    loc_id_to_node_index: dict[str, int],
) -> None:
    """
    Description: Adds a binary indicator feature to each node in the graph indicating whether
    that node matches the user's specified location (e.g. home). For each layer type, creates a
    zeros vector and sets the relevant node(s) to 1.0, then concatenates this as a new column
    to the existing node feature matrix data[layer_type].x.

    Input:
      - data (HeteroData): The graph to annotate. Modified in place.
      - user_attributes (pl.DataFrame): Single-user attributes containing the location IDs to mark.
        Must have exactly one unique user_id.
      - loc_id_column (str): The column name in user_attributes to use as location IDs
        (e.g. 'home_loc_id').
      - loc_id_to_layer_type (dict[str, LayerType]): Maps location ID → layer type.
      - loc_id_to_node_index (dict[str, int]): Maps location ID → node index within its layer.

    Output:
      - None. Modifies data[layer_type].x in place for each layer type.
    """
    user_ids = user_attributes["user_id"].unique()
    assert len(user_ids) == 1, f"Only a single user ID is allowed per graph, found {user_ids}"

    visited_locations = user_attributes.select(pl.col(loc_id_column).explode())
    visited_indices = visited_locations.select(
        loc_id_column,
        layer_type=pl.col(loc_id_column).replace_strict(loc_id_to_layer_type),
        node_index=pl.col(loc_id_column).replace_strict(loc_id_to_node_index),
    )

    for layer_type in data.node_types:
        df = visited_indices.filter(layer_type=layer_type)

        x = torch.zeros(data[layer_type].num_nodes)
        x[df["node_index"].to_numpy()] = 1

        data[layer_type].x = torch.cat([data[layer_type].x, x.unsqueeze(1)], dim=1)


def _add_visited_location_labels(
    data: HeteroData,
    user_visited_locations: pl.DataFrame,
    loc_id_to_layer_type: dict[str, LayerType],
    loc_id_to_node_index: dict[str, int],
) -> None:
    """
    Description: Adds binary ground-truth labels (y) to each node in the graph indicating
    whether the user visited that node. For each layer type, creates a zeros vector, sets
    visited nodes to 1.0, and assigns it as data[layer_type].y. This is the prediction
    target used during model training.

    Input:
      - data (HeteroData): The graph to annotate. Modified in place.
      - user_visited_locations (pl.DataFrame): DataFrame of visited locations for one user.
        Must have exactly one unique user_id and must contain a 'loc_id' column.
      - loc_id_to_layer_type (dict[str, LayerType]): Maps location ID → layer type.
      - loc_id_to_node_index (dict[str, int]): Maps location ID → node index within its layer.

    Output:
      - None. Sets data[layer_type].y for each layer type in the graph.
    """
    user_ids = user_visited_locations["user_id"].unique()
    assert len(user_ids) == 1, f"Only a single user ID is allowed per graph, found {user_ids}"

    visited_indices = user_visited_locations.select(
        "user_id",
        "loc_id",
        layer_type=pl.col("loc_id").replace_strict(loc_id_to_layer_type),
        node_index=pl.col("loc_id").replace_strict(loc_id_to_node_index),
    )

    for layer_type in data.node_types:
        df = visited_indices.filter(layer_type=layer_type)

        y = torch.zeros(data[layer_type].num_nodes)
        y[df["node_index"].to_numpy()] = 1

        data[layer_type].y = y


# =====================================
# Layer processing
# =====================================


def _process_layers_by_type(
    data: HeteroData,
    network: Network,
    layer_type: LayerType,
    node_encoders: Mapping[str, Encoder] | None = None,
    edge_encoders: Mapping[str, Encoder] | None = None,
) -> Mapping[LayerType | tuple[str, str], int]:
    """
    Description: Processes all network layers of a given type (e.g. all PLANAR layers or all
    PUBLIC_TRANSPORT layers) and populates the HeteroData object with their node features and
    internal edge connectivity. Uses type-specific node and edge processors to handle the
    structural differences between layer types.

    Input:
      - data (HeteroData): The graph being built. Modified in place.
      - network (Network): The multi-layer transport network.
      - layer_type (LayerType): The type of layers to process (e.g. LayerType.PLANAR).
      - node_encoders (Mapping[str, Encoder] | None): Optional dict of column name → encoder
        for encoding node features.
      - edge_encoders (Mapping[str, Encoder] | None): Optional dict of column name → encoder
        for encoding edge features.

    Output:
      - (Mapping): A node mapping from location IDs to integer node indices for this layer type.
    """
    assert not data[layer_type]

    node_processor = _select_node_processor(layer_type)
    edge_processor = _select_edge_processor(layer_type)

    layer_names = [name for name, l_type in network.layers.items() if l_type == layer_type]
    node_mapping, node_x = node_processor(network, layer_names, node_encoders)
    internal_edges = edge_processor(network, layer_names, node_mapping, edge_encoders)

    data[layer_type].x = node_x
    data[layer_type].num_nodes = len(node_mapping)

    for relation_name, (edge_index, edge_attr) in internal_edges.items():
        data[layer_type, relation_name, layer_type].edge_index = edge_index
        data[layer_type, relation_name, layer_type].edge_attr = edge_attr

    return node_mapping


def _select_node_processor(layer_type: LayerType) -> NodeProcessor:
    """
    Description: Returns the appropriate node processing function for the given layer type.
    PUBLIC_TRANSPORT layers use a specialised PT node processor that handles (stop, route) pairs;
    all other layer types use the standard base node processor.

    Input:
      - layer_type (LayerType): The layer type to select a processor for.

    Output:
      - (NodeProcessor): A callable that processes nodes for the given layer type.
    """
    match layer_type:
        case LayerType.PUBLIC_TRANSPORT:
            return _process_pt_layer_nodes
        case _:
            return _process_base_layer_nodes


def _select_edge_processor(layer_type: LayerType) -> EdgeProcessor:
    """
    Description: Returns the appropriate edge processing function for the given layer type.
    PUBLIC_TRANSPORT layers use a specialised PT edge processor that handles PT trip edges and
    transfer edges separately; all other layer types use the standard base edge processor.

    Input:
      - layer_type (LayerType): The layer type to select a processor for.

    Output:
      - (EdgeProcessor): A callable that processes edges for the given layer type.
    """
    match layer_type:
        case LayerType.PUBLIC_TRANSPORT:
            return _process_pt_layer_edges
        case _:
            return _process_base_layer_edges


# =====================================
# Node processing
# =====================================


def _process_base_layer_nodes(
    network: Network, layer_names: Sequence[str], node_encoders: Mapping[str, Encoder] | None = None
) -> tuple[dict[str, int], torch.Tensor | None]:
    """
    Description: Processes nodes for non-PT (base) network layers (e.g. PLANAR layers). Combines
    all locations from the specified layer names into one GeoDataFrame, assigns integer indices,
    and encodes node features. Each node has a unique loc_id string as its key.

    Input:
      - network (Network): The transport network.
      - layer_names (Sequence[str]): Names of the network layers to combine.
      - node_encoders (Mapping[str, Encoder] | None): Optional encoders for node feature columns.

    Output:
      - (tuple[dict[str, int], torch.Tensor | None]): A tuple of:
          node_mapping : dict mapping loc_id → integer node index
          x            : node feature tensor of shape=(n_nodes, n_features), or None if no encoders.
    """
    layer_locations_gdf = _combine_layer_locations(network, layer_names)

    # Map `loc_id` to indices in PyG graph and create node features
    node_mapping = layer_locations_gdf.reset_index(drop=True).reset_index().set_index("loc_id")["index"].to_dict()
    x = _encode_node_features(layer_locations_gdf, node_encoders)

    # noinspection PyTypeChecker
    return node_mapping, x


def _process_pt_layer_nodes(
    network: Network, layer_names: Sequence[str], node_encoders: Mapping[str, Encoder] | None = None
) -> tuple[dict[tuple[str, str], int], torch.Tensor | None]:
    """
    Description: Processes nodes for PUBLIC_TRANSPORT network layers. In PT networks, a physical
    stop can appear multiple times — once per route that serves it. Each unique (stop_id, route_id)
    pair becomes a separate node. Stops not served by any route receive a 'parent' node with
    route_id = PARENT_STOP_ROUTE_ID. This structure allows the model to distinguish between
    different service options at the same physical stop.

    Input:
      - network (Network): The transport network.
      - layer_names (Sequence[str]): Names of the PT network layers to combine.
      - node_encoders (Mapping[str, Encoder] | None): Optional encoders for node feature columns.

    Output:
      - (tuple[dict[tuple[str, str], int], torch.Tensor | None]): A tuple of:
          node_mapping : dict mapping (loc_id, route_id) → integer node index
          x            : node feature tensor of shape=(n_nodes, n_features), or None if no encoders.
    """
    layer_locations_gdf = _combine_layer_locations(network, layer_names)
    pt_layers = [network.get_pt_layer(name) for name in layer_names]

    # Find all unique (loc_id, route_id) pairs and create a dataframe with their info.
    pt_origins = pl.concat(layer.pt_edge_df.select(loc_id="orig_loc_id", route_id="route_id") for layer in pt_layers)
    tr_origins = pl.concat(
        layer.transfer_edge_df.select(loc_id="orig_loc_id", route_id="orig_route_id") for layer in pt_layers
    )
    pt_destinations = pl.concat(
        layer.pt_edge_df.select(loc_id="dest_loc_id", route_id="route_id") for layer in pt_layers
    )
    tr_destinations = pl.concat(
        layer.transfer_edge_df.select(loc_id="dest_loc_id", route_id="dest_route_id") for layer in pt_layers
    )

    pt_locations = pl.concat([pt_origins, tr_origins, pt_destinations, tr_destinations]).unique()

    # Make sure stops without routes going through them still have a (loc_id, `parent`) node
    # noinspection PyTypeChecker
    pt_locations_gdf: gpd.GeoDataFrame = pt_locations.to_pandas().merge(layer_locations_gdf, on="loc_id", how="right")
    pt_locations_gdf["route_id"] = pt_locations_gdf["route_id"].fillna(PARENT_STOP_ROUTE_ID)

    # Create mapping from (loc_id, route_id) to node index and encode node features
    node_mapping = (
        pt_locations_gdf.reset_index(drop=True).reset_index().set_index(["loc_id", "route_id"])["index"].to_dict()
    )
    x = _encode_node_features(pt_locations_gdf, node_encoders)

    # noinspection PyTypeChecker
    return node_mapping, x


def _combine_layer_locations(network: Network, layer_names: Sequence[str]) -> gpd.GeoDataFrame:
    """
    Description: Combines the location GeoDataFrames from multiple network layers into a single
    GeoDataFrame. Adds a 'layer_name' column to identify which layer each row came from.
    Used by node processing functions to merge all locations of the same type before encoding.

    Input:
      - network (Network): The transport network.
      - layer_names (Sequence[str]): Names of the layers whose locations to combine.

    Output:
      - (gpd.GeoDataFrame): A combined GeoDataFrame with all locations from the named layers,
        with an additional 'layer_name' column indicating the source layer.
    """
    layer_locations = {name: network.get_layer_locations(name) for name in layer_names}

    # Combine layer locations together and add a column identifying each layer by name
    layer_names_series = pd.concat(
        (pd.Series(name).repeat(len(gdf)) for name, gdf in layer_locations.items()), ignore_index=True
    )

    # noinspection PyTypeChecker
    layer_locations_gdf: gpd.GeoDataFrame = pd.concat(layer_locations.values(), ignore_index=True)
    layer_locations_gdf[LAYER_NAME_COL] = layer_names_series

    return layer_locations_gdf


def _encode_node_features(
    layer_locations_gdf: gpd.GeoDataFrame, encoders: Mapping[str, Encoder] | None
) -> torch.Tensor | None:
    """
    Description: Encodes node features from a GeoDataFrame by applying a set of column-specific
    encoders. For each encoder, if its associated column exists in the GeoDataFrame, the encoder
    is called on that column and produces a tensor. All encoded tensors are concatenated along the
    feature dimension. Returns None if no encoders are provided.

    Input:
      - layer_locations_gdf (gpd.GeoDataFrame): GeoDataFrame of locations (one row per node).
      - encoders (Mapping[str, Encoder] | None): A dict mapping column names to encoder callables.
        Each encoder takes a pd.Series and returns a torch.Tensor. If None, returns None.

    Output:
      - (torch.Tensor | None): Concatenated node feature tensor of shape=(n_nodes, total_features),
        or None if encoders is None.
    """
    if encoders is None:
        return None

    xs = [encoder(layer_locations_gdf[col]) for col, encoder in encoders.items() if col in layer_locations_gdf]
    x = torch.cat(xs, dim=-1)

    return x


# =====================================
# Layer edge processing (internal)
# =====================================


def _process_base_layer_edges(
    network: Network,
    layer_names: Sequence[str],
    node_mapping: Mapping[str, int],
    edge_encoders: Mapping[str, Encoder] | None,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """
    Description: Processes internal edges for non-PT (base) network layers. Combines the edge
    lists from all specified layers, maps location IDs to node indices, encodes edge features
    (travel time + any additional encoders), and returns the result as a single 'travel' relation.

    Input:
      - network (Network): The transport network.
      - layer_names (Sequence[str]): Names of the layers whose edges to combine.
      - node_mapping (Mapping[str, int]): Maps loc_id → integer node index.
      - edge_encoders (Mapping[str, Encoder] | None): Optional encoders for additional edge features.

    Output:
      - (dict[str, tuple[torch.Tensor, torch.Tensor]]): A dict with one key 'travel' mapping to
        a tuple of (edge_index, edge_attr) where:
          edge_index : shape=(2, E) — source and destination node indices
          edge_attr  : shape=(E, F) — edge feature vectors (e.g. travel time)
    """
    layers = [network[name] for name in layer_names]

    orig_index_expr = pl.col("orig_loc_id").replace_strict(node_mapping)
    dest_index_expr = pl.col("dest_loc_id").replace_strict(node_mapping)

    edge_list = pl.concat(layer.edge_list for layer in layers)

    edge_indices = edge_list.select(orig_index_expr, dest_index_expr).to_torch().T
    edge_attr = _encode_edge_features(edge_list, ["travel_time_min"], edge_encoders)

    return {"travel": (edge_indices, edge_attr)}


def _process_pt_layer_edges(
    network: Network,
    layer_names: Sequence[str],
    node_mapping: Mapping[LocID, int],
    edge_encoders: Mapping[str, Encoder] | None,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """
    Description: Processes internal edges for PUBLIC_TRANSPORT network layers. Separates two
    types of edges:
      1. 'pt' edges: trips along a route between consecutive stops (stop A, route R) → (stop B, route R).
         Features: travel time, dwell time, daily trip count, average headway.
      2. 'transfer' edges: transfers between different routes at the same stop
         (stop X, route R1) → (stop X, route R2). Features: transfer travel time.

    Input:
      - network (Network): The transport network.
      - layer_names (Sequence[str]): Names of the PT layers to combine.
      - node_mapping (Mapping[LocID, int]): Maps (loc_id, route_id) → integer node index.
      - edge_encoders (Mapping[str, Encoder] | None): Optional encoders for additional edge features.

    Output:
      - (dict[str, tuple[torch.Tensor, torch.Tensor]]): A dict with two keys:
          'pt'       : (edge_index, edge_attr) for PT trip edges
          'transfer' : (edge_index, edge_attr) for transfer edges
    """
    layers = [network.get_pt_layer(name) for name in layer_names]

    pt_orig_idx_expr = pl.concat_list("orig_loc_id", "route_id").replace_strict(node_mapping)
    pt_dest_idx_expr = pl.concat_list("dest_loc_id", "route_id").replace_strict(node_mapping)
    tr_orig_idx_expr = pl.concat_list("orig_loc_id", "orig_route_id").replace_strict(node_mapping)
    tr_dest_idx_expr = pl.concat_list("dest_loc_id", "dest_route_id").replace_strict(node_mapping)

    pt_edge_list = pl.concat(layer.pt_edge_df for layer in layers)
    pt_edge_indices = pt_edge_list.select(pt_orig_idx_expr, pt_dest_idx_expr).to_torch().T
    pt_edge_attr = _encode_edge_features(
        pt_edge_list,
        ["travel_time_min", "avg_dwell_time_min", "daily_trip_count", "avg_headway_min"],
        edge_encoders,
    )

    tr_edges = pl.concat(layer.transfer_edge_df for layer in layers)
    tr_edge_indices = tr_edges.select(tr_orig_idx_expr, tr_dest_idx_expr).to_torch().T
    tr_edge_attr = _encode_edge_features(
        tr_edges,
        ["travel_time_min"],
        edge_encoders,
    )

    return {
        "pt": (pt_edge_indices, pt_edge_attr),
        "transfer": (tr_edge_indices, tr_edge_attr),
    }


def _encode_edge_features(
    edge_df: pl.DataFrame, numerical_features: list[str], edge_encoders: Mapping[str, Encoder] | None
) -> torch.Tensor:
    """
    Description: Encodes edge features by combining numerical columns and any optional
    encoder-based features into a single tensor. The numerical features are extracted directly
    as floats; encoder-based features (e.g. embeddings for mode, time of day) are appended.

    Input:
      - edge_df (pl.DataFrame): The DataFrame of edges, with one row per edge.
      - numerical_features (list[str]): Column names to include as raw numerical features
        (e.g. ['travel_time_min']).
      - edge_encoders (Mapping[str, Encoder] | None): Optional dict of column name → encoder
        callable for additional encoded features. Only applied if the column exists in edge_df.

    Output:
      - (torch.Tensor): Combined edge feature tensor of shape=(n_edges, total_features).
    """
    edge_attr = edge_df.select(*numerical_features).to_torch()

    if edge_encoders:
        encoded_attrs = [encoder(edge_df[col]) for col, encoder in edge_encoders.items() if col in edge_df]
        edge_attr = torch.cat([edge_attr, *encoded_attrs], dim=1)

    return edge_attr


# =====================================
# Inter-layer link processing
# =====================================


def _process_layer_links(
    network: Network, lower_type: LayerType, upper_type: LayerType, node_mapping: Mapping[str, NodeMapping]
) -> dict[tuple[str, str, str], tuple[torch.Tensor, torch.Tensor]]:
    """
    Description: Processes inter-layer links (connections between different layer types, e.g.
    between PLANAR zones and PUBLIC_TRANSPORT stops). Splits links into upward edges (lower → upper)
    and downward edges (upper → lower) and returns both directions with appropriate relation names.
    For layers of the same type, only upward (intra-type) links are returned.

    Input:
      - network (Network): The transport network containing link definitions.
      - lower_type (LayerType): The "lower" layer type in the hierarchy (e.g. PLANAR).
      - upper_type (LayerType): The "upper" layer type in the hierarchy (e.g. PUBLIC_TRANSPORT).
      - node_mapping (Mapping[str, NodeMapping]): Maps each LayerType to its node mapping.

    Output:
      - (dict[tuple[str, str, str], tuple[torch.Tensor, torch.Tensor]]): A dict mapping
        (source_type, relation_name, dest_type) → (edge_index, edge_attr) for each direction
        of the inter-layer link.
    """
    matching_links = [
        (ll, ul) for ll, ul in network.links if network[ll].type == lower_type and network[ul].type == upper_type
    ]

    link_edge_df = pl.concat(network.get_links(ll, ul) for ll, ul in matching_links)
    lower_mapping = _node_mapping_to_dict(node_mapping[lower_type])
    upper_mapping = _node_mapping_to_dict(node_mapping[upper_type])

    upwards_edges = link_edge_df.filter(pl.col("orig_loc_id").is_in(lower_mapping))
    downwards_edges = link_edge_df.filter(~pl.col("orig_loc_id").is_in(lower_mapping))

    upwards_edge_attr = upwards_edges.select("travel_time_min").to_torch()
    upwards_edge_indices = upwards_edges.select(
        pl.col("orig_loc_id").replace_strict(lower_mapping), pl.col("dest_loc_id").replace_strict(upper_mapping)
    ).to_torch()

    if lower_type == upper_type:
        assert downwards_edges.is_empty()
        return {(lower_type, "links", upper_type): (upwards_edge_indices.T, upwards_edge_attr)}

    downwards_edge_attr = downwards_edges.select("travel_time_min").to_torch()
    downwards_edge_indices = downwards_edges.select(
        pl.col("orig_loc_id").replace_strict(upper_mapping), pl.col("dest_loc_id").replace_strict(lower_mapping)
    ).to_torch()

    rel_up, rel_down = ("contains", "is_contained_by") if lower_type == LayerType.PLANAR else ("links_to", "links_from")

    return {
        (lower_type, rel_up, upper_type): (upwards_edge_indices.T, upwards_edge_attr),
        (upper_type, rel_down, lower_type): (downwards_edge_indices.T, downwards_edge_attr),
    }


def _node_mapping_to_dict(node_mapping: NodeMapping, parent_key: str = PARENT_STOP_ROUTE_ID) -> dict[str, int]:
    """
    Description: Converts a NodeMapping (which may have either plain string keys or (str, str)
    tuple keys for PT stops) into a plain string→int dictionary. For PT (tuple-key) mappings,
    only the 'parent' entries (those with route_id = parent_key) are included, using the stop ID
    (first element of the tuple) as the key. This normalises all mappings for consistent use in
    inter-layer link processing.

    Input:
      - node_mapping (NodeMapping): A mapping from location IDs (possibly tuples) to node indices.
      - parent_key (str): The route_id value that identifies parent stop nodes in PT mappings.
        Defaults to PARENT_STOP_ROUTE_ID.

    Output:
      - (dict[str, int]): A plain string-keyed dictionary mapping loc_id → node index.
    """
    if all(isinstance(key, str) for key in node_mapping.keys()):
        return dict(node_mapping.items())

    return {key[0]: item for key, item in node_mapping.items() if key[1] == parent_key}
