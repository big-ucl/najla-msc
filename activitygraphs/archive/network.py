"""
Module: activitygraphs/archive/network.py

Description:
    Defines the multi-layer heterogeneous transport Network data structure used to
    represent the Geneva (and similar) transport network graphs.

    Key classes:
      - LayerType: StrEnum distinguishing PUBLIC_TRANSPORT, PLANAR, POINT, and NA layers.
      - Layer: Base class for a named network layer with a set of location IDs and an
              edge list (origin, destination, travel_time_min).
      - PTLayer: Specialisation of Layer for public-transport data that stores separate
              PT route edges and inter-stop transfer edges.
      - Network: The full multi-layer graph.  Manages layers, inter-layer link edges,
              the locations GeoDataFrame, and user journey records.  Provides a fluent
              builder API (add_pt_layer, add_planar_layer, connect_layers, etc.) and
              save/load methods for disk persistence.

    Module-level functions:
      - build_planar_edges: compute rook-adjacency walking edges between polygon zones.
      - build_layer_link_edges: spatial-join-based inter-layer link edge construction.
      - build_na_link_edges: NA source/sink link edges to every other layer.
      - load_layer: factory that deserialises the correct Layer subclass from disk.
"""

import pickle
import shutil
from enum import StrEnum
from pathlib import Path
from typing import Iterable, Self, Literal

import geopandas as gpd
import pandas as pd

import polars as pl
from geopandas.sindex import SpatialIndex
from shapely import Polygon

import utils
from base import (
    EDGE_LIST_SCHEMA,
    PTNodeType,
    PT_EDGE_LIST_SCHEMA,
    TRANSFER_EDGE_LIST_SCHEMA,
    LOCATIONS_SCHEMA,
    USER_JOURNEY_SCHEMA,
    CRS,
    LINK_EDGE_LIST_SCHEMA,
    WALK_EDGE_LIST_SCHEMA,
)
from config import GenevaDataConfig
from network import NetworkData, PTLayerBuilder, TravelTimeFactory, NA_LON, NA_LAT, NA_SOURCE, NA_SINK, NA
from routing import TravelTimeCalculator
from utils import check_schema, extract_unique_loc_ids, check_geometry_shapes, convert_locations_to_point_geometry


class LayerType(StrEnum):
    """
    Description:
        String enumeration of the four layer types recognised in the network.

        Values (also used as PyG node-type strings):
          - PUBLIC_TRANSPORT: PT stop nodes; edges are route services and transfer walks.
          - POINT: point locations (currently unused).
          - PLANAR: polygon zones (subsectors, municipalities) connected by adjacency walks.
          - NA: the not-assigned source/sink placeholder node(s).
    """
    PUBLIC_TRANSPORT = "public_transport"
    POINT = "point"
    PLANAR = "planar"
    NA = "na"

    def __repr__(self):
        """
        Description:
            Returns the member's name string when the LayerType value is printed or
            inspected (e.g. in repr() calls or error messages).

        Output:
          - (str): the enum member's name, e.g. "PUBLIC_TRANSPORT".
        """
        return self.name


class Layer:
    """
    Description:
        Represents a single named layer within the multi-layer transport network.
        A layer is a set of location IDs of the same type (e.g. all PT stops, or all
        subsectors) together with an optional edge list describing how those locations
        are directly connected within the layer.

    Attributes:
      - name (str): human-readable layer name, e.g. "subsector" or "public_transport".
      - type (LayerType): the category of this layer.
      - loc_ids (list[str]): ordered list of all location IDs belonging to this layer.
      - edge_list (pl.DataFrame): edge DataFrame with at minimum
            (orig_loc_id, dest_loc_id, travel_time_min) columns.
    """
    def __init__(self, name: str, layer_type: LayerType, loc_ids: Iterable[str], edge_list: pl.DataFrame | None):
        """
        Description:
            Creates a Layer and validates that all edge endpoints are within the layer.

        Input:
          - name (str): the layer's name.
          - layer_type (LayerType): the layer category.
          - loc_ids (Iterable[str]): all location IDs belonging to this layer.
          - edge_list (pl.DataFrame | None): edges within this layer.  If None, an empty
                DataFrame with the correct schema is used.

        Output:
          - (Layer): the new Layer instance.

        Raises:
          - ValueError: if any edge connects nodes outside the loc_ids set.
        """
        # Default to an empty edge DataFrame if no edges are provided
        edge_list = edge_list if edge_list is not None else pl.DataFrame(schema=EDGE_LIST_SCHEMA)
        check_schema(edge_list, EDGE_LIST_SCHEMA, ignore_extra_cols=True)

        self.name = name           # human-readable identifier for this layer
        self.type = layer_type     # LayerType enum value (used as PyG node type)
        self.loc_ids = list(loc_ids)  # ordered list of location IDs in this layer

        # Validate: no edge should connect nodes outside this layer's loc_ids
        mismatches = edge_list.filter(
            ~pl.col("orig_loc_id").is_in(self.loc_ids) | ~pl.col("dest_loc_id").is_in(self.loc_ids)
        )

        if len(mismatches) > 0:
            mismatches_fmt = edge_list.select(pl.format("({}, {})", "orig_loc_id", "dest_loc_id"))
            mismatches_str = ",".join(mismatches_fmt.to_series())
            raise ValueError(f"Found edges that connect outside of layer:\n*******{mismatches_str}\n*********")

        self.edge_list = edge_list  # validated intra-layer edge DataFrame

    def __repr__(self):
        """
        Description:
            Returns a concise string representation of the Layer showing its name
            and type, useful for debugging and interactive inspection.

        Output:
          - (str): e.g. "Layer(subsector, type=PLANAR)".
        """
        return f"Layer({self.name}, type={self.type.name})"

    @classmethod
    def load(cls, layer_dir: Path) -> Self:
        """
        Description:
            Deserialises a Layer from its on-disk directory (saved by Layer.save).

        Input:
          - layer_dir (Path): the directory containing attrs.pickle and edge_df.parquet.

        Output:
          - (Layer): the reconstructed Layer instance.
        """
        with open(layer_dir / "attrs.pickle", "rb") as f:
            attrs = pickle.load(f)  # contains name, type, loc_ids

        edge_list = pl.read_parquet(layer_dir / "edge_df.parquet")

        return cls(attrs["name"], attrs["type"], attrs["loc_ids"], edge_list)

    def save(self, layers_dir: Path):
        """
        Description:
            Serialises the Layer to a sub-directory of layers_dir named after the
            layer type and name (e.g. "PLANAR-subsector/").

        Input:
          - layers_dir (Path): the parent directory under which the layer sub-directory
                will be created.

        Output:
          - None (writes files to disk).
        """
        layer_dir = self._dir(layers_dir)
        layer_dir.mkdir(parents=True, exist_ok=True)

        # attrs dict stores the non-DataFrame fields needed to reconstruct the Layer
        attrs = {"name": self.name, "type": self.type, "loc_ids": self.loc_ids}

        with open(layer_dir / "attrs.pickle", "wb") as f:
            # noinspection PyTypeChecker
            pickle.dump(attrs, f)

        self.edge_list.write_parquet(layer_dir / "edge_df.parquet")

    def _dir(self, layers_dir: Path) -> Path:
        """Returns the canonical sub-directory path for this layer under layers_dir."""
        return layers_dir / f"{self.type.name}-{self.name}"


class PTLayer(Layer):
    """
    Description:
        Specialised Layer for the public-transport subgraph.  Stores the GTFS-derived
        route edges and transfer (walking) edges separately so that downstream code can
        access them with their full PT-specific columns (route_id, headway, etc.) while
        the base Layer still provides a unified edge_list for generic graph algorithms.

    Additional attributes (beyond Layer):
      - node_type (PTNodeType): whether nodes are one-per-stop or one-per-(stop, route).
      - pt_edge_df (pl.DataFrame): PT route edges with full schema PT_EDGE_LIST_SCHEMA.
      - transfer_edge_df (pl.DataFrame): inter/intra-stop transfer edges.
    """
    def __init__(
        self,
        name: str,
        loc_ids: Iterable[str],
        node_type: PTNodeType,
        pt_edge_df: pl.DataFrame,
        transfer_edge_df: pl.DataFrame,
    ):
        """
        Description:
            Creates a PTLayer by merging the route and transfer edges into a combined
            edge_list (with base columns only), then storing the full tables separately.

        Input:
          - name (str): layer name (typically "public_transport").
          - loc_ids (Iterable[str]): all PT node location IDs in this layer.
          - node_type (PTNodeType): the node representation strategy.
          - pt_edge_df (pl.DataFrame): route edges conforming to PT_EDGE_LIST_SCHEMA.
          - transfer_edge_df (pl.DataFrame): transfer edges conforming to
                TRANSFER_EDGE_LIST_SCHEMA.
        """
        # Build the unified edge list from the base columns of both edge tables
        edge_cols = EDGE_LIST_SCHEMA.keys()
        edge_df = pl.concat([pt_edge_df.select(*edge_cols), transfer_edge_df.select(*edge_cols)])

        super().__init__(name, LayerType.PUBLIC_TRANSPORT, loc_ids, edge_df)

        self.node_type = node_type  # ONE_PER_STOP or ONE_PER_ROUTE
        # Validate and store the full-schema PT edge tables
        self.pt_edge_df = check_schema(pt_edge_df, PT_EDGE_LIST_SCHEMA)
        self.transfer_edge_df = check_schema(transfer_edge_df, TRANSFER_EDGE_LIST_SCHEMA)

    @classmethod
    def load(cls, layer_dir) -> Self:
        """
        Description:
            Deserialises a PTLayer from its on-disk directory (saved by PTLayer.save).
            Loads the shared attrs.pickle (name, loc_ids), the PT-specific
            pt_attrs.pickle (node_type), and both Parquet edge files.

        Input:
          - layer_dir (Path): directory containing attrs.pickle, pt_attrs.pickle,
                pt_edge_df.parquet, and transfer_edge_df.parquet.

        Output:
          - (PTLayer): the reconstructed PTLayer instance.
        """
        with open(layer_dir / "attrs.pickle", "rb") as f:
            attrs = pickle.load(f)  # Shared layer attributes: name, type, loc_ids

        with open(layer_dir / "pt_attrs.pickle", "rb") as f:
            pt_attrs = pickle.load(f)  # PT-specific attribute: node_type

        # Load the full-schema PT edge tables from Parquet
        pt_edge_df = pl.read_parquet(layer_dir / "pt_edge_df.parquet", schema=PT_EDGE_LIST_SCHEMA)
        transfer_edge_df = pl.read_parquet(layer_dir / "transfer_edge_df.parquet", schema=TRANSFER_EDGE_LIST_SCHEMA)

        return cls(attrs["name"], attrs["loc_ids"], pt_attrs["node_type"], pt_edge_df, transfer_edge_df)

    def save(self, layers_dir: Path):
        """
        Description:
            Serialises the PTLayer to disk by calling the parent Layer.save() for
            the shared files, then additionally writing pt_attrs.pickle,
            pt_edge_df.parquet, and transfer_edge_df.parquet.

        Input:
          - layers_dir (Path): parent directory under which the layer sub-directory
                will be created (same convention as Layer.save).

        Output:
          - None (writes files to disk).
        """
        # Save base Layer files (attrs.pickle, edge_df.parquet) via parent class
        super().save(layers_dir)
        layer_dir = self._dir(layers_dir)  # Resolve the sub-directory path

        # Save PT-specific attribute (node_type) separately so PTLayer can be distinguished
        with open(layer_dir / "pt_attrs.pickle", "wb") as f:
            pickle.dump({"node_type": self.node_type}, f)

        # Write both full-schema PT edge tables
        self.pt_edge_df.write_parquet(layer_dir / "pt_edge_df.parquet")
        self.transfer_edge_df.write_parquet(layer_dir / "transfer_edge_df.parquet")


def load_layer(layer_dir: Path) -> Layer:
    """
    Description:
        Factory function that reads the layer type from the attrs.pickle file and
        delegates deserialization to the appropriate class (PTLayer or Layer).

    Input:
      - layer_dir (Path): the directory containing the serialised layer files.

    Output:
      - (Layer | PTLayer): the deserialized layer object of the correct subclass.
    """
    with open(layer_dir / "attrs.pickle", "rb") as f:
        attrs = pickle.load(f)

    # Read the stored LayerType to decide which class to use for deserialization
    layer_type: LayerType = attrs["type"]

    match layer_type:
        case LayerType.PUBLIC_TRANSPORT:
            # PTLayer has additional .pt_edge_df and .transfer_edge_df files
            return PTLayer.load(layer_dir)
        case LayerType.NA | LayerType.PLANAR | LayerType.POINT:
            # Base Layer only has edge_df.parquet
            return Layer.load(layer_dir)


class Network:
    """
    Description:
        The complete multi-layer transport network for a study area.  Stores:
          - A locations GeoDataFrame with all locations across all layers.
          - A user journeys DataFrame (raw survey trip records).
          - A dict of Layer objects indexed by layer name.
          - A dict of inter-layer link edge DataFrames indexed by (lower, upper) name pairs.

        The class exposes a fluent builder API so that layers and links can be added
        one by one using method chaining (Network.empty_network(...).add_pt_layer(...)...).
        The final Network is then saved to disk and reloaded without reconstruction.

    Key methods:
      - add_pt_layer / add_planar_layer / add_na_layer: add a layer of the given type.
      - connect_layers: create spatial inter-layer link edges between two existing layers.
      - connect_na_layer: create source/sink links from the NA node to all specified layers.
      - save / load: persist and restore the full Network to/from disk.
    """
    def __init__(
        self,
        locations_gdf: gpd.GeoDataFrame,
        user_journeys_df: pl.DataFrame,
        layers: dict[str, Layer],
        links: dict[tuple[str, str], pl.DataFrame],
    ):
        """
        Description:
            Constructs a Network from pre-built components.  Validates that both the
            locations GeoDataFrame and user journeys DataFrame conform to their expected
            schemas.

        Input:
          - locations_gdf (gpd.GeoDataFrame): all locations across all layers with
                columns (loc_id, loc_name, type, lon, lat, geometry).
          - user_journeys_df (pl.DataFrame): cleaned survey trip records.
          - layers (dict[str, Layer]): mapping from layer name to Layer object.
          - links (dict[tuple[str, str], pl.DataFrame]): mapping from (lower, upper)
                layer name pair to inter-layer link edge DataFrame.
        """
        self._locations_gdf = check_schema(locations_gdf, LOCATIONS_SCHEMA)
        self._user_journeys = check_schema(user_journeys_df, USER_JOURNEY_SCHEMA)
        self._layers = layers.copy()  # copy to avoid external mutation
        self._links = links.copy()    # copy to avoid external mutation

    @classmethod
    def empty_network(cls, network_data: NetworkData) -> Self:
        """
        Description:
            Creates a new empty Network (no layers, no links) pre-populated with
            the locations GeoDataFrame and user journeys from a NetworkData object.
            Use this as the starting point of the fluent builder API before adding
            layers with add_pt_layer, add_planar_layer, etc.

        Input:
          - network_data (NetworkData): raw data container with locations_gdf and
                user_journeys_df already loaded (from a GenevaDataConfig pipeline).

        Output:
          - (Network): an empty Network with locations and journeys loaded, ready
                for layer construction.
        """
        return cls(network_data.locations_gdf, network_data.user_journeys_df, {}, {})

    @property
    def locations_gdf(self) -> gpd.GeoDataFrame:
        """
        Description:
            Returns a copy of the full locations GeoDataFrame (all layers combined).
            Returns a copy to prevent external code from accidentally mutating the
            Network's internal state.

        Output:
          - (gpd.GeoDataFrame): copy of all locations with columns from LOCATIONS_SCHEMA.
        """
        return self._locations_gdf.copy()

    @property
    def locations_df(self) -> pl.DataFrame:
        """
        Description:
            Returns the locations GeoDataFrame converted to a plain Polars DataFrame
            (dropping the geometry column). Useful for downstream Polars operations
            that do not need geospatial functionality.

        Output:
          - (pl.DataFrame): Polars DataFrame of all location attributes (no geometry column).
        """
        return utils.gdf_to_polars(self._locations_gdf)

    @property
    def location_types(self) -> list[str]:
        """
        Description:
            Returns the set of distinct location type strings present in the network
            (e.g. ["public_transport", "planar", "na"]). Useful for understanding
            which kinds of locations are in the dataset.

        Output:
          - (list[str]): unique values of the "type" column in the locations table.
        """
        return self._locations_gdf["type"].unique().tolist()

    @property
    def layers(self):
        """
        Description:
            Returns a summary dictionary mapping each layer name to its LayerType,
            without exposing the full Layer objects. Useful for inspection and repr.

        Output:
          - (dict[str, LayerType]): mapping from layer name string to LayerType enum value.
        """
        return {name: layer.type for name, layer in self._layers.items()}

    @property
    def links(self):
        """
        Description:
            Returns a list of (lower, upper) name pairs for all inter-layer link
            edge tables currently stored in the network.

        Output:
          - (list[tuple[str, str]]): list of (lower_layer_name, upper_layer_name) pairs.
        """
        return [(name1, name2) for (name1, name2), _ in self._links.items()]

    @property
    def has_separate_na_source_sink(self):
        """
        Description:
            Checks whether the NA layer uses two separate nodes (NA_SOURCE and NA_SINK)
            or a single shared NA node.  This matters when building NA link edges:
            NA_SOURCE is used for edges leaving the NA node, NA_SINK for edges arriving.

        Output:
          - (bool): True if there are two NA locations (separate source and sink),
                False if there is only one (shared in/out NA node).

        Raises:
          - ValueError: if there are neither 1 nor 2 NA locations (data integrity error).
        """
        # Count how many rows in the locations table have type == 'na'
        num_na_locations = len(self._locations_gdf.query("type == 'na'"))

        if num_na_locations not in [1, 2]:
            raise ValueError(f"Invalid number of NA locaitons {num_na_locations}")

        return num_na_locations == 2

    def __getitem__(self, name: str) -> Layer:
        """
        Description:
            Returns the Layer object for the given layer name, enabling dictionary-style
            access: network["subsector"] returns the subsector Layer.

        Input:
          - name (str): the layer name to look up.

        Output:
          - (Layer): the Layer (or PTLayer) object for that name.

        Raises:
          - ValueError: if no layer with that name exists in the network.
        """
        if name not in self._layers:
            raise ValueError(f"Cannot find layer `{name}`")

        return self._layers[name]

    def get_layer_locations(self, layer_name: str) -> gpd.GeoDataFrame:
        """
        Description:
            Returns the subset of the locations GeoDataFrame containing only locations
            that belong to the specified layer. Used when building inter-layer link edges.

        Input:
          - layer_name (str): name of the layer whose locations to retrieve.

        Output:
          - (gpd.GeoDataFrame): rows from locations_gdf whose loc_id is in the layer.
        """
        # Get the ordered list of location IDs for this layer
        loc_ids = self[layer_name].loc_ids
        return self._locations_gdf[self._locations_gdf["loc_id"].isin(loc_ids)]

    def get_pt_layer(self, name: str) -> PTLayer:
        """
        Description:
            Retrieves a layer by name and validates that it is a PTLayer (not a
            base Layer). Returns it typed as PTLayer so callers can access PT-specific
            attributes (pt_edge_df, transfer_edge_df, node_type).

        Input:
          - name (str): name of the layer to retrieve.

        Output:
          - (PTLayer): the PTLayer object for that name.

        Raises:
          - ValueError: if the layer exists but is not a PTLayer.
        """
        layer = self[name]

        if not isinstance(layer, PTLayer):
            raise ValueError(f"Layer `{name}` is not a PT layer")

        return layer

    def get_links(self, lower_layer: str, upper_layer: str) -> pl.DataFrame:
        """
        Description:
            Returns the inter-layer link edge DataFrame connecting the lower layer
            to the upper layer (as created by connect_layers or connect_na_layer).

        Input:
          - lower_layer (str): name of the lower-level layer (e.g. "subsector").
          - upper_layer (str): name of the upper-level layer (e.g. "public_transport").

        Output:
          - (pl.DataFrame): link edge DataFrame conforming to LINK_EDGE_LIST_SCHEMA.

        Raises:
          - KeyError: if no link between (lower_layer, upper_layer) exists.
        """
        return self._links[(lower_layer, upper_layer)]

    def add_layer(
        self,
        name: str,
        layer_type: LayerType,
        loc_ids: str | gpd.GeoDataFrame | Iterable[str],
        edge_list: pl.DataFrame | None,
    ) -> Self:
        """
        Description:
            Generic method to add any type of layer to the network. Validates that
            the layer name is not already taken and that all loc_ids are known
            locations in the network. Returns self for method chaining.

        Input:
          - name (str): unique name for the new layer.
          - layer_type (LayerType): the category of this layer (PLANAR, POINT, NA, etc.).
          - loc_ids (str | gpd.GeoDataFrame | Iterable[str]): locations belonging to
                this layer. A string is treated as a location type filter (selects all
                locations of that type). A GeoDataFrame must conform to LOCATIONS_SCHEMA.
          - edge_list (pl.DataFrame | None): intra-layer edges, or None for no edges.

        Output:
          - (Network): self, to allow fluent method chaining (e.g. .add_layer(...).add_pt_layer(...)).
        """
        loc_ids = self._check_layer_loc_ids(name, loc_ids)  # Validate and normalise loc_ids
        self._layers[name] = Layer(name, layer_type, loc_ids, edge_list)
        return self

    def add_pt_layer(
        self,
        name: str,
        loc_ids: str | gpd.GeoDataFrame | Iterable[str] | None = None,
        pt_layer_builder: PTLayerBuilder | None = None,
        pt_edge_df: pl.DataFrame | None = None,
        transfer_edge_df: pl.DataFrame | None = None,
        pt_node_type: PTNodeType = PTNodeType.ONE_PER_ROUTE,
    ) -> Self:
        """
        Description:
            Adds a public-transport layer to the network. PT edges (route and transfer)
            can be provided directly or generated via a PTLayerBuilder callable.

            Two ways to call this method:
              1. Provide a `pt_layer_builder`: edges are computed automatically from
                 the network's locations table.
              2. Provide `pt_edge_df` and `transfer_edge_df` directly (both must be
                 given together; providing only one raises an error).

        Input:
          - name (str): unique name for the new PT layer (e.g. "public_transport").
          - loc_ids (str | gpd.GeoDataFrame | Iterable[str] | None): PT stop/node
                location IDs. If None, all unique node IDs found in the edge tables
                are used automatically.
          - pt_layer_builder (PTLayerBuilder | None): callable that builds PT edges
                from the locations table. Required if pt_edge_df/transfer_edge_df are None.
          - pt_edge_df (pl.DataFrame | None): pre-built route edges (PT_EDGE_LIST_SCHEMA).
          - transfer_edge_df (pl.DataFrame | None): pre-built transfer edges
                (TRANSFER_EDGE_LIST_SCHEMA).
          - pt_node_type (PTNodeType): whether to use one node per stop or one per
                (stop, route) combination. Default: ONE_PER_ROUTE.

        Output:
          - (Network): self, for fluent method chaining.

        Raises:
          - ValueError: if neither a builder nor both edge DataFrames are provided,
                or if only one of pt_edge_df / transfer_edge_df is given.
        """
        if pt_edge_df is None and transfer_edge_df is None and pt_layer_builder is not None:
            # Build edges from the locations table using the provided builder callable
            pt_edge_df, transfer_edge_df = pt_layer_builder(self.locations_df, pt_node_type)
        elif pt_layer_builder is None:
            raise ValueError("No PTLayerBuilder provided, cannot build edges.")
        elif pt_edge_df is None or transfer_edge_df is None:
            raise ValueError("Arguments `pt_edge_df` and `transfer_edge_df` must be both None or both DataFrames")

        # For ONE_PER_STOP mode, verify that no node appears on more than one route
        if pt_node_type == PTNodeType.ONE_PER_STOP:
            _check_only_one_route_per_node(pt_edge_df, transfer_edge_df)

        # If no loc_ids provided, derive them from the edge tables
        if loc_ids is None:
            loc_ids = extract_unique_loc_ids(pt_edge_df, transfer_edge_df)

        loc_ids = self._check_layer_loc_ids(name, loc_ids)  # Validate against known locations

        self._layers[name] = PTLayer(name, loc_ids, pt_node_type, pt_edge_df, transfer_edge_df)
        return self

    def add_planar_layer(
        self, name: str, loc_ids: str | gpd.GeoDataFrame | Iterable[str], travel_time_f: TravelTimeFactory | None = None
    ) -> Self:
        """
        Description:
            Adds a planar (polygon zone) layer to the network. If a travel_time_f
            is provided, rook-adjacency walking edges between neighbouring zones are
            built automatically using build_planar_edges(). If no travel_time_f is
            given, the layer is added without any intra-layer edges.

        Input:
          - name (str): unique name for the new planar layer (e.g. "subsector").
          - loc_ids (str | gpd.GeoDataFrame | Iterable[str]): polygon zone location IDs
                belonging to this layer.
          - travel_time_f (TravelTimeFactory | None): source for computing walking times
                between adjacent zones. Pass None to skip edge creation.

        Output:
          - (Network): self, for fluent method chaining.
        """
        loc_ids = self._check_layer_loc_ids(name, loc_ids)  # Validate and normalise loc_ids
        # Subset the locations GeoDataFrame to only the zones in this layer
        planar_locations_gdf = self._locations_gdf[self.locations_gdf["loc_id"].isin(loc_ids)]
        # Build adjacency edges only if a travel-time source is provided
        planar_edges = build_planar_edges(planar_locations_gdf, travel_time_f) if travel_time_f is not None else None

        return self.add_layer(name, LayerType.PLANAR, loc_ids, planar_edges)

    def add_na_layer(self, separate_in_out_nodes: bool = False, na_coords: tuple[float, float] | None = None) -> Self:
        """
        Description:
            Adds the special "not-assigned" (NA) placeholder layer to the network.
            The NA layer acts as a virtual source/sink for trips whose origin or
            destination is outside the study area.

            When separate_in_out_nodes=True, two NA nodes are created (NA_SOURCE
            for trip origins and NA_SINK for trip destinations). Otherwise a single
            shared NA node is used.

            The NA location(s) are inserted at the top of the locations GeoDataFrame
            (replacing any existing 'na' entries) so they are always present in the
            graph node set.

        Input:
          - separate_in_out_nodes (bool): if True, create distinct source and sink
                NA nodes; if False (default), use a single shared NA node.
          - na_coords (tuple[float, float] | None): (longitude, latitude) for the
                NA node placeholder position on the map. Defaults to (NA_LON, NA_LAT).

        Output:
          - (Network): self, for fluent method chaining.

        Raises:
          - ValueError: if an NA layer already exists (call connect_na_layer instead).
        """
        if "na" in self._layers:
            raise ValueError("NA Layer is already in the network. Connect it using `connect_na_layer`")

        # Default NA geographic coordinates (defined in network.py constants)
        na_coords = (NA_LON, NA_LAT) if na_coords is None else na_coords

        # Create either 1 or 2 NA location rows depending on separate_in_out_nodes
        n_duplicates = 2 if separate_in_out_nodes else 1
        # NA_SOURCE and NA_SINK are the two separate IDs; NA is the single shared ID
        loc_ids = [NA_SOURCE, NA_SINK] if separate_in_out_nodes else NA
        # NA layer has no intra-layer edges (it's a virtual source/sink)
        empty_edge_list = pl.DataFrame(schema=EDGE_LIST_SCHEMA)

        # Build a GeoDataFrame row for each NA node at the placeholder coordinates
        na_locations = gpd.GeoDataFrame(
            {
                "loc_id": loc_ids,
                "loc_name": loc_ids,
                "type": ["na"] * n_duplicates,
                "lon": [na_coords[0]] * n_duplicates,
                "lat": [na_coords[1]] * n_duplicates,
            },
            crs=CRS,
            geometry=gpd.points_from_xy([na_coords[0]] * n_duplicates, [na_coords[1]] * n_duplicates, crs=CRS),
        )

        # Prepend the NA rows to the locations table, removing any pre-existing 'na' entries
        self._locations_gdf = pd.concat([
            na_locations,
            self._locations_gdf.query("type != 'na'"),
        ]).reset_index(drop=True)

        return self.add_layer("na", LayerType.NA, na_locations, empty_edge_list)

    def _check_layer_loc_ids(self, name: str, loc_ids: str | gpd.GeoDataFrame | Iterable[str]) -> list[str]:
        """
        Description:
            Internal helper used by all add_*_layer methods to validate and normalise
            the loc_ids argument before constructing a Layer.

            Handles three input forms:
              - str: treated as a location type string — selects all locations in
                    the network with that type (e.g. "public_transport").
              - gpd.GeoDataFrame: must conform to LOCATIONS_SCHEMA; the loc_id
                    column is extracted.
              - Iterable[str]: converted to a plain list.

            In all cases, the resulting list is checked against the network's known
            locations to catch typos or missing data early.

        Input:
          - name (str): name of the layer being added (used to check for duplicates).
          - loc_ids (str | gpd.GeoDataFrame | Iterable[str]): the location IDs
                in one of the three accepted forms.

        Output:
          - (list[str]): a validated, plain Python list of location ID strings.

        Raises:
          - ValueError: if the layer name already exists, or if any loc_id is not
                in the network's locations GeoDataFrame.
        """
        # Prevent adding a layer with a name that is already registered
        if name in self._layers:
            raise ValueError(f"Layer `{name}` is already in the network.")

        # Normalise the three input forms into a plain list of strings
        if isinstance(loc_ids, str):
            # String input: select all locations whose 'type' column matches
            loc_ids = self._locations_gdf[self._locations_gdf["type"] == loc_ids]["loc_id"].tolist()
        elif isinstance(loc_ids, gpd.GeoDataFrame):
            # GeoDataFrame input: validate schema, then extract the loc_id column
            check_schema(loc_ids, LOCATIONS_SCHEMA)
            loc_ids = loc_ids["loc_id"].tolist()
        else:
            # Any other iterable: just convert to a list
            loc_ids = list(loc_ids)

        # Check that every loc_id is actually in the network's known locations
        unknown_locations = set(loc_ids).difference(self._locations_gdf["loc_id"])
        if unknown_locations:
            raise ValueError(f"Locations in `loc_ids` are not in network locations:\n\t{', '.join(unknown_locations)}")

        return loc_ids

    def connect_layers(
        self,
        lower: str,
        upper: str,
        travel_time_f: TravelTimeFactory,
        mode: Literal["strict", "centroid_strict", "nearest", "centroid_nearest"] = "strict",
        direction: Literal["both", "ascending", "descending"] = "both",
    ) -> Self:
        """
        Description:
            Creates inter-layer link edges between two existing layers using spatial
            operations, then stores them in the network's links dict.

            The `mode` parameter controls the spatial matching strategy:
              - "strict": upper-layer points must intersect lower-layer polygons.
              - "centroid_strict": same, but using the upper layer's polygon centroids.
              - "nearest": each upper-layer node is linked to its nearest lower-layer node.
              - "centroid_nearest": nearest match using upper centroids.

            The `direction` parameter controls which link edges are created:
              - "both": edges in both directions (lower->upper and upper->lower).
              - "ascending": edges from upper to lower only.
              - "descending": edges from lower to upper only.

        Input:
          - lower (str): name of the lower-granularity layer (e.g. "subsector").
          - upper (str): name of the upper-granularity layer (e.g. "public_transport").
          - travel_time_f (TravelTimeFactory): source for computing link travel times.
          - mode (str): spatial matching mode (default "strict").
          - direction (str): which edge directions to create (default "both").

        Output:
          - (Network): self, for fluent method chaining.
        """
        # Retrieve the GeoDataFrames for both layers (only their locations)
        lower_locations_gdf = self.get_layer_locations(lower)
        upper_locations_gdf = self.get_layer_locations(upper)

        # Build the inter-layer link edges via spatial join
        link_edges = build_layer_link_edges(lower_locations_gdf, upper_locations_gdf, travel_time_f, mode, direction)
        # Register the link under the (lower, upper) key
        self._links[(lower, upper)] = link_edges

        return self

    def connect_na_layer(self, others: str | list[str], travel_time_f: TravelTimeFactory):
        """
        Description:
            Creates inter-layer link edges between the NA (not-assigned) layer and
            one or more other layers. This enables trips originating or ending outside
            the study area to be represented as starting/ending at the NA node(s).

            Two edge sets are built per target layer:
              - NA_SOURCE -> every location in the target layer (trip starts at NA).
              - Every location in the target layer -> NA_SINK (trip ends at NA).

            If the network uses separate source/sink NA nodes (has_separate_na_source_sink
            is True), the two directions use different node IDs (NA_SOURCE / NA_SINK).
            Otherwise both directions use the single NA node.

        Input:
          - others (str | list[str]): name(s) of the layer(s) to connect the NA layer to.
          - travel_time_f (TravelTimeFactory): source for computing NA link travel times.

        Output:
          - (Network): self, for fluent method chaining.
        """
        # Normalise a single layer name string to a one-element list
        others = [others] if isinstance(others, str) else others

        for layer in others:
            # Select the correct NA node ID for each direction
            na_source = NA_SOURCE if self.has_separate_na_source_sink else NA
            na_sink = NA_SINK if self.has_separate_na_source_sink else NA

            layer_locations = self.get_layer_locations(layer)  # Locations in the target layer
            # Build NA_SOURCE -> all layer locations edges (trip starts at NA)
            na_source_link_edges = build_na_link_edges(na_source, layer_locations, travel_time_f, "na_to_loc")
            # Build all layer locations -> NA_SINK edges (trip ends at NA)
            na_sink_link_edges = build_na_link_edges(na_sink, layer_locations, travel_time_f, "loc_to_na")

            # Store both source and sink edges together under the ("na", layer) key
            self._links[("na", layer)] = pl.concat([na_source_link_edges, na_sink_link_edges])

        return self

    def __repr__(self):
        """
        Description:
            Returns a multi-line string representation of the Network showing all
            layer names/types and all registered inter-layer links, useful for
            interactive inspection in a notebook or REPL.

        Output:
          - (str): formatted string listing layers and links.
        """
        # Format each layer as "name: TYPE" indented under "layers=("
        layers = "\n".join(f"\t\t{n}: {t}" for n, t in self.layers.items())
        # Format each link as "lower |--> upper" to show direction
        links = "\n".join(f"\t\t{low} |--> {up}" for low, up in self.links)

        return f"""Network(\n\tlayers=(\n{layers}\n\t), links=(\n{links}\n\t)\n)"""

    @classmethod
    def _dir(cls, cfg: GenevaDataConfig, project_root: Path | None = None, name: str | None = None) -> Path:
        """
        Description:
            Computes the canonical on-disk directory path for a Network given a
            GenevaDataConfig. Uses the config's processed data directory and appends
            an optional name suffix (e.g. "Network-v2").

        Input:
          - cfg (GenevaDataConfig): the data configuration object that specifies paths.
          - project_root (Path | None): the root of the project tree. Defaults to ".".
          - name (str | None): optional suffix appended to the directory name
                (e.g. "v2" produces "Network-v2"). None means no suffix.

        Output:
          - (Path): the full directory path where the Network is or should be saved.
        """
        # Default to current directory if no project root is supplied
        project_root = project_root if project_root is not None else Path(".")
        # An optional name suffix distinguishes multiple saved Network variants
        suffix = "" if name is None else f"-{name}"
        data_dir = project_root / cfg.paths.processed / f"{cls.__name__}{suffix}"

        return data_dir

    @classmethod
    def exists_on_disk(cls, cfg: GenevaDataConfig, project_root: Path | None = None, name: str | None = None):
        """
        Description:
            Checks whether a previously saved Network exists at the expected disk
            location without loading it. Useful for conditional rebuild logic.

        Input:
          - cfg (GenevaDataConfig): data configuration specifying processed data path.
          - project_root (Path | None): project root directory. Defaults to ".".
          - name (str | None): optional name suffix used when the Network was saved.

        Output:
          - (bool): True if the Network directory exists, False otherwise.
        """
        return cls._dir(cfg, project_root, name).exists()

    @classmethod
    def load(cls, cfg: GenevaDataConfig, project_root: Path | None = None, name: str | None = None) -> Self:
        """
        Description:
            Deserialises a complete Network from disk, reconstructing all layers,
            inter-layer links, the locations GeoDataFrame, and user journeys.

        Input:
          - cfg (GenevaDataConfig): data configuration specifying processed data path.
          - project_root (Path | None): project root directory. Defaults to ".".
          - name (str | None): optional name suffix to distinguish multiple saved variants.

        Output:
          - (Network): the fully reconstructed Network object.

        Raises:
          - ValueError: if the expected directory does not exist on disk.
        """
        data_dir = cls._dir(cfg, project_root, name)  # Resolve the save directory

        if data_dir.exists():
            # Reconstruct all Layer objects from the layers/ sub-directory
            layers = cls._load_layers(data_dir / "layers")
            # Reconstruct all link edge DataFrames from the links/ sub-directory
            links = cls._load_links(data_dir / "links")

            user_journeys_df = pl.read_parquet(data_dir / "user_journeys_df.parquet")
            locations_gdf = gpd.read_parquet(data_dir / "locations_gdf.parquet")

            return cls(locations_gdf, user_journeys_df, layers, links)
        else:
            raise ValueError(f"Data directory does not exist: {data_dir}")

    @classmethod
    def _load_layers(cls, layers_dir: Path) -> dict[str, Layer]:
        """
        Description:
            Iterates over all sub-directories in layers_dir and deserialises each
            into the appropriate Layer or PTLayer object using the load_layer factory.

        Input:
          - layers_dir (Path): directory containing one sub-directory per layer.

        Output:
          - (dict[str, Layer]): mapping from layer name to its Layer (or PTLayer) object.
        """
        layers = {}  # Accumulator: layer name -> Layer object
        for layer_dir in layers_dir.iterdir():
            layer = load_layer(layer_dir)  # Factory function handles PTLayer vs Layer
            layers[layer.name] = layer

        return layers

    @classmethod
    def _load_links(cls, links_dir: Path) -> dict[tuple[str, str], pl.DataFrame]:
        """
        Description:
            Iterates over all sub-directories in links_dir and deserialises each
            inter-layer link edge DataFrame along with its (lower, upper) name pair.

        Input:
          - links_dir (Path): directory containing one sub-directory per link pair.

        Output:
          - (dict[tuple[str, str], pl.DataFrame]): mapping from (lower, upper) layer
                name pair to its link edge DataFrame.
        """
        links = {}  # Accumulator: (lower, upper) -> link edge DataFrame
        for link_dir in links_dir.iterdir():
            with open(link_dir / "attrs.pickle", "rb") as f:
                attrs = pickle.load(f)  # Contains "lower" and "upper" layer names

            link_edge_df = pl.read_parquet(link_dir / "link_edges.parquet", schema=LINK_EDGE_LIST_SCHEMA)
            links[(attrs["lower"], attrs["upper"])] = link_edge_df

        return links

    def save(
        self,
        cfg: GenevaDataConfig,
        project_root: Path | None = None,
        name: str | None = None,
        can_overwrite: bool = False,
    ):
        """
        Description:
            Serialises the entire Network to disk, creating the directory structure:
              <data_dir>/
                layers/<TYPE-name>/       -- one sub-dir per layer (via Layer.save)
                links/<lower-upper>/      -- one sub-dir per link pair
                user_journeys_df.parquet  -- user journey records
                locations_gdf.parquet     -- all locations GeoDataFrame

            If can_overwrite is True and layers/ or links/ already exist, they are
            deleted before re-saving (to avoid stale sub-directories from previous runs).

        Input:
          - cfg (GenevaDataConfig): data configuration specifying processed data path.
          - project_root (Path | None): project root directory. Defaults to ".".
          - name (str | None): optional suffix to distinguish multiple saved variants.
          - can_overwrite (bool): if True, existing layers/ and links/ directories
                will be deleted and replaced. False (default) raises an error if they exist.

        Output:
          - None (writes files to disk).
        """
        # Resolve the canonical save directory and its sub-directories
        data_dir = self._dir(cfg, project_root, name)
        layers_dir = data_dir / "layers"
        links_dir = data_dir / "links"

        # Create the root directory (allow existing dirs if can_overwrite)
        data_dir.mkdir(parents=True, exist_ok=can_overwrite)

        # If overwriting is allowed, wipe and recreate layers/ and links/ directories
        if can_overwrite and layers_dir.exists():
            shutil.rmtree(layers_dir)  # Delete the old layers directory tree

        if can_overwrite and links_dir.exists():
            shutil.rmtree(links_dir)   # Delete the old links directory tree

        layers_dir.mkdir(parents=True, exist_ok=can_overwrite)
        links_dir.mkdir(parents=True, exist_ok=can_overwrite)

        # Serialise each Layer (or PTLayer) to its own sub-directory
        for _, layer in self._layers.items():
            layer.save(layers_dir)

        # Serialise each link edge DataFrame to its own sub-directory
        for (lower, upper), link_edge_df in self._links.items():
            link_dir = links_dir / f"{lower}-{upper}"  # Sub-dir named after the pair
            link_dir.mkdir(parents=True, exist_ok=True)

            # Save the (lower, upper) name pair so it can be reconstructed on load
            attrs = {"lower": lower, "upper": upper}
            with open(link_dir / "attrs.pickle", "wb") as f:
                pickle.dump(attrs, f)

            link_edge_df.write_parquet(link_dir / "link_edges.parquet")

        # Save the flat tables at the root level
        self._user_journeys.write_parquet(data_dir / "user_journeys_df.parquet")
        self._locations_gdf.to_parquet(data_dir / "locations_gdf.parquet")


def build_planar_edges(planar_locations_gdf: gpd.GeoDataFrame, travel_time_f: TravelTimeFactory) -> pl.DataFrame:
    """
    Description:
        Builds walking edges between rook-adjacent polygon zones (e.g. subsectors).
        Two zones are adjacent if their geometries share a boundary or overlap.

    Input:
      - planar_locations_gdf (gpd.GeoDataFrame): a GeoDataFrame of polygon locations
            conforming to LOCATIONS_SCHEMA.  Each row is one zone.
      - travel_time_f (TravelTimeFactory): a travel-time source used to compute the
            walking time between adjacent zone centroids.

    Output:
      - (pl.DataFrame): a WALK_EDGE_LIST_SCHEMA DataFrame with columns
            (orig_loc_id, dest_loc_id, travel_time_min) for each adjacent zone pair.
    """
    check_schema(planar_locations_gdf, LOCATIONS_SCHEMA)
    check_geometry_shapes(planar_locations_gdf.geometry, "Polygon", "MultiPolygon")

    # Find all (loc_id, neighbour) pairs using rook adjacency (spatial index intersection)
    neighbours = _compute_neighbour_loc_ids(planar_locations_gdf)
    planar_edge_df = neighbours.select(orig_loc_id="loc_id", dest_loc_id="neighbour")
    # Add the travel_time_min column using the provided travel-time source
    planar_edge_df = _add_travel_time_column(planar_edge_df, travel_time_f)

    return check_schema(planar_edge_df, WALK_EDGE_LIST_SCHEMA)


def build_layer_link_edges(
    lower_locations_gdf: gpd.GeoDataFrame,
    upper_locations_gdf: gpd.GeoDataFrame,
    travel_time_f: TravelTimeFactory,
    mode: Literal["strict", "centroid_strict", "nearest", "centroid_nearest"] = "strict",
    direction: Literal["both", "ascending", "descending"] = "both",
) -> pl.DataFrame:
    """
    Description:
        Builds inter-layer link edges between a lower-granularity layer (e.g. polygon
        zones / subsectors) and an upper-granularity layer (e.g. PT stops) using
        spatial join operations in a projected CRS.

        The CRS is estimated automatically from the upper layer's geometry using
        estimate_utm_crs() to ensure distance-based spatial operations are accurate.

        Matching modes:
          - "strict": upper geometry must intersect lower geometry (e.g. a PT stop
                that falls inside a subsector polygon).
          - "centroid_strict": as strict but converts upper polygon to its centroid first.
          - "nearest": each upper location is matched to its nearest lower location
                regardless of overlap.
          - "centroid_nearest": as nearest but using the upper polygon centroid.

        Direction options control which edges are kept in the output:
          - "both": edges in both directions (lower->upper AND upper->lower).
          - "descending": only lower->upper edges (from fine to coarse, i.e. a PT stop
                pointing to the subsector it is inside).
          - "ascending": only upper->lower edges (reversed direction).

    Input:
      - lower_locations_gdf (gpd.GeoDataFrame): locations for the lower-level layer
            (e.g. subsector polygons), conforming to LOCATIONS_SCHEMA.
      - upper_locations_gdf (gpd.GeoDataFrame): locations for the upper-level layer
            (e.g. PT stops), conforming to LOCATIONS_SCHEMA.
      - travel_time_f (TravelTimeFactory): source for computing link edge travel times.
      - mode (str): spatial matching strategy (default "strict").
      - direction (str): which direction edges to include (default "both").

    Output:
      - (pl.DataFrame): link edge DataFrame conforming to LINK_EDGE_LIST_SCHEMA with
            columns (orig_loc_id, dest_loc_id, travel_time_min).

    Raises:
      - ValueError: if mode or direction is not one of the accepted values.
    """
    check_schema(lower_locations_gdf, LOCATIONS_SCHEMA)
    check_schema(upper_locations_gdf, LOCATIONS_SCHEMA)

    # For centroid modes, convert upper polygon geometries to their centroids
    if mode == "centroid_strict" or mode == "centroid_nearest":
        upper_locations_gdf = convert_locations_to_point_geometry(upper_locations_gdf)

    # Project both layers to a local UTM CRS for accurate distance-based matching
    projected_crs = upper_locations_gdf.estimate_utm_crs()
    upper = upper_locations_gdf[["loc_id", "geometry"]].to_crs(projected_crs)
    lower = lower_locations_gdf[["loc_id", "geometry"]].to_crs(projected_crs)

    # Perform the spatial join using the selected mode
    if mode == "strict" or mode == "centroid_strict":
        # Intersects predicate: upper point/centroid must lie inside a lower polygon
        intersection = upper.sjoin(lower, predicate="intersects")
    elif mode == "nearest" or mode == "centroid_nearest":
        # Nearest: each upper location is matched to the spatially closest lower location
        intersection = upper.sjoin_nearest(lower)
    else:
        raise ValueError(f"Invalid mode {mode}, must be 'strict' or 'nearest'")

    # Extract (upper_loc_id, lower_loc_id) pairs from the spatial join result
    edges = pl.DataFrame(intersection[["loc_id_left", "loc_id_right"]]).select(
        orig_loc_id="loc_id_left", dest_loc_id="loc_id_right"
    )
    # Build the reversed edges for the ascending direction
    edged_reversed = edges.select(orig_loc_id="dest_loc_id", dest_loc_id="orig_loc_id")

    # Filter to the requested edge direction(s)
    if direction == "both":
        edges = pl.concat([edges, edged_reversed])  # Include both directions
    elif direction == "ascending":
        edges = edged_reversed  # Only upper -> lower edges
    elif direction != "descending":
        raise ValueError(f"Invalid direction {direction}, must be 'both', 'ascending' or 'descending'")

    # Append the travel_time_min column using the provided travel-time source
    edges = _add_travel_time_column(edges, travel_time_f)

    return check_schema(edges, LINK_EDGE_LIST_SCHEMA)


def build_na_link_edges(
    na_loc_id: str,
    locations_gdf: gpd.GeoDataFrame,
    travel_time_f: TravelTimeFactory,
    direction: Literal["na_to_loc", "loc_to_na"],
) -> pl.DataFrame:
    """
    Description:
        Builds link edges between a single NA (not-assigned) placeholder node and
        all locations in a target layer. Used to represent trips that originate or
        end outside the modelled study area.

        Two directions are supported:
          - "na_to_loc": NA -> every location (one row per location in locations_gdf,
                all with orig_loc_id == na_loc_id).
          - "loc_to_na": every location -> NA (one row per location, all with
                dest_loc_id == na_loc_id).

    Input:
      - na_loc_id (str): the NA node's location ID (e.g. NA_SOURCE or NA_SINK constant).
      - locations_gdf (gpd.GeoDataFrame): the target layer's locations
            (conforming to LOCATIONS_SCHEMA).
      - travel_time_f (TravelTimeFactory): source for computing NA link travel times.
      - direction (str): "na_to_loc" for outbound edges or "loc_to_na" for inbound edges.

    Output:
      - (pl.DataFrame): link edge DataFrame conforming to LINK_EDGE_LIST_SCHEMA.

    Raises:
      - ValueError: if direction is not "na_to_loc" or "loc_to_na".
    """
    if direction not in ["na_to_loc", "loc_to_na"]:
        raise ValueError(f"Invalid direction {direction}, must be 'na_to_loc' or 'loc_to_na'")

    # For na_to_loc: the NA node is the origin; for loc_to_na: all locations are origins
    origins = na_loc_id if direction == "na_to_loc" else locations_gdf["loc_id"]
    # For na_to_loc: all locations are destinations; for loc_to_na: the NA node is the destination
    destinations = locations_gdf["loc_id"] if direction == "na_to_loc" else na_loc_id

    # Build the edge DataFrame — one row per (origin, destination) pair
    edges = pl.DataFrame({
        "orig_loc_id": origins,
        "dest_loc_id": destinations,
    })

    # Add the travel_time_min column and validate schema before returning
    edges = _add_travel_time_column(edges, travel_time_f)
    return check_schema(edges, LINK_EDGE_LIST_SCHEMA)


def _check_only_one_route_per_node(pt_edge_df: pl.DataFrame, transfer_edge_df: pl.DataFrame):
    """
    Description:
        Validates that, when using PTNodeType.ONE_PER_STOP, each PT stop node ID
        appears in only one route. In ONE_PER_STOP mode every node represents a
        physical stop, not a (stop, route) pair, so a node appearing on two different
        routes would be ambiguous.

        Checks both the origin and destination columns of pt_edge_df, and both
        the orig_loc_id and dest_loc_id columns of transfer_edge_df.

    Input:
      - pt_edge_df (pl.DataFrame): PT route edge DataFrame (schema: PT_EDGE_LIST_SCHEMA).
      - transfer_edge_df (pl.DataFrame): transfer edge DataFrame
            (schema: TRANSFER_EDGE_LIST_SCHEMA).

    Output:
      - None (raises an error if the constraint is violated).

    Raises:
      - ValueError: if any node ID appears with more than one route ID.
    """
    def check_df(name: str, df: pl.DataFrame, loc_id_col: str, route_id_col: str):
        """
        Description:
            Inner helper that checks one (loc_id, route_id) column pair in a DataFrame
            for any node that appears with multiple route IDs.

        Input:
          - name (str): human-readable name of the DataFrame being checked (for error messages).
          - df (pl.DataFrame): the DataFrame to check.
          - loc_id_col (str): name of the node location ID column.
          - route_id_col (str): name of the route ID column.
        """
        # Count how many distinct routes each node appears on
        counts = (
            df
            .select(loc_id_col, route_id_col)
            .unique()  # Remove exact duplicate (loc_id, route_id) pairs
            .group_by(loc_id_col)
            .agg(count=pl.len(), route_id=pl.col(route_id_col))
        )
        # Flag any node with count > 1 (appears on multiple routes)
        mismatches = counts.filter(pl.col("count") != 1)

        if len(mismatches) > 0:
            raise ValueError(
                f"Some nodes ({loc_id_col}, {route_id_col}) in {name} have more than one route: \n{mismatches}"
            )

    # Check all four (loc_id, route_id) combinations across both edge DataFrames
    check_df("pt_edge_df", pt_edge_df, "orig_loc_id", "route_id")
    check_df("pt_edge_df", pt_edge_df, "dest_loc_id", "route_id")
    check_df("transfer_edge_df", transfer_edge_df, "orig_loc_id", "orig_route_id")
    check_df("transfer_edge_df", transfer_edge_df, "dest_loc_id", "dest_route_id")


def _add_travel_time_column(edge_df: pl.DataFrame, travel_time_f: TravelTimeFactory) -> pl.DataFrame:
    """
    Description:
        Appends a `travel_time_min` column to an edge DataFrame using any of the
        five supported TravelTimeFactory types. This function acts as a dispatcher
        that chooses the appropriate strategy based on the runtime type of travel_time_f.

        Supported factory types and their behaviour:
          - float: a constant value is assigned to all edges.
          - pl.Expr: a Polars expression is evaluated against the DataFrame columns.
          - pl.DataFrame: a pre-computed lookup table is joined on (orig_loc_id, dest_loc_id).
          - TravelTimeCalculator: delegates to the calculator's add_travel_times() method.
          - callable: a Python function f(orig_loc_id, dest_loc_id) -> float is applied
                row-by-row via map_elements.

    Input:
      - edge_df (pl.DataFrame): edge DataFrame with at least columns
            (orig_loc_id: String, dest_loc_id: String).
      - travel_time_f (TravelTimeFactory): the travel-time source in one of the five
            supported forms described above.

    Output:
      - (pl.DataFrame): the same edge DataFrame with an added `travel_time_min` column
            (Float64) representing travel time in minutes.

    Raises:
      - ValueError: if travel_time_f is none of the five supported types.
    """
    # Validate that the edge DataFrame has the required ID columns before proceeding
    check_schema(edge_df, pl.Schema({"orig_loc_id": pl.String, "dest_loc_id": pl.String}), ignore_extra_cols=True)

    def travel_time_f_helper(struct):
        """
        Description:
            Row-level helper used when travel_time_f is a plain callable.
            Unpacks the (orig_loc_id, dest_loc_id) struct passed by map_elements
            and calls the travel_time_f function.

        Input:
          - struct (dict): Polars struct with keys "orig_loc_id" and "dest_loc_id".

        Output:
          - (float): the travel time in minutes for this origin-destination pair.
        """
        return travel_time_f(struct["orig_loc_id"], struct["dest_loc_id"])

    if isinstance(travel_time_f, float):
        # Constant travel time: assign the same value to every edge
        return edge_df.with_columns(travel_time_min=travel_time_f)
    elif isinstance(travel_time_f, pl.Expr):
        # Polars expression: evaluated against the DataFrame (e.g. pl.lit(5.0))
        return edge_df.with_columns(travel_time_min=travel_time_f)
    elif isinstance(travel_time_f, pl.DataFrame):
        # Pre-computed lookup table: joined on (orig_loc_id, dest_loc_id)
        check_schema(travel_time_f, WALK_EDGE_LIST_SCHEMA)
        return edge_df.join(travel_time_f, on=["orig_loc_id", "dest_loc_id"], how="left")
    elif isinstance(travel_time_f, TravelTimeCalculator):
        # Routing-engine calculator: delegates to its add_travel_times method
        return travel_time_f.add_travel_times(edge_df)
    elif callable(travel_time_f):
        # Python callable f(orig, dest) -> float: applied row-by-row via map_elements
        return edge_df.with_columns(input=travel_time_f).with_columns(
            travel_time_min=pl.col("input").map_elements(travel_time_f_helper, return_dtype=pl.Float64)
        )

    raise ValueError(f"Invalid travel_time_f of type {type(travel_time_f)}")


def _find_neighbours(row_geometry: Polygon, sindex: SpatialIndex, geometries: gpd.GeoSeries):
    """
    Description:
        Finds all geometries in `geometries` that intersect a given polygon.
        Uses the spatial index (sindex) for a fast bounding-box pre-filter, then
        confirms with an exact intersects() check to avoid false positives.

        This implements the "rook adjacency" criterion: two zones are neighbours
        if their geometries share at least a boundary point or overlap.

    Input:
      - row_geometry (Polygon): the polygon whose neighbours to find.
      - sindex (SpatialIndex): the spatial index built from `geometries` (used
            for fast bounding-box candidate lookup).
      - geometries (gpd.GeoSeries): the full set of zone geometries indexed by loc_id.

    Output:
      - (pd.Index): index values (loc_ids) of all zones that intersect row_geometry.
    """
    # Step 1: fast bounding-box filter to find candidate neighbour indices
    candidates = geometries.iloc[sindex.intersection(row_geometry.bounds)]
    # Step 2: exact intersection check on the candidates only
    true_neighbours = candidates[candidates.intersects(row_geometry)].index

    return true_neighbours


def _compute_neighbour_loc_ids(planar_locations_gdf: gpd.GeoDataFrame, exclude_self: bool = True) -> pl.DataFrame:
    """
    Description:
        Computes all rook-adjacent zone pairs for a planar (polygon) layer. For each
        zone, finds all other zones whose geometry intersects it (using a spatial
        index for efficiency), then returns the full list of (loc_id, neighbour) pairs.

        This is the core step in build_planar_edges(): the resulting pairs define
        which zones should have a walking edge between them.

    Input:
      - planar_locations_gdf (gpd.GeoDataFrame): polygon zone GeoDataFrame with at
            least columns (loc_id, geometry). Geometry type must be Polygon or
            MultiPolygon.
      - exclude_self (bool): if True (default), removes (zone, zone) pairs where
            a zone is its own neighbour (always the case since a polygon intersects itself).

    Output:
      - (pl.DataFrame): two-column DataFrame with columns (loc_id, neighbour), one
            row per adjacent zone pair (including both (A, B) and (B, A) directions).
    """
    check_schema(planar_locations_gdf, {"loc_id": "object", "geometry": "geometry"}, ignore_extra_cols=True)
    check_geometry_shapes(planar_locations_gdf.geometry, "Polygon", "MultiPolygon")

    # Build the spatial index once for use across all zones
    sindex = planar_locations_gdf.sindex
    # Index the geometries by loc_id so that _find_neighbours returns loc_id values
    geometries = planar_locations_gdf.set_index("loc_id").geometry

    # For each zone, find its neighbours; explode the resulting list of lists into rows
    neighbours = (
        geometries
        .apply(_find_neighbours, sindex=sindex, geometries=geometries)
        .rename("neighbour")  # The neighbour column holds the found loc_ids
        .explode()            # Convert list-valued column to one row per neighbour
        .reset_index()        # Make loc_id a regular column (was the index)
    )

    # Remove self-adjacency: each polygon intersects itself, which we don't want as an edge
    if exclude_self:
        neighbours = neighbours[neighbours["loc_id"] != neighbours["neighbour"]]

    # Convert pandas DataFrame to Polars for consistency with the rest of the codebase
    return pl.DataFrame(neighbours)
