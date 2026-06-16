"""
OSRM-backed travel-time routers with optional in-memory caching.

This module provides classes to compute road-network travel times between pairs of locations
by calling the OSRM (Open Source Routing Machine) HTTP API. The main components are:

  - ``Router``: Abstract base class defining the interface for any travel-time router.
  - ``TravelTimeCalculator``: High-level helper that joins a Router with a location lookup table
    so you can pass edge DataFrames (with loc_ids) instead of raw coordinates.
  - ``OSRMRouter``: Concrete router that calls a locally running OSRM server.
  - ``CachedRouter``: Decorator that wraps any Router and caches (origin, destination) pairs
    in memory to avoid redundant HTTP calls.

Typical usage:
  1. Start a local OSRM server for the desired network profile (car, walk, or cycle).
  2. Create an ``OSRMRouter`` pointing at that server.
  3. Wrap it in ``TravelTimeCalculator`` to resolve location IDs to coordinates automatically.
  4. Optionally wrap the router in ``CachedRouter`` for repeated queries.
"""

import abc
from collections.abc import Iterable
from typing import Self

import geopandas as gpd
import numpy as np
import polars as pl
import requests
from pypolyline.cutil import encode_coordinates

from activitygraphs.base import Mode
from activitygraphs.utils import check_schema

# Maximum number of unique locations allowed in a single OSRM /table request.
# OSRM encodes all locations as a polyline; very large sets may hit URL length limits.
MAX_COORDS = 10_000

# Polyline encoding precision: 5 decimal places (standard for most routing APIs).
PRECISION = 5

# Mapping from internal Mode enum values to OSRM routing profile names.
# Modes that share a profile (e.g. TAXI and CAR both use "car") are grouped together.
MODE_TO_OSRM_PROFILE_MAP = {
    Mode.CAR: "car",
    Mode.MOTORCYCLE: "car",   # Motorcycles routed the same as cars in OSRM
    Mode.TAXI: "car",         # Taxis also use the car profile
    Mode.VEH_PASS: "car",     # Vehicle passengers also use the car profile
    Mode.WALK: "walk",
    Mode.CYCLE: "cycle",
}

# Schema for the edge coordinate DataFrame passed to ``Router.travel_times()``.
# Each row describes one directed edge from an origin location to a destination location,
# including their geographic coordinates (longitude and latitude).
EDGE_COORDS_SCHEMA = pl.Schema({
    "orig_loc_id": pl.String,   # Unique ID of the origin location
    "orig_lon": pl.Float64,     # Longitude of the origin location (degrees)
    "orig_lat": pl.Float64,     # Latitude of the origin location (degrees)
    "dest_loc_id": pl.String,   # Unique ID of the destination location
    "dest_lon": pl.Float64,     # Longitude of the destination location (degrees)
    "dest_lat": pl.Float64,     # Latitude of the destination location (degrees)
})


class Router(abc.ABC):
    """
    Description: Abstract base class for travel-time routers. Any concrete router
    (e.g. ``OSRMRouter``) must inherit from this class and implement ``travel_times``.
    The class also provides a convenience method ``with_cache`` to wrap itself in a
    ``CachedRouter`` without changing the calling code.
    """

    @abc.abstractmethod
    def travel_times(self, edge_coords_df: pl.DataFrame) -> pl.DataFrame:
        """
        Description: Compute the road-network travel time (in minutes) for every
        origin-destination pair described in ``edge_coords_df``.

        Input:
          - edge_coords_df (pl.DataFrame): DataFrame conforming to ``EDGE_COORDS_SCHEMA``.
            Each row is one directed (orig, dest) edge with longitude/latitude for both ends.

        Output:
          - (pl.DataFrame): Three-column DataFrame: orig_loc_id (String), dest_loc_id (String),
            travel_time_min (Float64). Row order may differ from input.
        """

    def with_cache(self) -> Self:
        """
        Description: Convenience wrapper that returns a ``CachedRouter`` around this router.
        The cached version stores computed (orig, dest) -> travel_time_min results in memory
        so repeated queries for the same pair are answered instantly without HTTP calls.

        Output:
          - (CachedRouter): A new CachedRouter wrapping this router instance.
        """
        return CachedRouter(self)


class TravelTimeCalculator:
    """
    Description: High-level helper that bridges a ``Router`` (which needs raw coordinates)
    with an edge DataFrame that only contains location IDs. It looks up the coordinates
    for each location automatically and then delegates to the underlying router.

    Use this class when you have a table of edges identified by loc_id strings and want
    to add a ``travel_time_min`` column without manually managing coordinate lookups.
    """

    def __init__(self, locations_gdf: gpd.GeoDataFrame, router: Router):
        """
        Description: Initialise the calculator by validating the locations table and
        pre-building origin/destination lookup DataFrames.

        Input:
          - locations_gdf (gpd.GeoDataFrame): Table of locations containing at minimum
            columns: loc_id (string), lon (float64), lat (float64). Extra columns are ignored.
          - router (Router): Any Router instance (e.g. OSRMRouter or CachedRouter) to
            delegate actual travel-time computation to.

        Output:
          - (None): Initialises the instance; no return value.
        """
        # Validate that the required columns exist and have correct types
        check_schema(locations_gdf, {"loc_id": "object", "lon": "float64", "lat": "float64"}, ignore_extra_cols=True)

        # Store a defensive copy of the GeoDataFrame to prevent external mutation
        self._locations_gdf = locations_gdf.copy()
        # Lightweight Polars DataFrame with just loc_id, lon, lat for fast joins
        self._locations_df = pl.DataFrame(locations_gdf[["loc_id", "lon", "lat"]])
        # Pre-built origin lookup: columns renamed to orig_loc_id, orig_lon, orig_lat
        self._origins = self._locations_df.select(pl.all().name.prefix("orig_"))
        # Pre-built destination lookup: columns renamed to dest_loc_id, dest_lon, dest_lat
        self._destinations = self._locations_df.select(pl.all().name.prefix("dest_"))

        # Store reference to the underlying router
        self._router = router

    def add_travel_times(self, edge_df: pl.DataFrame) -> pl.DataFrame:
        """
        Description: Look up coordinates for every (orig_loc_id, dest_loc_id) pair in
        ``edge_df``, call the router to compute travel times, and return ``edge_df``
        with an additional ``travel_time_min`` column.

        Input:
          - edge_df (pl.DataFrame): DataFrame with at least columns ``orig_loc_id`` (String)
            and ``dest_loc_id`` (String). All location IDs must exist in the locations table.
            Any additional columns are preserved unchanged.

        Output:
          - (pl.DataFrame): Same as ``edge_df`` but with a new ``travel_time_min`` (Float64)
            column appended. Uses a left join so unmatched rows get null instead of being dropped.
        """
        # Validate that the required ID columns are present
        check_schema(edge_df, pl.Schema({"orig_loc_id": pl.String, "dest_loc_id": pl.String}), ignore_extra_cols=True)

        # Collect all unique location IDs appearing in either origin or destination columns
        edge_locations = pl.concat([edge_df["orig_loc_id"], edge_df["dest_loc_id"]]).unique()
        if not edge_locations.is_in(self._locations_df["loc_id"]).all():
            raise ValueError("All locations in `edge_df` must exist in the router locations")

        # Build the EDGE_COORDS_SCHEMA DataFrame by joining coordinates onto both ends of each edge
        edge_coords_df = (
            edge_df
            .select("orig_loc_id", "dest_loc_id")
            .join(self._origins, on="orig_loc_id")        # Add orig_lon, orig_lat
            .join(self._destinations, on="dest_loc_id")   # Add dest_lon, dest_lat
        )

        # Delegate to the router (e.g. OSRM HTTP call)
        result = self._router.travel_times(edge_coords_df)
        # Validate that the router returned the expected schema
        check_schema(
            result, pl.Schema({"orig_loc_id": pl.String, "dest_loc_id": pl.String, "travel_time_min": pl.Float64})
        )

        # Left-join travel times back onto the original edge DataFrame (preserves all columns)
        return edge_df.join(result, on=["orig_loc_id", "dest_loc_id"], how="left")


class OSRMRouter(Router):
    """
    Description: Concrete Router that computes road-network travel times by calling
    a locally running OSRM (Open Source Routing Machine) HTTP server.

    OSRM requires a pre-processed road network file for the desired profile (car/walk/cycle).
    This router uses the ``/table`` endpoint to request a full NxN duration matrix for all
    unique locations appearing in the edge list, then extracts the required (orig, dest) pairs.
    """

    def __init__(self, url: str, mode: Mode, max_coordinates: int = MAX_COORDS, precision: int = PRECISION):
        """
        Description: Initialise the OSRM router with the server URL and travel mode.

        Input:
          - url (str): Base URL of the running OSRM server, e.g. ``"http://localhost:5000"``.
            A trailing slash is stripped automatically.
          - mode (Mode): Travel mode enum value. Must be one of the keys in
            ``MODE_TO_OSRM_PROFILE_MAP`` (CAR, MOTORCYCLE, TAXI, VEH_PASS, WALK, CYCLE).
          - max_coordinates (int): Maximum number of unique locations per single OSRM /table
            request. Defaults to 10,000. Larger values increase URL length and memory usage.
          - precision (int): Polyline encoding precision (number of decimal places).
            Defaults to 5, which is the standard for most routing APIs.

        Output:
          - (None): Initialises the instance; no return value.
        """
        # Guard against unsupported modes before attempting any HTTP calls
        if mode not in MODE_TO_OSRM_PROFILE_MAP.keys():
            raise ValueError(f"Invalid mode: {mode}, accepted: {MODE_TO_OSRM_PROFILE_MAP.keys()}")

        # Store the cleaned base URL (no trailing slash)
        self.url = url.rstrip("/")
        # Store the mode enum for reference
        self.mode = mode
        # Translate the mode to the OSRM profile string (e.g. "car", "walk", "cycle")
        self.profile = MODE_TO_OSRM_PROFILE_MAP[mode]
        # Max locations per request to avoid exceeding OSRM URL length limits
        self.max_coordinates = max_coordinates
        # Polyline precision; 5 = ~1.1 m accuracy at the equator
        self.precision = precision

    def travel_times(self, edge_coords_df: pl.DataFrame):
        """
        Description: Compute travel times for all (orig, dest) pairs in ``edge_coords_df``
        by first deduplicating locations, calling the OSRM /table endpoint once, and then
        indexing into the resulting duration matrix.

        Input:
          - edge_coords_df (pl.DataFrame): DataFrame conforming to ``EDGE_COORDS_SCHEMA``.
            Each row is one directed edge with coordinates for both endpoints.

        Output:
          - (pl.DataFrame): Columns: orig_loc_id (String), dest_loc_id (String),
            travel_time_min (Float64). One row per input edge.
        """
        # Validate the input schema before making any HTTP calls
        check_schema(edge_coords_df, EDGE_COORDS_SCHEMA)

        # Build a deduplicated, sorted list of unique locations with integer row indices.
        # The row index will be used to index into the OSRM duration matrix.
        locations = (
            pl
            .concat([
                edge_coords_df.select(loc_id="orig_loc_id", lon="orig_lon", lat="orig_lat"),
                edge_coords_df.select(loc_id="dest_loc_id", lon="dest_lon", lat="dest_lat"),
            ])
            .unique("loc_id")          # Deduplicate so each location appears once
            .sort("loc_id")            # Deterministic ordering for reproducibility
            .with_row_index()          # Add column "index" with values 0, 1, 2, ...
        )

        # Extract (lon, lat) pairs as an iterable to pass to the OSRM table call
        coordinates = locations.select("lon", "lat").iter_rows()
        # Call OSRM: returns an NxN numpy matrix of travel times in minutes
        durations = self.table(coordinates)

        # Map each edge's orig and dest loc_ids to their integer row indices in the matrix
        edge_indices = edge_coords_df.join(
            locations.select(orig_index="index", orig_loc_id="loc_id"), on="orig_loc_id"
        ).join(locations.select(dest_index="index", dest_loc_id="loc_id"), on="dest_loc_id")

        # Convert the index columns to a numpy array for fast 2D matrix indexing
        edge_indices_np = edge_indices.select("orig_index", "dest_index").to_numpy()
        # Index into the duration matrix: durations[orig_idx, dest_idx]
        travel_times = durations[edge_indices_np[:, 0], edge_indices_np[:, 1]]

        # Return a DataFrame with just the IDs and the computed travel times
        return edge_coords_df.select("orig_loc_id", "dest_loc_id").with_columns(travel_time_min=travel_times)

    def table(self, coordinates: Iterable[tuple[float, float]]) -> np.ndarray:
        """
        Description: Call the OSRM ``/table/v1/{profile}`` endpoint with a set of
        (lon, lat) coordinates and return the full NxN travel-time matrix in minutes.
        OSRM returns durations in seconds; this method divides by 60 before returning.

        Input:
          - coordinates (Iterable[tuple[float, float]]): Sequence of (longitude, latitude)
            pairs. Length must not exceed ``self.max_coordinates``.

        Output:
          - (np.ndarray): 2-D array of shape [n, n] where entry [i, j] is the travel time
            in minutes from coordinate i to coordinate j.
        """
        # Materialise the iterable so we can measure length and pass it to encode_coordinates
        coordinates = list(coordinates)

        # Enforce the coordinate limit to avoid HTTP request size errors
        if len(coordinates) > self.max_coordinates:
            raise ValueError(f"Too many coordinates: {len(coordinates)}, maximum: {self.max_coordinates}")

        # Encode coordinates as a Google polyline string (compact binary-like URL encoding)
        polyline = encode_coordinates(coordinates, self.precision)
        # URL-encode the polyline so it's safe to embed in the query URL
        polyline = requests.utils.quote(polyline, safe="")
        # Build the full OSRM table endpoint URL
        full_url = f"{self.url}/table/v1/{self.profile}/polyline({polyline})"
        response = requests.get(full_url)

        # Check for HTTP-level errors (4xx, 5xx status codes)
        if not response.ok:
            raise ConnectionError(
                f"OSRM routing service error: status code={response.status_code}, content={response.content}"
            )

        # Parse the JSON response
        json = response.json()
        # Check for OSRM application-level errors (e.g. "NoRoute")
        if json["code"] != "Ok":
            raise ConnectionError(f"OSRM routing error:\nstatus code={json['code']}\nmessage={json['message']}")

        # Convert the NxN duration matrix from seconds to minutes
        travel_times_min = np.array(json["durations"]) / 60
        return travel_times_min


class CachedRouter(Router):
    """
    Description: Decorator (wrapper) that adds an in-memory cache to any ``Router``.
    The cache stores computed travel times keyed by (orig_loc_id, dest_loc_id) tuples.
    On each call to ``travel_times``, edges already in the cache are served instantly;
    only cache-miss edges are forwarded to the underlying router.

    This is useful when the same set of locations is queried repeatedly (e.g. across
    multiple model training steps) to avoid redundant OSRM HTTP calls.
    """

    def __init__(self, router: Router):
        """
        Description: Initialise the CachedRouter wrapping a given router.

        Input:
          - router (Router): The underlying router to delegate cache-miss queries to.
            Can be any Router subclass, including another CachedRouter (nested caching).

        Output:
          - (None): Initialises the instance; no return value.
        """
        # Reference to the underlying router for cache-miss queries
        self.router = router
        # In-memory dict mapping (orig_loc_id, dest_loc_id) -> travel_time_min (float)
        # Starts empty and grows as new pairs are computed
        self.cache: dict[tuple[str, str], float] = {}

    def travel_times(self, edge_coords_df: pl.DataFrame) -> pl.DataFrame:
        """
        Description: Serve travel times from cache where possible, delegate the rest to
        the underlying router, and return a combined result.

        Input:
          - edge_coords_df (pl.DataFrame): DataFrame conforming to ``EDGE_COORDS_SCHEMA``.

        Output:
          - (pl.DataFrame): Columns: orig_loc_id (String), dest_loc_id (String),
            travel_time_min (Float64). Row order may differ from input (cache hits first).
        """
        # Validate input schema
        check_schema(edge_coords_df, EDGE_COORDS_SCHEMA)

        # Build a Polars Series of all cached [orig, dest] pairs for membership testing
        cache_keys = pl.Series(self.cache.keys(), dtype=pl.List(pl.String)).implode()

        # Expression: True if the [orig_loc_id, dest_loc_id] list is in the cache
        is_edge_in_cache = pl.concat_list("orig_loc_id", "dest_loc_id").is_in(cache_keys)
        # Split edges into those already cached and those that need computation
        cache_hits = edge_coords_df.filter(is_edge_in_cache)
        cache_misses = edge_coords_df.filter(~is_edge_in_cache)

        # Compute travel times for cache-miss edges via the underlying router
        cache_misses_tt = self.router.travel_times(cache_misses)
        # Retrieve cached travel times for cache-hit edges
        cache_hits_tt = cache_hits.select(
            "orig_loc_id",
            "dest_loc_id",
            travel_time_min=pl
            .concat_list("orig_loc_id", "dest_loc_id")  # Build [orig, dest] list per row
            .replace(self.cache)                         # Look up each list in the cache dict
            .list.item()                                 # Unwrap the single-element list
            .cast(pl.Float64),                           # Ensure Float64 type consistency
        )

        # Concatenate cache hits (fast) and cache misses (computed) into a single result
        return pl.concat([cache_hits_tt, cache_misses_tt])
