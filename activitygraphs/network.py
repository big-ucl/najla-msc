"""
NetworkData class that wraps raw DataFrames from different travel surveys into a unified interface.

This module defines the abstract base class ``NetworkData`` which all dataset-specific classes
(e.g. GenevaData, TorontoData) inherit from. It provides a consistent interface for accessing:
  - Filtered locations (spatial zones like subsectors or census tracts)
  - User journey records (who travelled from where to where and when)
  - Aggregated journey summaries (one row per trip, not per leg)
  - Location visit counts per user
  - Home, work, and education locations per user

The module also defines type aliases for callable factories used to build travel-time lookups
and PT (public transport) layer edges, as well as a helper to create the special "NA" location.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import cached_property
from typing import Self, TypeVar

import geopandas as gpd
import polars as pl

from activitygraphs import utils
from activitygraphs.base import (
    CRS,
    LOCATIONS_SCHEMA,
    USER_JOURNEY_SCHEMA,
    PTNodeType,
    Purpose,
)
from activitygraphs.routing import TravelTimeCalculator
from activitygraphs.utils import check_schema

# Longitude and latitude for the special "NA" location used as a sentinel
# for trips with unknown or missing origin/destination.
# This coordinate falls roughly in the Lake Geneva area.
NA_LON, NA_LAT = 6.1709475192397605, 46.24348817355701

# String constants for the NA (not available) location and its source/sink variants
# used to represent unknown trip endpoints in graph edge construction.
NA, NA_SOURCE, NA_SINK = "NA", "NA_SOURCE", "NA_SINK"

# Type alias: a callable that takes a DataFrame of locations and a PTNodeType
# and returns a tuple of (PT nodes DataFrame, PT edges DataFrame).
type PTLayerBuilder = Callable[[pl.DataFrame, PTNodeType], tuple[pl.DataFrame, pl.DataFrame]]

# Type alias: different ways to specify travel time between locations.
# Can be a fixed float (e.g. 5.0 minutes), a Polars expression, a precomputed
# DataFrame, a TravelTimeCalculator object, or any callable (loc_a, loc_b) -> float.
type TravelTimeFactory = float | pl.Expr | pl.DataFrame | TravelTimeCalculator | Callable[[str, str], float]

# Generic type variable allowing functions to accept either a Polars DataFrame
# or a GeoPandas GeoDataFrame and return the same type they received.
TDataFrame = TypeVar("TDataFrame", pl.DataFrame, gpd.GeoDataFrame)


class NetworkData(ABC):
    """
    Description: Abstract base class that wraps raw travel-survey DataFrames into a unified,
    filterable network view. Each dataset (Geneva, Toronto, etc.) inherits from this class
    and provides its own ``_copy`` implementation.

    Locations can be filtered by their ``type`` field (e.g. ``"subsector"``, ``"municipality_swiss"``).
    When a filter is active:
      - Only locations whose ``type`` is in the filter list are shown.
      - Only journeys whose both departure AND arrival are within filtered locations are shown.
      - Only users whose home location is within a filtered location are shown.

    Most properties are computed lazily and cached (``@cached_property``), so the first access
    triggers computation and subsequent accesses are instant.

    Use ``with_filter(loc_types)`` to create a new restricted copy without changing the original.
    Concrete subclasses must implement ``_copy``.
    """

    # The raw (unfiltered) journey legs table; conforms to USER_JOURNEY_SCHEMA.
    # Set once in __init__ and never mutated directly.
    _user_journeys_df: pl.DataFrame

    # The raw (unfiltered) locations spatial table; conforms to LOCATIONS_SCHEMA.
    # Set once in __init__ and never mutated directly.
    _locations_gdf: gpd.GeoDataFrame

    def __init__(
        self, user_journeys_df: pl.DataFrame, locations_gdf: gpd.GeoDataFrame, filters: list[str] | None = None
    ):
        """
        Description: Initialise the NetworkData by validating schemas, sorting locations,
        and precomputing aggregated journeys, location visits, and home locations.

        Input:
          - user_journeys_df (pl.DataFrame): Journey legs conforming to ``USER_JOURNEY_SCHEMA``.
            Each row is one leg of a multi-modal trip (e.g. walk + bus = 2 rows for same journey_id).
          - locations_gdf (gpd.GeoDataFrame): Locations conforming to ``LOCATIONS_SCHEMA``,
            e.g. subsectors, PT stops, municipalities. Will be sorted by ``loc_id`` in-place.
          - filters (list[str] | None): Optional list of location ``type`` values to restrict
            the view to. E.g. ``["subsector"]`` keeps only subsector-type locations.
            Pass ``None`` (default) to include all locations.

        Output:
          - (None): Initialises the instance; no return value.
        """
        # Sort locations alphabetically by loc_id so row-index lookups are deterministic
        locations_gdf = locations_gdf.sort_values("loc_id")

        # Validate and store the full, unfiltered journey legs table
        self._user_journeys_df = check_schema(user_journeys_df, USER_JOURNEY_SCHEMA)
        # Validate and store the full, unfiltered locations GeoDataFrame
        self._locations_gdf = check_schema(locations_gdf, LOCATIONS_SCHEMA)

        # Store a defensive copy of the filter list; None means "no filtering"
        self._type_filters = None if filters is None else filters.copy()
        # Precompute aggregated journeys (one row per trip, collapsing multi-leg journeys)
        self._aggregated_journeys = self._compute_agg_journeys(user_journeys_df)
        # Precompute how many times each user visited each location with each purpose
        self._location_visits = self._compute_location_visits(self._aggregated_journeys)
        # Precompute each user's home location (most frequent HOME-purpose location)
        self._home_locations = self._compute_home_locations(self._location_visits)

    @cached_property
    def user_journeys_df(self) -> pl.DataFrame:
        """
        Description: Filtered journey-legs table. Only rows whose departure AND arrival
        location belong to the currently active location-type filter are returned.
        Users whose home is outside the filter are also excluded.
        The result is cached after the first access.

        Output:
          - (pl.DataFrame): Journey legs conforming to USER_JOURNEY_SCHEMA, filtered by
            active location types and valid users.
        """
        # Filter rows where the departure location matches allowed location types,
        # and also removes users whose home is outside the filter
        dep_filtered = self._filter_loc_types(self._user_journeys_df, loc_id_col="dep_loc_id", filter_users=True)
        # Further filter rows where the arrival location also matches allowed types
        arr_filtered = self._filter_loc_types(dep_filtered, loc_id_col="arr_loc_id")

        return arr_filtered

    @cached_property
    def locations_gdf(self) -> gpd.GeoDataFrame:
        """
        Description: Filtered locations GeoDataFrame. Only locations whose ``type`` is in the
        active filter list are returned. The result is cached after the first access.

        Output:
          - (gpd.GeoDataFrame): Location rows conforming to LOCATIONS_SCHEMA, filtered by type.
        """
        return self._filter_loc_types(self._locations_gdf)

    @cached_property
    def locations_df(self) -> pl.DataFrame:
        """
        Description: Polars DataFrame version of the filtered locations GeoDataFrame.
        Converts the GeoDataFrame to a plain Polars DataFrame (dropping geometry).
        The result is cached after the first access.

        Output:
          - (pl.DataFrame): Same rows as ``locations_gdf`` but as a flat Polars DataFrame.
        """
        return utils.gdf_to_polars(self.locations_gdf)

    @cached_property
    def aggregated_journeys(self) -> pl.DataFrame:
        """
        Description: Filtered aggregated-journey table (one row per trip, not per leg).
        Multi-leg journeys are collapsed to a single row with summary info (departure/arrival
        location, purpose, modes list, etc.). Only journeys with both endpoints in filtered
        locations and from valid users are returned.

        Output:
          - (pl.DataFrame): Aggregated trips with columns: user_id, journey_id, num_legs,
            dep_day, dep_time, dep_purpose, dep_loc_id, arr_loc_id, arr_purpose, modes.
        """
        # Filter by departure location and valid users
        filtered_trips = self._filter_loc_types(self._aggregated_journeys, loc_id_col="dep_loc_id", filter_users=True)
        # Filter by arrival location
        filtered_trips = self._filter_loc_types(filtered_trips, loc_id_col="arr_loc_id")

        return filtered_trips

    @cached_property
    def location_visits(self) -> pl.DataFrame:
        """
        Description: Filtered location-visit counts per user, purpose, and location.
        Tells you how many times each user visited each location for each activity purpose.
        Only valid users (home inside filter) are included.

        Output:
          - (pl.DataFrame): Columns: user_id, purpose, loc_id, num_visits.
        """
        return self._filter_loc_types(self._location_visits, filter_users=True)

    @cached_property
    def home_locations(self) -> pl.DataFrame:
        """
        Description: Filtered table mapping each user to their home location.
        Computed from the most common HOME-purpose location per user.
        Only valid users (home inside filter) are included.

        Output:
          - (pl.DataFrame): Columns: user_id, loc_id (the home location ID).
        """
        return self._filter_loc_types(self._home_locations, filter_users=True)

    @cached_property
    def work_locations(self) -> pl.DataFrame:
        """
        Description: Table mapping each user to the unique locations they visited for work
        (Purpose.WORK_MAIN). A user may have multiple work locations.

        Output:
          - (pl.DataFrame): Columns: user_id, loc_id. One row per unique (user, work location) pair.
        """
        return self.location_visits.filter(purpose=Purpose.WORK_MAIN).select("user_id", "loc_id").unique()

    @cached_property
    def edu_locations(self) -> pl.DataFrame:
        """
        Description: Table mapping each user to the unique locations they visited for study
        (Purpose.STUDY). A user may have multiple education locations.

        Output:
          - (pl.DataFrame): Columns: user_id, loc_id. One row per unique (user, edu location) pair.
        """
        return self.location_visits.filter(purpose=Purpose.STUDY).select("user_id", "loc_id").unique()

    @cached_property
    def users_df(self) -> pl.DataFrame:
        """
        Description: Sorted user table with each user's home location.
        Derived from home_locations, sorted by user_id. Columns are renamed
        from ``loc_id`` to ``home_loc_id`` for clarity.

        Output:
          - (pl.DataFrame): Columns: user_id, home_loc_id. One row per user.
        """
        return self._filter_loc_types(self.home_locations).sort("user_id").select("user_id", home_loc_id="loc_id")

    @cached_property
    def user_ids(self) -> pl.Series:
        """
        Description: Sorted Series of all valid user IDs. A user is considered valid if
        their home location falls within the currently active location-type filter.
        If no filter is active, all users are returned.

        Output:
          - (pl.Series): String user IDs, sorted alphabetically/numerically.
        """
        if self._type_filters is None:
            # No filter: return all users who have a home location
            return self._home_locations["user_id"].unique().sort()

        # Only keep locations that match the current filter
        valid_locations = utils.gdf_to_polars(self._locations_gdf).filter(pl.col("type").is_in(self._type_filters))
        # Keep only users whose home is in a valid location
        valid_homes = self._home_locations.join(valid_locations.select("loc_id"), on="loc_id")

        # Return unique, sorted user IDs
        users = valid_homes["user_id"].unique().sort()
        return users

    @cached_property
    def num_obs_days_per_user(self) -> pl.DataFrame:
        """
        Description: Per-user observed-day count (t_i), i.e. how many distinct calendar
        days each user has travel records for. Used as a denominator when computing
        per-day visit rates.

        Output:
          - (pl.DataFrame): Columns: user_id, n_days (Int32 count of unique departure days).
        """
        return self.user_journeys_df.group_by("user_id").agg(n_days=pl.col("dep_day").n_unique().cast(pl.Int32))

    def with_filter(self, loc_types: str | list[str]) -> Self:
        """
        Description: Return a new instance of the same concrete NetworkData subclass,
        restricted to locations whose ``type`` field is in ``loc_types``.
        The original instance is NOT modified (immutable filter pattern).

        Input:
          - loc_types (str | list[str]): One or more location type strings to keep.
            E.g. ``"subsector"`` or ``["subsector", "municipality_swiss"]``.

        Output:
          - (Self): A new NetworkData of the same concrete type (e.g. GenevaData)
            with the given filter applied to all cached properties.
        """
        # Normalise single string to list for uniform handling
        filters = [loc_types] if isinstance(loc_types, str) else loc_types
        return self._copy(filters)

    def _filter_loc_types(self, df: TDataFrame, loc_id_col: str = "loc_id", filter_users: bool = False) -> TDataFrame:
        """
        Description: Internal helper that filters a DataFrame or GeoDataFrame by the
        active location-type filter. Rows whose ``loc_id_col`` column refers to a location
        that does not match any type in ``_type_filters`` are removed. Optionally also
        filters rows to only those belonging to valid users.

        Input:
          - df (TDataFrame): A Polars DataFrame or GeoPandas GeoDataFrame to filter.
          - loc_id_col (str): Name of the column containing location IDs to filter on.
            Defaults to ``"loc_id"``.
          - filter_users (bool): If True, also join against valid user IDs so that only
            rows for users with homes in filtered locations are kept. Requires a ``user_id``
            column in ``df``.

        Output:
          - (TDataFrame): Filtered version of the input ``df`` (same type as input).
        """
        # If no filter is active, return the input unchanged
        if self._type_filters is None:
            return df

        # Build a single-column DataFrame of valid location IDs (matching the filter)
        locations_df = utils.gdf_to_polars(self._locations_gdf)
        valid_loc_ids = locations_df.filter(pl.col("type").is_in(self._type_filters)).select(
            pl.col("loc_id").alias(loc_id_col)  # Rename to match the target column
        )

        # Optionally restrict to valid users first
        if filter_users and "user_id" not in df.columns:
            raise ValueError(f"Cannot find column `user_id` in columns {df.columns}")
        elif filter_users and isinstance(df, pl.DataFrame):
            # Polars join: keep only rows whose user_id is in the valid users Series
            df = df.join(self.user_ids.to_frame(), on="user_id")
        elif filter_users:
            # Pandas/GeoDataFrame merge: equivalent to an inner join on user_id
            df = df.merge(self.user_ids.to_frame().to_pandas(), on="user_id")

        # Now filter by location ID
        if isinstance(df, pl.DataFrame):
            return df.join(valid_loc_ids, on=loc_id_col)

        # GeoDataFrame path: use pandas merge
        return df.merge(valid_loc_ids.to_pandas(), on=loc_id_col)

    @abstractmethod
    def _copy(self, filters: list[str] | None = None): ...
    """
    Description: Abstract method that each subclass must implement.
    Should create a new instance of the same subclass with the given filters applied,
    sharing the same underlying raw data (``_user_journeys_df``, ``_locations_gdf``, etc.).

    Input:
      - filters (list[str] | None): List of location type strings to filter by, or None.

    Output:
      - (Self): A new instance of the concrete subclass with the filter applied.
    """

    @staticmethod
    def _compute_agg_journeys(user_journeys_df: pl.DataFrame) -> pl.DataFrame:
        """
        Description: Collapse a multi-leg journey table (one row per leg) into a
        single-row-per-trip summary. For each journey, extract the first departure
        info and the last arrival info, count legs, and collect all modes used.

        Input:
          - user_journeys_df (pl.DataFrame): Raw journey legs conforming to USER_JOURNEY_SCHEMA.
            Multiple rows may share the same (user_id, journey_id) if the trip had multiple legs.

        Output:
          - (pl.DataFrame): One row per journey with columns: user_id, journey_id, num_legs,
            dep_day, dep_time, dep_purpose, dep_loc_id, arr_loc_id, arr_purpose, modes (list).
        """
        # Group by journey identifier; maintain_order preserves original sort order
        group_keys = ["user_id", "journey_id"]
        grouped_journeys = user_journeys_df.group_by(group_keys, maintain_order=True)

        # Collect all transport modes used across legs into a list per journey
        modes = grouped_journeys.agg(modes="leg_mode")
        # Gather first and last rows of each group to get departure (first leg) and arrival (last leg) info
        trips = grouped_journeys.agg(pl.all().gather([0, -1])).select(
            "user_id",
            "journey_id",
            (pl.col("leg_id").list.last() + 1).alias("num_legs"),  # Number of legs = last leg_id + 1
            pl.col("dep_day").list.first(),       # Departure date = first leg's date
            pl.col("dep_time").list.first(),      # Departure time = first leg's start time
            pl.col("dep_purpose").list.first(),   # Activity at origin = first leg's purpose
            pl.col("dep_loc_id").list.first(),    # Origin location = first leg's departure
            pl.col("arr_loc_id").list.last(),     # Destination = last leg's arrival location
            pl.col("arr_purpose").list.last(),    # Activity at destination = last leg's purpose
        )

        # Join the modes list back onto the trip summary
        return trips.join(modes, on=group_keys)

    @staticmethod
    def _compute_location_visits(agg_journeys: pl.DataFrame) -> pl.DataFrame:
        """
        Description: Count how many times each user visited each location for each purpose.
        Both departure and arrival locations from aggregated journeys are counted as visits.

        Input:
          - agg_journeys (pl.DataFrame): Aggregated (one row per trip) journey table from
            ``_compute_agg_journeys``.

        Output:
          - (pl.DataFrame): Columns: user_id, purpose, loc_id, num_visits.
            Sorted by user_id.
        """
        # Treat departure location + purpose as a visit
        departures = agg_journeys.select("user_id", purpose="dep_purpose", loc_id="dep_loc_id")
        # Treat arrival location + purpose as a visit
        arrivals = agg_journeys.select("user_id", purpose="arr_purpose", loc_id="arr_loc_id")
        # Stack both sets of visits together
        all_visits = pl.concat([departures, arrivals])

        # Count visits per (user, purpose, location) triple
        return all_visits.group_by("user_id", "purpose", "loc_id").agg(num_visits=pl.len()).sort("user_id")

    @staticmethod
    def _compute_home_locations(location_visits: pl.DataFrame) -> pl.DataFrame:
        """
        Description: Determine each user's home location from the location-visit table.
        The home is the first location (arbitrarily, from the unique list) that the user
        visited with purpose HOME.

        Input:
          - location_visits (pl.DataFrame): Output of ``_compute_location_visits``.
            Columns: user_id, purpose, loc_id, num_visits.

        Output:
          - (pl.DataFrame): Columns: user_id, loc_id. One row per user giving their home location.
        """
        # Group locations by (user, purpose) to get all unique locations per purpose
        locations_by_purpose = (
            location_visits.group_by("user_id", "purpose").agg(pl.col("loc_id").unique()).sort("user_id")
        )
        # Filter to only HOME purpose, take the first location in the list as the home
        home_locations = (
            locations_by_purpose
            .filter(purpose=Purpose.HOME)
            .with_columns(pl.col("loc_id").list.first())  # Unwrap single-element list to scalar
            .drop("purpose")  # Drop the purpose column; it's always HOME here
        )

        return home_locations


def build_special_locations() -> gpd.GeoDataFrame:
    """
    Description: Create a GeoDataFrame containing a single special "NA" location used
    as a sentinel value for trips whose origin or destination is unknown, outside the
    study area, or cannot be geocoded. All datasets include this location so that
    journey records with missing endpoints still have a valid loc_id.

    Output:
      - (gpd.GeoDataFrame): A one-row GeoDataFrame with columns loc_id, loc_name, type,
        lon, lat, and Point geometry. The point is placed in the Lake Geneva area.
    """
    # Define the single NA location as a plain dictionary
    na_location = {
        "loc_id": "NA",       # Unique identifier used in journey records
        "loc_name": "NA",     # Human-readable name
        "type": "na",         # Location type label
        "lon": NA_LON,        # Longitude (Lake Geneva area)
        "lat": NA_LAT,        # Latitude (Lake Geneva area)
    }

    # Create a GeoDataFrame with a proper Point geometry and the project CRS
    return gpd.GeoDataFrame(
        [na_location],
        geometry=gpd.points_from_xy([NA_LON], [NA_LAT]),
        crs=CRS,
    )
