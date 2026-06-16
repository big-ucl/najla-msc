"""
Toronto THATS (Toronto Household Activity and Travel Survey) loader and NetworkData subclass.

This module handles loading and parsing the Toronto THATS survey dataset. The survey
records individual trip legs and hourly activity diaries for household members across
a 7-day observation period.

Key spatial unit: Census Tracts (CTs) within the Toronto Census Metropolitan Area (CMA).
Each CT is treated as a "subsector" location.

Key classes/functions:
  - ``TorontoInputs``: Frozen dataclass holding all parsed raw input files.
  - ``TorontoData``: NetworkData subclass for Toronto with caching and demographic features.
  - ``load_files``: Reads all raw Toronto files from disk.
  - ``build_toronto_data``: Orchestrates parsing from raw inputs to standardised TorontoData.
  - ``build_toronto_journeys``: Parses raw trip records into standardised journey DataFrame.
  - ``build_toronto_activities``: Converts wide hourly activity diary to long collapsed format.
  - ``build_toronto_users``: Builds user demographics and home locations.
  - ``collapse_activities``: Merges consecutive same-purpose hourly entries into activity spans.
"""

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import geopandas as gpd
import pandas as pd
import polars as pl
import polars.selectors as cs

from activitygraphs.base import CRS, LOCATIONS_COLUMNS, LOCATIONS_SCHEMA, USER_JOURNEY_SCHEMA, Mode, Purpose
from activitygraphs.config import TorontoDataConfig
from activitygraphs.network import NA, NetworkData, build_special_locations
from activitygraphs.utils import (
    DataFrameStore,
    add_lon_lat_from_centroid,
    check_schema,
    get_project_root,
    invert_mapping,
)

# Statistics Canada identifier for the Toronto Census Metropolitan Area (CMA).
# Used to spatially filter census tract boundaries to the Toronto study area only.
TORONTO_CMA = 535

# Mapping from THATS predicted mode strings (English) to the internal Mode enum.
# The THATS survey uses an ML-predicted mode classifier; these are the output labels.
MODE_MAP = {
    "AIR_OR_HSR": Mode.OTHER,
    "BICYCLING": Mode.CYCLE,
    "BUS": Mode.BUS,
    "CAR": Mode.CAR,
    "FERRY": Mode.BOAT,
    "LIGHT_RAIL": Mode.TRAMWAY,
    "SUBWAY": Mode.TRAIN,
    "TRAIN": Mode.TRAIN,
    "TRAM": Mode.TRAMWAY,
    "TROLLEYBUS": Mode.BUS,
    "UNKNOWN": Mode.UNKNOWN,
    "WALKING": Mode.WALK,
}

# Mapping from THATS manually-annotated mode strings to the internal Mode enum.
# Participants could enter free-text mode descriptions; this dict maps the most
# common variants to the canonical Mode values. Takes precedence over MODE_MAP.
MANUAL_MODE_MAP = {
    "walk": Mode.WALK,
    "car-park-off-street-res": Mode.CAR,
    "car-park-off-street-non-res": Mode.CAR,
    "transit": Mode.BUS,
    "car-passenger": Mode.VEH_PASS,
    "bike": Mode.CYCLE,
    "car-park-on-street-res": Mode.CAR,
    "car-park-on-street-non-res": Mode.CAR,
    "train": Mode.TRAIN,
    "taxi": Mode.TAXI,
    "train-transit": Mode.TRAIN,
    "golf_cart": Mode.OTHER,
    "operate_heavy equipment": Mode.OTHER,
    "work/tractor": Mode.OTHER,
    "run": Mode.WALK,
    "boat": Mode.BOAT,
    "shunting": Mode.CAR,
    "auto": Mode.CAR,
    "driving_company vehicle for work": Mode.CAR,
    "motorcycle": Mode.MOTORCYCLE,
    '"auto': Mode.CAR,
    "auto_(drive) parked in driveway": Mode.CAR,
    "airplane": Mode.OTHER,
    "auto_(work related)": Mode.CAR,
    "car": Mode.CAR,
    "canoe": Mode.BOAT,
    "subway": Mode.TRAIN,
    "not_aplicable": Mode.UNKNOWN,
    "transit_+ walk": Mode.BUS,
    "walk_and transit": Mode.BUS,
    "auto_(work)": Mode.CAR,
    "delivery": Mode.CAR,
    '"walking': Mode.WALK,
    "ferry": Mode.BOAT,
    "sail": Mode.BOAT,
}

# Mapping from THATS manually-annotated trip purpose strings to the internal Purpose enum.
# Participants entered free-text trip purposes; this dict maps common variants.
PURPOSE_MAP = {
    str(Purpose.UNKNOWN): Purpose.UNKNOWN,
    "home": Purpose.HOME,
    "entertainment-visit": Purpose.ENTERTAINMENT,
    "excercice-recreation": Purpose.LEISURE_OTHER,
    "work": Purpose.WORK_MAIN,
    "shopping-grocerie": Purpose.SHOP,
    "pick-drop": Purpose.ESCORT,
    "appointment-visit": Purpose.PERSONAL,
    "personal-business": Purpose.PERSONAL,
    "school": Purpose.STUDY,
    "dog_walk": Purpose.LEISURE_OTHER,
    "church": Purpose.LEISURE_OTHER,
    "shopping": Purpose.SHOP,
    "golf": Purpose.LEISURE_OTHER,
    "walk_dog": Purpose.LEISURE_OTHER,
    "shunting": Purpose.ESCORT,
    "return_home": Purpose.HOME,
    "volunteering": Purpose.LEISURE_OTHER,
    "vacation": Purpose.LONG_DISTANCE_TRIP,
    "volunteer": Purpose.LEISURE_OTHER,
    "visit_family": Purpose.VISIT,
    "lunch": Purpose.ENTERTAINMENT,
    "travel": Purpose.LONG_DISTANCE_TRIP,
    "religious": Purpose.LEISURE_OTHER,
    "going_home": Purpose.HOME,
    "parcel_deliveries in brampton": Purpose.WORK_OTHER,
    "library": Purpose.LEISURE_OTHER,
    "cat_sitting for a friend": Purpose.VISIT,
    "coffee": Purpose.ENTERTAINMENT,
    "family_visit": Purpose.VISIT,
    "walk": Purpose.LEISURE_OTHER,
    "traveling": Purpose.LONG_DISTANCE_TRIP,
    # More can be added
}

# Mapping from THATS numeric activity type codes (integers) to internal Purpose enum values.
# The THATS activity diary uses numeric codes for each activity type; this dict maps them.
# ``invert_mapping`` is called because multiple activity codes map to the same Purpose,
# so we define the reverse (Purpose -> list of codes) and invert it.
ACTIVITY_MAP = invert_mapping({
    Purpose.OTHER: [41],
    Purpose.UNKNOWN: [43],
    Purpose.HOME: [*range(1, 14)],
    Purpose.WORK_MAIN: [14],
    Purpose.WORK_OTHER: [],
    Purpose.STUDY: [15],
    Purpose.VISIT: [17, 40],
    Purpose.ESCORT: [26],
    Purpose.PERSONAL: [*range(36, 40)],
    Purpose.SHOP: [*range(29, 36)],
    Purpose.ENTERTAINMENT: [16, 18, 22, 24],
    Purpose.LEISURE_OTHER: [19, 20, 21, 23, 25, 27, 28],
    Purpose.LONG_DISTANCE_TRIP: [42],
})


@dataclass(frozen=True)
class TorontoInputs:
    """
    Description: Immutable container (frozen dataclass) holding all raw input DataFrames
    and GeoDataFrames parsed from disk for the Toronto THATS survey. Passed to
    ``build_toronto_data`` and stored on the ``TorontoData`` instance.

    Attributes:
      - raw_journeys_df (pl.DataFrame): Raw trip-leg records (one row per GPS-detected leg).
      - raw_person_df (pl.DataFrame): Person-level demographics and survey metadata.
        Includes person_id, driving licence, transit pass, experiment start date.
      - raw_household_df (pl.DataFrame): Household-level demographics.
        Includes hh_id, home census tract, number of adults/children/vehicles/bikes.
      - raw_activities_df (pl.LazyFrame): Wide-format activity diary (lazily loaded).
        Columns: hh_id, person_id, Hour (e.g. "0h-d1"), then 7 activity slots per hour.
      - metropolitan_areas_gdf (gpd.GeoDataFrame): CMA boundary polygons.
        Used to spatially filter census tracts to the Toronto CMA.
      - census_tracts_gdf (gpd.GeoDataFrame): Census tract (CT) polygon boundaries.
        These are the subsector locations for Toronto.
      - dissemination_areas_gdf (gpd.GeoDataFrame): Dissemination area polygon boundaries.
        Finer-grained census units; loaded for potential future use.
    """

    raw_journeys_df: pl.DataFrame      # Raw trip-leg GPS records from the survey
    raw_person_df: pl.DataFrame        # Person demographics and survey metadata
    raw_household_df: pl.DataFrame     # Household demographics (home location, vehicles, etc.)
    raw_activities_df: pl.LazyFrame    # Wide-format hourly activity diary (lazy for memory)

    metropolitan_areas_gdf: gpd.GeoDataFrame   # CMA boundary polygons
    census_tracts_gdf: gpd.GeoDataFrame        # Census tract (CT/subsector) polygons
    dissemination_areas_gdf: gpd.GeoDataFrame  # Dissemination area polygons (finer grain)


class TorontoData(NetworkData, DataFrameStore):
    """
    Description: Concrete NetworkData subclass for the Toronto THATS survey.
    Extends NetworkData with Toronto-specific features:
      - Individual demographic features (household size, vehicles, driving licence, transit pass).
      - Activity diary data (hourly activity records collapsed into time spans).
      - Home locations derived from the household survey rather than from trip records.
      - User IDs sourced directly from the users DataFrame (not inferred from trip purposes).

    Inherits disk-caching from ``DataFrameStore``.
    """

    def __init__(
        self,
        inputs: TorontoInputs,
        locations_gdf: gpd.GeoDataFrame,
        user_journeys_df: pl.DataFrame,
        activities_df: pl.DataFrame,
        users_df: pl.DataFrame,
        filters: list[str] | None = None,
    ):
        """
        Description: Initialise TorontoData with all required DataFrames and the raw inputs.

        Input:
          - inputs (TorontoInputs): All raw parsed input data for this dataset.
          - locations_gdf (gpd.GeoDataFrame): Standardised location table (census tracts + NA).
          - user_journeys_df (pl.DataFrame): Standardised trip-leg table.
          - activities_df (pl.DataFrame): Collapsed hourly activity diary.
          - users_df (pl.DataFrame): User demographics with home location.
          - filters (list[str] | None): Optional location-type filter.

        Output:
          - (None): Initialises the instance; no return value.
        """
        super().__init__(user_journeys_df, locations_gdf, filters)
        # Store raw inputs for reference
        self.inputs = inputs
        # Activity diary: one row per contiguous activity span per user per day
        self.activities_df = activities_df
        # Store the user demographics table privately; accessed via the users_df property
        self._users_df = users_df

    @cached_property
    def home_locations(self) -> pl.DataFrame:
        """
        Description: Return each user's home location from the users_df table.
        Overrides NetworkData's home_locations which infers home from trip purposes.
        Toronto's home location comes directly from the household survey.

        Output:
          - (pl.DataFrame): Columns: user_id, loc_id. One row per user.
        """
        return self.users_df.select("user_id", loc_id="home_loc_id")

    @cached_property
    def users_df(self) -> pl.DataFrame:
        """
        Description: Filtered and sorted user demographics table. Only users whose
        home location matches the active location-type filter are included.

        Output:
          - (pl.DataFrame): Columns: user_id, home_loc_id, plus demographic columns.
            Sorted by user_id.
        """
        return self._filter_loc_types(self._users_df, "home_loc_id").sort("user_id")

    @cached_property
    def user_ids(self) -> pl.Series:
        """
        Description: Sorted Series of all valid user IDs, derived from the filtered users_df.
        Overrides NetworkData.user_ids since valid users are defined by their home location
        in the users table, not by trip home-purpose inference.

        Output:
          - (pl.Series): String user IDs, sorted.
        """
        return self.users_df["user_id"].sort()

    @cached_property
    def num_obs_days_per_user(self) -> pl.DataFrame:
        """
        Description: Per-user observed-day count (0–7) from the activity diary.
        Unlike the base class which counts trip departure days, Toronto uses the
        diary's unique activity dates as the day count (more reliable).

        Output:
          - (pl.DataFrame): Columns: user_id, n_days (Int32).
        """
        return (
            self.activities_df
            .group_by("person_id")
            .agg(n_days=pl.col("act_date").n_unique().cast(pl.Int32))
            .rename({"person_id": "user_id"})  # Align column name with NetworkData convention
        )

    def _copy(self, filters: list[str] | None = None):
        """
        Description: Create a new TorontoData instance sharing the same raw data
        but with a different location-type filter.

        Input:
          - filters (list[str] | None): New filter list.

        Output:
          - (TorontoData): New instance with the filter applied.
        """
        return TorontoData(
            self.inputs, self._locations_gdf, self._user_journeys_df, self.activities_df, self.users_df, filters
        )

    @classmethod
    def load(cls, cfg: TorontoDataConfig, project_root: Path | None = None, name: str | None = None) -> "TorontoData":
        """
        Description: Load TorontoData from a Parquet cache if available, otherwise build
        it from scratch and save to disk.

        Input:
          - cfg (TorontoDataConfig): Toronto-specific configuration with file paths.
          - project_root (Path | None): Project root directory.
          - name (str | None): Optional cache sub-directory name.

        Output:
          - (TorontoData): Loaded or freshly built TorontoData instance.
        """
        # Resolve cache directory
        project_root, data_dir = cls._dirs(cfg, project_root, name)
        # Always load raw inputs (needed even when loading from cache, for the inputs attribute)
        toronto_inputs = load_files(cfg, project_root)

        if data_dir.exists():
            # Cache hit: load all four parquet files
            locations_gdf = gpd.read_parquet(data_dir / "locations_gdf.parquet")
            user_journeys_df = pl.read_parquet(data_dir / "user_journeys_df.parquet", schema=USER_JOURNEY_SCHEMA)
            activities_df = pl.read_parquet(data_dir / "activities_df.parquet")
            users_df = pl.read_parquet(data_dir / "users_df.parquet")

            return cls(toronto_inputs, locations_gdf, user_journeys_df, activities_df, users_df)
        else:
            # Cache miss: build from raw and save
            data = build_toronto_data(toronto_inputs)
            data.save(cfg, project_root, name)

            return data

    def save(self, cfg: TorontoDataConfig, project_root: Path | None = None, name: str | None = None):
        """
        Description: Persist all four processed DataFrames to Parquet for fast reloading.

        Input:
          - cfg (TorontoDataConfig): Configuration with processed data path.
          - project_root (Path | None): Project root directory.
          - name (str | None): Optional cache sub-directory name.

        Output:
          - (None): Writes files to disk; no return value.
        """
        # Resolve cache directory and create it if needed
        project_root, data_dir = self._dirs(cfg, project_root, name)

        data_dir.mkdir(parents=True, exist_ok=True)
        # Save the filtered locations GeoDataFrame (preserves polygon geometry)
        self.locations_gdf.to_parquet(data_dir / "locations_gdf.parquet")
        # Save the journey legs table
        self.user_journeys_df.write_parquet(data_dir / "user_journeys_df.parquet")
        # Save the collapsed activity diary
        self.activities_df.write_parquet(data_dir / "activities_df.parquet")
        # Save user demographics with home locations
        self.users_df.write_parquet(data_dir / "users_df.parquet")


def load_files(cfg: TorontoDataConfig, project_root: Path | None = None) -> TorontoInputs:
    """
    Description: Read all raw Toronto THATS input files from disk into a ``TorontoInputs``
    container. Survey data is loaded eagerly (Parquet); the activity diary is loaded
    lazily (scan_parquet) because it is large.

    Input:
      - cfg (TorontoDataConfig): Configuration specifying file paths for all inputs.
      - project_root (Path | None): Root directory for resolving relative paths.

    Output:
      - (TorontoInputs): Frozen dataclass with all raw DataFrames and GeoDataFrames.
    """
    # Resolve the project root using the utility function
    project_root = get_project_root(project_root)

    # Directory containing all raw survey parquet files
    raw_data_dir = project_root / cfg.paths.raw
    # Load trip-leg records (one row per detected leg)
    raw_journeys_df = pl.read_parquet(raw_data_dir / cfg.inputs.raw_journeys)
    # Load person demographics and survey metadata
    raw_persons_df = pl.read_parquet(raw_data_dir / cfg.inputs.raw_person)
    # Load household demographics (home CT, vehicle counts, etc.)
    raw_household_df = pl.read_parquet(raw_data_dir / cfg.inputs.raw_household)
    # Lazily scan the activity diary (large file: one row per person-hour-slot combination)
    raw_activities_df = pl.scan_parquet(raw_data_dir / cfg.inputs.raw_activities)

    # Directory containing the Statistics Canada boundary shapefiles
    boundaries_dir = project_root / cfg.inputs.boundaries.directory
    # CMA boundary: defines the Toronto study area extent
    boundary_cma = gpd.read_file(boundaries_dir / cfg.inputs.boundaries.metropolitan_areas)
    # Census tract boundaries: these are the subsector locations for Toronto
    boundary_ct = gpd.read_file(boundaries_dir / cfg.inputs.boundaries.census_tracts)
    # Dissemination area boundaries: finer grain, loaded for potential future use
    boundary_da = gpd.read_file(boundaries_dir / cfg.inputs.boundaries.dissemination_areas)

    return TorontoInputs(
        raw_journeys_df, raw_persons_df, raw_household_df, raw_activities_df, boundary_cma, boundary_ct, boundary_da
    )


def build_toronto_data(inputs: TorontoInputs) -> TorontoData:
    """
    Description: Orchestrate the full parsing pipeline from raw Toronto inputs to a
    standardised ``TorontoData`` instance. Builds locations, journeys, activities, and
    user demographics in dependency order.

    Input:
      - inputs (TorontoInputs): All raw parsed files from ``load_files``.

    Output:
      - (TorontoData): Standardised TorontoData instance with validated schemas.
    """
    # Step 1: Build the location vocabulary (census tracts + NA sentinel)
    locations_gdf = build_toronto_locations(inputs)
    # Step 2: Parse raw trip legs into the standardised journey format
    user_journeys_df = build_toronto_journeys(inputs, locations_gdf)
    # Step 3: Parse the hourly activity diary into collapsed activity spans
    activities_df = build_toronto_activities(inputs, user_journeys_df)
    # Step 4: Build the user demographics table (home location, household attributes)
    users_df = build_toronto_users(inputs, locations_gdf, user_journeys_df)

    return TorontoData(inputs, locations_gdf, user_journeys_df, activities_df, users_df)


# =========================================
# Locations
# =========================================


def build_toronto_locations(inputs: TorontoInputs) -> gpd.GeoDataFrame:
    """
    Description: Build the Toronto location vocabulary by concatenating the NA sentinel
    and the census tract subsector locations into a single GeoDataFrame.

    Input:
      - inputs (TorontoInputs): Raw inputs containing census tract boundaries.

    Output:
      - (gpd.GeoDataFrame): Combined location table with the NA sentinel and all
        Toronto CMA census tract subsectors. Conforms to LOCATIONS_SCHEMA.
    """
    # Build the single "NA" sentinel location
    special_locations = build_special_locations()
    # Build census tract locations (filtered to Toronto CMA)
    subsector_locations = build_subsector_locations(inputs)
    # Concatenate and return with the project CRS
    return gpd.GeoDataFrame(
        pd.concat(
            [
                special_locations,
                subsector_locations,
            ],
            ignore_index=True,
        ),
        crs=CRS,
    )


def build_subsector_locations(inputs: TorontoInputs) -> gpd.GeoDataFrame:
    """
    Description: Extract census-tract polygon boundaries within the Toronto CMA boundary
    via a spatial join, and return them as "subsector" locations with centroid coordinates.

    Input:
      - inputs (TorontoInputs): Raw inputs with metropolitan_areas_gdf and census_tracts_gdf.

    Output:
      - (gpd.GeoDataFrame): One row per CT within the Toronto CMA. Columns: loc_id (CTUID),
        loc_name (CTNAME), type ("subsector"), lon, lat, geometry. Validates against
        LOCATIONS_SCHEMA.
    """
    # Filter the CMA boundaries to just the Toronto CMA using its Statistics Canada code
    toronto_cma = inputs.metropolitan_areas_gdf.query(f"CMAUID == '{TORONTO_CMA}'")
    # Use local UTM for accurate spatial join (avoids issues with geographic CRS)
    utm_crs = toronto_cma.estimate_utm_crs()

    # Spatial join: keep only census tracts that intersect the Toronto CMA boundary
    boundaries_gdf = inputs.census_tracts_gdf
    boundaries_gdf = boundaries_gdf.to_crs(utm_crs).sjoin(toronto_cma.to_crs(utm_crs))
    # Drop the join helper column and reset to clean integer index; reproject to project CRS
    boundaries_gdf = boundaries_gdf.drop(columns=["index_right"]).reset_index(drop=True).to_crs(CRS)

    # Rename Statistics Canada columns to the standard location vocabulary
    boundaries_gdf = boundaries_gdf.rename(columns={"CTUID": "loc_id", "CTNAME": "loc_name"})
    boundaries_gdf["type"] = "subsector"  # All census tracts are subsector locations
    # Compute centroid-based lon/lat columns from the polygon geometry
    boundaries_gdf = add_lon_lat_from_centroid(boundaries_gdf, index_col="loc_id")

    # Keep only the standard columns and validate the schema
    # noinspection PyTypeChecker
    gdf: gpd.GeoDataFrame = boundaries_gdf[LOCATIONS_COLUMNS].copy()

    return check_schema(gdf, LOCATIONS_SCHEMA)


# =========================================
# Journeys
# =========================================


def build_toronto_journeys(inputs: TorontoInputs, locations_gdf: gpd.GeoDataFrame) -> pl.DataFrame:
    """
    Description: Parse the raw THATS trip-leg records into a standardised journey DataFrame
    conforming to USER_JOURNEY_SCHEMA. Handles mode translation (predicted + manual override),
    purpose translation, leg ranking within trips, and location ID validation.

    Input:
      - inputs (TorontoInputs): Raw inputs with trip legs and person mapping.
      - locations_gdf (gpd.GeoDataFrame): The Toronto location vocabulary (used to validate
        that census tract IDs in the trips exist in the location table).

    Output:
      - (pl.DataFrame): Standardised trip-leg records conforming to USER_JOURNEY_SCHEMA.
        Sorted by user_id, dep_day, dep_time.
    """
    # Raw trip legs DataFrame from the survey
    trips = inputs.raw_journeys_df
    # Person table used to map from app_id (device) to person_id (survey respondent)
    persons = inputs.raw_person_df

    # Replace app_id with user_id
    # The survey uses app_id (mobile device) to record trips; person_id is the true user ID
    app_to_user_id_map = persons.select("app_id", "person_id").unique()
    user_journeys_df = trips.join(app_to_user_id_map, on="app_id")

    # Replace transport modes
    # Use the ML-predicted mode as default; override with manually annotated mode if available
    user_journeys_df = user_journeys_df.with_columns(
        pl.col("predicted_modes").replace_strict(MODE_MAP, default=Mode.UNKNOWN),  # ML prediction
        pl.col("manual_mode").replace_strict(MANUAL_MODE_MAP, default=None),        # Manual override
    ).with_columns(
        pl
        .when(pl.col("manual_mode").is_not_null())  # If manual annotation exists, use it
        .then(pl.col("manual_mode"))
        .otherwise("predicted_modes")               # Otherwise fall back to ML prediction
        .alias("leg_mode")
    )

    # Rank trip legs by start time
    # Within each (app_id, trip_id) group, rank legs by start time to get a 0-based leg index
    leg_rank_expr = pl.col("start_fmt_time_section").rank("dense").over("app_id", "trip_id") - 1

    # Replace trip purposes
    # Use manual purpose annotation; fill nulls with UNKNOWN; any unmapped value → OTHER
    purpose_expr = (
        pl.col("manual_purpose").fill_null(Purpose.UNKNOWN).replace_strict(PURPOSE_MAP, default=Purpose.OTHER)
    )

    # Select, rename, and parse columns to match USER_JOURNEY_SCHEMA
    user_journeys_df = user_journeys_df.select(
        user_id="person_id",            # Survey respondent ID
        journey_id="trip_id",           # Trip ID (groups legs of the same trip)
        leg_id=leg_rank_expr.cast(pl.Int8),  # 0-based leg index within trip
        leg_mode=pl.col("leg_mode").cast(pl.Categorical),  # Translated transport mode
        leg_line=pl.lit(None, dtype=pl.String),             # No line info in Toronto
        dep_datetime=pl.col("start_fmt_time_section").str.to_datetime(format="%+", time_zone="EST"),
        dep_loc_id=pl.col("start_loc_CT_section"),   # Census tract ID of origin
        arr_loc_id=pl.col("end_loc_CT_section"),     # Census tract ID of destination
        arr_datetime=pl.col("end_fmt_time_section").str.to_datetime(format="%+", time_zone="EST"),
        arr_purpose=purpose_expr,                     # Activity purpose at destination
    )

    # Handle duration and start times/dates
    # Split the datetime into date and time components; compute leg duration
    user_journeys_df = user_journeys_df.with_columns(
        dep_day=pl.col("dep_datetime").dt.date(),                       # Calendar date
        dep_time=pl.col("dep_datetime").dt.time(),                      # Time of day
        duration=pl.col("arr_datetime") - pl.col("dep_datetime"),       # Leg duration
    ).drop("dep_datetime", "arr_datetime")  # Remove temporary datetime columns

    # Add departure purposes (purpose of previous trip)
    # Not a valid assumption, as trips are not necessarily linked => comment out
    """unique_trips = (
        user_journeys_df
        .sort("user_id", "dep_day", "dep_time")
        .unique(["user_id", "journey_id"], keep="first")
        .select(
            "journey_id",
            "arr_purpose",
            previous_journey_id=pl.col("journey_id").shift(1).over(["user_id", "dep_day"], order_by="dep_time"),
        )
    )

    dep_purposes = unique_trips.join(
        unique_trips.select("journey_id", dep_purpose="arr_purpose"),
        left_on="previous_journey_id",
        right_on="journey_id",
    ).drop("arr_purpose", "previous_journey_id")

    user_journeys_df = user_journeys_df.join(dep_purposes, on="journey_id", how="left").with_columns(
        pl.col("arr_purpose").cast(pl.Categorical),
        pl.col("dep_purpose").fill_null(Purpose.UNKNOWN).cast(pl.Categorical),
    )"""

    # Departure purpose is set to UNKNOWN for all trips (see commented block above)
    user_journeys_df = user_journeys_df.with_columns(
        pl.col("arr_purpose").cast(pl.Categorical),
        pl.lit(Purpose.UNKNOWN).alias("dep_purpose").cast(pl.Categorical),  # Default UNKNOWN
    )

    # Replace null loc_ids with NA
    # Some trips have missing census tract IDs; replace with the "NA" sentinel
    user_journeys_df = user_journeys_df.with_columns(
        pl.col("dep_loc_id").fill_null(NA), pl.col("arr_loc_id").fill_null(NA)
    )

    # Replace unknown locations with NA
    # If the census tract ID doesn't exist in the location vocabulary, replace with "NA"
    locations_ids = locations_gdf["loc_id"]
    user_journeys_df = user_journeys_df.with_columns(
        dep_loc_id=pl.when(pl.col("dep_loc_id").is_in(locations_ids)).then(pl.col("dep_loc_id")).otherwise(pl.lit(NA)),
        arr_loc_id=pl.when(pl.col("arr_loc_id").is_in(locations_ids)).then(pl.col("dep_loc_id")).otherwise(pl.lit(NA)),
    )

    # Select only the schema columns in order and sort chronologically
    return user_journeys_df.select(USER_JOURNEY_SCHEMA.keys()).sort(
        "user_id",
        "dep_day",
        "dep_time",
    )


# =========================================
# Activities
# =========================================


def build_toronto_activities(inputs: TorontoInputs, user_journeys_df: pl.DataFrame):
    """
    Description: Parse the wide-format THATS hourly activity diary into a long, collapsed
    activity DataFrame. The diary records up to 7 concurrent activities per hour for each
    person over a 7-day period. This function converts it to one row per contiguous
    activity span (same purpose, consecutive hours).

    Steps:
      1. Unpivot the wide hourly×index format to a long format.
      2. Filter to only users who also have trip records.
      3. Parse the "Hour" column (e.g. "0h-d1") to hour-of-day and day-of-week.
      4. Convert relative day numbers to calendar dates using each person's start date.
      5. Translate numeric activity types to Purpose enum values.
      6. Collapse consecutive same-purpose hours into activity spans.
      7. Drop rows with null purpose (empty activity slots).

    Input:
      - inputs (TorontoInputs): Raw inputs with raw_activities_df and raw_person_df.
      - user_journeys_df (pl.DataFrame): Journey table to identify valid user IDs.

    Output:
      - (pl.DataFrame): One row per contiguous activity span with columns:
        hh_id, person_id, act_date, act_purpose, start_time, end_time, n_hours.
    """
    # Move from a wide to long data format by adding an "index" column for activities that happen in the same hour
    # The raw diary has columns like "1_ActivityType", "2_ActivityType", ... for up to 7 concurrent activities
    activities = unpivot_activities(inputs.raw_activities_df)

    # Only keep users with recorded trips
    # The activity diary may include household members who didn't record any trips
    user_ids = user_journeys_df["user_id"].unique()
    activities = activities.filter(pl.col("person_id").is_in(user_ids.implode()))

    # Parse Hour column into a day and hour column
    # The "Hour" column format is e.g. "0h-d1": hour 0, day 1
    activities = (
        activities
        .with_columns(
            act_hour=pl.col("Hour").str.split("-").list.first(),   # Extract "0h" part → "0"
            act_day=pl.col("Hour").str.split("d").list.last().cast(int),  # Extract day number → 1
        )
        .with_columns(
            # Convert "0" → "00:00" → time object
            (pl.col("act_hour").str.pad_start(2, "0") + ":00").str.to_time("%H:%M"),
            pl.col("act_day").cast(int)
        )
        .drop("Hour")  # Remove the original Hour string column
    )

    # Replace day numbering with actual activity dates
    # The diary uses relative day numbers (1-7); we convert to calendar dates using each
    # person's experiment start date from the person table.
    start_dates = inputs.raw_person_df.select(
        "person_id", start_date=pl.col("THATS Experiment start date").str.to_date("%Y-%m-%d")
    ).lazy()

    activities = (
        activities
        .join(start_dates, on="person_id", how="left")
        # Compute actual date: start_date + (act_day - 1) days (or adjust as needed)
        .with_columns(act_day=pl.col("start_date") + pl.duration(days="act_day"))
        .drop("start_date")  # Remove the temporary start_date column
    )

    # Define the index columns vs value columns for schema ordering
    row_index_cols = ["hh_id", "person_id", "act_day", "act_hour", "act_index"]
    value_cols = [col for col in activities.collect_schema().names() if col not in row_index_cols]

    # Sort by index columns and rename act_day → act_date for clarity
    activities = activities.sort(row_index_cols).select(*row_index_cols, *value_cols).rename({"act_day": "act_date"})

    # Replace activity type with Purposes
    # ActivityType is stored as a numeric string; cast to int and map to Purpose enum
    activities = activities.with_columns(
        pl.col("ActivityType").cast(float).cast(int).replace_strict(ACTIVITY_MAP, return_dtype=pl.Categorical)
    ).rename({"ActivityType": "act_purpose"})

    # Collapse hourly breakdown into list of activities with start and end time
    # Consecutive hours with the same purpose are merged into a single activity span
    activities = collapse_activities(activities)

    # Filter out empty purpose activities
    # Null purpose means the activity slot was unused (not all 7 slots are always filled)
    activities = activities.drop_nulls("act_purpose")

    return activities.collect()


def unpivot_activities(activity_df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Description: Convert the wide-format THATS activity diary (one row per person-hour,
    with up to 7 activity columns per value type) into a long-format LazyFrame where
    each row represents one activity slot (person-hour-index combination).

    The input has columns like ``1_ActivityType``, ``2_ActivityType``, ..., ``7_ActivityType``
    for each of the 11 value types. This function unpivots each value type separately
    and joins the results on the (hh_id, person_id, Hour, act_index) primary key.

    Input:
      - activity_df (pl.LazyFrame): Wide-format activity diary with columns:
        hh_id, person_id, Hour, and up to 7 columns per value type (prefixed 1–7).

    Output:
      - (pl.LazyFrame): Long-format LazyFrame with columns:
        hh_id, person_id, Hour, act_index (1–7), StrtTripID, EndTripID,
        ActivityType, ActivityTypeCategory, and other activity attributes.
    """
    # The 11 activity attribute columns that need to be unpivoted from wide to long
    value_cols = [
        "StrtTripID",              # Trip ID of the trip that started during this activity
        "EndTripID",               # Trip ID of the trip that ended during this activity
        "ActivityType",            # Numeric activity type code (maps to Purpose)
        "ActivityTypeCategory",    # Higher-level activity category
        "ActivityWithPartner",     # Binary: activity done with partner
        "ActivityWithChild",       # Binary: activity done with child
        "ActivityWithOtherRel",    # Binary: activity done with other relative
        "ActivityWithFriends",     # Binary: activity done with friends
        "ActivityWithOther",       # Binary: activity done with other person
        "ActivityHorizon",         # Planning horizon for this activity
        "ActivityExpenditure",     # Money spent during this activity
    ]

    # Row identifiers that stay fixed during unpivoting
    row_indentifiers = ["hh_id", "person_id", "Hour"]
    # Primary key for joining: row identifiers + activity slot index (1-7)
    primary_key = [*row_indentifiers, "act_index"]
    # Start with a base DataFrame that has one row per (person, hour, slot index)
    unpivoted_df = (
        activity_df
        .select(*row_indentifiers, act_index=list(range(1, 8)))  # 7 slots per hour
        .explode("act_index")          # One row per slot
        .unique(maintain_order=True)   # Deduplicate (some slots may be empty)
    )

    # Unpivot each value type and join it onto the base DataFrame
    for value_name in value_cols:
        df = (
            activity_df
            # Select the row identifiers and the 7 columns for this value type
            .select(*row_indentifiers, cs.ends_with(value_name))
            .unpivot(
                index=row_indentifiers,   # Keep these as row identifiers
                variable_name="act_index",  # The slot number (e.g. "1_ActivityType" → "1")
                value_name=value_name,      # The actual value
            )
            # Extract just the first character (slot number) from "1_ActivityType" → 1
            .with_columns(pl.col("act_index").str.slice(0, 1).cast(int))
        )

        # Left join onto the base: adds one value column per iteration
        unpivoted_df = unpivoted_df.join(df, on=primary_key, how="left")

    return unpivoted_df


def collapse_activities(activity_df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Description: Merge consecutive same-purpose hourly activity rows into contiguous
    activity spans. For example, if a person works at HOME from 09:00 to 12:00
    (3 consecutive rows), these are collapsed into one row with start_time=09:00,
    end_time=11:00 (last hour), and n_hours=3.

    A new span is started whenever:
      - There is no previous occurrence of this purpose that day (first occurrence), OR
      - The time gap between consecutive same-purpose hours exceeds 1 hour (gap in diary).

    Input:
      - activity_df (pl.LazyFrame): Long-format activity diary with columns:
        hh_id, person_id, act_date, act_purpose, act_hour. Must be sorted.

    Output:
      - (pl.LazyFrame): One row per contiguous activity span with columns:
        hh_id, person_id, act_date, act_purpose, start_time, end_time, n_hours.
        Sorted by hh_id, person_id, act_date, start_time.
    """
    # Find the hour of the previous activity with the same purpose that day
    # Shift by 1 within the (household, person, date, purpose) window to get the preceding hour
    activities = activity_df.with_columns(
        prev_hour_same_purpose=pl.col("act_hour").shift(1).over("hh_id", "person_id", "act_date", "act_purpose")
    )

    # Compute the hour gap between the current activity and the previous activity with the same purpose
    # A gap of 1 hour = consecutive; a gap > 1 hour = new span
    activities = activities.with_columns(
        hour_gap=pl
        .when(pl.col("prev_hour_same_purpose").is_not_null())
        .then(
            (
                # Compute the difference in hours between current and previous occurrence
                pl.col("act_hour").cast(pl.Duration("ms")) - pl.col("prev_hour_same_purpose").cast(pl.Duration("ms"))
            ).dt.total_hours()
        )
        .otherwise(None)  # No previous occurrence → null gap (will be treated as new span)
    )

    # Indicate if there is a change in activity (hour gap larger than 1) and identify the spans
    # is_new_span=True marks the start of a new contiguous block
    activities = activities.with_columns(
        is_new_span=pl.col("hour_gap").is_null() | (pl.col("hour_gap") > 1)
    # span_id: cumulative sum of is_new_span within each (person, date, purpose) group
    # Each time is_new_span is True, span_id increments → unique ID per span
    ).with_columns(span_id=pl.col("is_new_span").cum_sum().over("hh_id", "person_id", "act_date", "act_purpose"))

    # Group by span and find the start, end times and duration
    activities = (
        activities
        .group_by("hh_id", "person_id", "act_date", "act_purpose", "span_id")
        .agg(
            start_time=pl.col("act_hour").min(),      # First hour in the span
            end_time=pl.col("act_hour").max(),         # Last hour in the span
            n_hours=pl.col("act_hour").n_unique(),     # Total number of hours in the span
        )
        .drop("span_id")  # span_id is a helper; not needed in output
    )

    return activities.sort("hh_id", "person_id", "act_date", "start_time")


# =========================================
# Users
# =========================================


def build_toronto_users(
    inputs: TorontoInputs, locations_gdf: gpd.GeoDataFrame, user_journeys_df: pl.DataFrame
) -> pl.DataFrame:
    """
    Description: Build the user demographics DataFrame for Toronto. Joins person-level
    and household-level attributes, validates home location IDs against the location
    vocabulary, and restricts to users who also have trip records.

    Input:
      - inputs (TorontoInputs): Raw inputs with raw_person_df and raw_household_df.
      - locations_gdf (gpd.GeoDataFrame): Location vocabulary for home CT validation.
      - user_journeys_df (pl.DataFrame): Journey table to get the set of valid user IDs.

    Output:
      - (pl.DataFrame): One row per user with columns: user_id, has_driving_license,
        has_pt_pass, home_loc_id, hh_num_adults, hh_num_children, hh_num_vehicles,
        hh_num_bikes. Unknown home locations are replaced with "NA".
    """
    # Person-level demographics: driving licence and transit pass
    persons = inputs.raw_person_df.select(
        "person_id", "hh_id", has_driving_license="THATS driverslicence", has_pt_pass="THATS transitpass"
    )  # TODO add demographics: HH role, age, gender, education, employment status, student status, driving license, PT pass.

    # Household-level demographics: home census tract and vehicle/bike counts
    hhs = inputs.raw_household_df.select(
        "hh_id",
        home_loc_id="THATS HomeCT",       # Home census tract ID
        hh_num_adults="THATS NumAdults",   # Number of adults in household
        hh_num_children="THATS NumChildren",  # Number of children in household
        hh_num_vehicles="THATS NumVeh",    # Number of motorised vehicles
        hh_num_bikes="THATS NumBike",      # Number of bikes
    )  # TODO add HH demographics: HH size (num adults, num children), HH income, HH location, num vehicles, num bikes.

    # Join person and household tables; drop hh_id (no longer needed)
    demographics = persons.join(hhs, on="hh_id", how="left").drop("hh_id")
    # Get unique user IDs from the trip table (only users who made trips)
    user_ids = user_journeys_df.select("user_id").unique()

    # Replace unknown locations with NA
    # If the recorded home CT doesn't exist in the location vocabulary, replace with "NA"
    locations_ids = locations_gdf["loc_id"]
    demographics = demographics.with_columns(
        home_loc_id=pl.when(pl.col("home_loc_id").is_in(locations_ids)).then("home_loc_id").otherwise(pl.lit(NA))
    ).with_columns(pl.col("home_loc_id").fill_null(NA))  # Also fill any remaining nulls with NA

    # Inner join: only keep users that appear in both demographics and journey table
    return user_ids.join(demographics, left_on="user_id", right_on="person_id")
