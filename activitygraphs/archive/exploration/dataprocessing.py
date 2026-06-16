"""
Module: activitygraphs/archive/exploration/dataprocessing.py

Description:
    Defines data schemas, enumeration types, and the ActivityDataset class used to
    represent cleaned travel-survey data for activity scheduling research.

    This module handles the UK London Travel Demand Survey (LTDS) style data structure,
    providing:
      - Coordinate conversion utilities (British National Grid -> WGS84 lat/lon).
      - IntFlag enumerations for trip Purpose, LandUse, and transport Mode.
        IntFlag lets multiple values be combined with bitwise OR (e.g. a location
        visited for both WORK and SHOPPING has a combined purpose flag).
      - Polars schema constants (HH_PERSON_SCHEMA, TRIP_SCHEMA, LOCATION_SCHEMA)
        that define the expected column names and data types for each table.
      - ActivityDataset: a container that bundles the three tables (household/person
        info, trip records, location metadata) with save/load helpers for Parquet files.
"""

from enum import IntFlag, auto
from pathlib import Path
from typing import Self

import numpy as np
import polars as pl
from pyproj import Transformer

from activitygraphs.utils import check_schema


def bng_to_lat_long(df: pl.DataFrame, eastings_col: str, northings_col: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Description:
        Converts British National Grid (BNG / EPSG:27700) easting/northing coordinates
        stored in a Polars DataFrame into WGS84 latitude and longitude arrays.
        BNG is the standard coordinate system used in UK datasets; WGS84 is the
        familiar GPS lat/lon system.

    Input:
      - df (pl.DataFrame): DataFrame containing the easting and northing columns.
      - eastings_col (str): name of the column that holds BNG easting values (metres east).
      - northings_col (str): name of the column that holds BNG northing values (metres north).

    Output:
      - (tuple[np.ndarray, np.ndarray]): a pair (latitudes, longitudes) as NumPy arrays
            in WGS84 decimal degrees, with one value per row of the input DataFrame.
    """
    # EPSG code for British National Grid (used in UK Ordnance Survey data)
    bng_epsg_code = 27700
    # EPSG code for WGS84 lat/lon (standard GPS coordinate system)
    lat_long_epsg_code = 4326
    # Build a pyproj Transformer object to re-project coordinates between the two CRS
    transformer = Transformer.from_crs(bng_epsg_code, lat_long_epsg_code)

    return transformer.transform(df[eastings_col], df[northings_col])


class _PolarsEnum(IntFlag):
    """
    Description:
        Internal base class that combines Python's IntFlag with helper methods for
        use with Polars DataFrames.

        IntFlag allows enum members to be combined with bitwise operators (e.g.
        Purpose.HOME | Purpose.WORK), which makes it possible to store multiple
        activity purposes for a single location in a single integer column.

        All subclasses (Purpose, LandUse, Mode) inherit these methods so that
        their members can be stored in Polars columns typed as pl.UInt32.
    """

    @classmethod
    def names(cls):
        """
        Description:
            Returns human-readable description strings for every member of this enum.

        Output:
          - (list[str]): list of description strings, one per enum member, in
                definition order. Uses the subclass's description() method.
        """
        return [cls.description(member) for member in cls]

    @classmethod
    def polars_enum(cls):
        """
        Description:
            Returns the Polars data type used to store this enum's values in a DataFrame.
            All enum values are stored as unsigned 32-bit integers (pl.UInt32) so that
            bitwise-OR combinations of multiple flags can be represented.

        Output:
          - (pl.DataType): pl.UInt32, the Polars column type for this enum.
        """
        # UInt32 supports up to 32 individual flag bits, enough for any of our enums
        return pl.UInt32

    @classmethod
    def description(cls, member: Self):
        """
        Description:
            Returns a human-readable string label for a single enum member.
            Subclasses override this to return their own description dictionaries.
            The base implementation just returns the member's name string.

        Input:
          - member (Self): one member of this enum class.

        Output:
          - (str): a readable label for that member (e.g. "Home", "Work - Usual workplace").
        """
        return cls(member).name

    @classmethod
    def _check_description(cls, description: dict[Self, str]) -> dict[Self, str]:
        """
        Description:
            Validates that a description dictionary provided by a subclass covers
            exactly all enum members — no extra, no missing. Raises KeyError if
            any member is absent or if any key is not a valid member.

        Input:
          - description (dict[Self, str]): mapping from each enum member to its
                human-readable label, as defined in the subclass's description().

        Output:
          - (dict[Self, str]): the same dictionary, unchanged, if validation passes.

        Raises:
          - KeyError: if the dictionary does not cover every member exactly once.
        """
        # Symmetric difference: members in the enum but not the dict, or vice versa
        mismatch = set(cls) ^ set(description.keys())
        if mismatch:
            raise KeyError(f"Missing keys in description: {mismatch}")

        return description


class Purpose(_PolarsEnum):
    """
    Description:
        Enumeration of all possible trip purposes recorded in the travel survey.
        Each member is a distinct power-of-two flag so that multiple purposes can
        be combined with bitwise OR into a single UInt32 column (e.g. a node that
        was visited for both SHOPPING_FOOD and PERSONAL_BUSINESS purposes will have
        a combined integer value).

        MISSING and NOT_ASKED handle data-quality cases where purpose was unknown.
    """
    MISSING = auto()          # Survey record has no purpose value at all
    NOT_ASKED = auto()        # Respondent was not asked about purpose
    HOME = auto()             # Trip ends at the respondent's home
    WORK = auto()             # Trip to usual workplace
    WORK_DELIVERY = auto()    # Work-related trip for delivery or loading
    WORK_OTHER = auto()       # Other work-related trip
    ENTERTAINMENT = auto()    # Entertainment or recreation
    SHOPPING_FOOD = auto()    # Food / grocery shopping
    PERSONAL_BUSINESS = auto()# Personal errand or using services
    EDUCATION = auto()        # School, college or university (as pupil)
    HOTEL = auto()            # Hotel or holiday accommodation
    ESCORT_WORK = auto()      # Dropping off / picking up someone for work
    ESCORT_SCHOOL = auto()    # Dropping off / picking up someone for school
    ESCORT_HEALTH = auto()    # Dropping off / picking up someone for health visit
    WORSHIP = auto()          # Religious or worship trip
    OTHER = auto()            # Catch-all for unclassified purposes
    HEALTH = auto()           # Medical or health visit
    ESCORT_OTHER = auto()     # Dropping off / picking up someone elsewhere
    SPORT = auto()            # Participating in sport
    LEISURE = auto()          # Leisure trip for enjoyment
    SOCIAL_VISIT = auto()     # Visiting friends or relatives at home
    SOCIAL_OTHER = auto()     # Other social trip
    SHOPPING_OTHER = auto()   # Non-food shopping

    @classmethod
    def description(cls, member: Self):
        """
        Description:
            Returns the full human-readable survey label for a given Purpose member.

        Input:
          - member (Self): one Purpose enum member.

        Output:
          - (str): the survey question label, e.g. "Work - Usual workplace".
        """
        descriptions = {
            cls.MISSING: "Missing",
            cls.NOT_ASKED: "Not asked",
            cls.HOME: "Home",
            cls.WORK: "Work - Usual workplace",
            cls.WORK_DELIVERY: "Work - Delivery/loading",
            cls.WORK_OTHER: "Work - Other",
            cls.ENTERTAINMENT: "Entertainment/recreation",
            cls.SHOPPING_FOOD: "Shopping - Food",
            cls.PERSONAL_BUSINESS: "Personal business / use services",
            cls.EDUCATION: "Education (as a pupil)",
            cls.HOTEL: "Hotel / holiday home",
            cls.ESCORT_WORK: "Drop off/pick up someone to/from work",
            cls.ESCORT_SCHOOL: "Drop off/pick up someone to/from school",
            cls.ESCORT_HEALTH: "Drop off/pick up someone to/from health visit",
            cls.WORSHIP: "Worship or religious observance",
            cls.OTHER: "Other",
            cls.HEALTH: "Health or medical visit",
            cls.ESCORT_OTHER: "Drop off/pick up someone to/from other place",
            cls.SPORT: "Participate in Sport",
            cls.LEISURE: "Leisure trip - enjoyment",
            cls.SOCIAL_VISIT: "Visit friends/relatives at home",
            cls.SOCIAL_OTHER: "Other Social",
            cls.SHOPPING_OTHER: "Shopping - Other",
        }

        return cls._check_description(descriptions)[member]


class LandUse(_PolarsEnum):
    """
    Description:
        Enumeration of land-use categories associated with trip destinations, as
        recorded in the travel survey.  Like Purpose, members are power-of-two
        flags so that multiple land uses can be OR-combined for a single location.

        MYSTERY represents an unrecognised category present in the raw data.
    """
    MISSING = auto()         # No land-use value recorded
    NOT_ASKED = auto()       # Respondent not asked about land use
    RESIDENTIAL = auto()     # Housing / residential area
    OFFICE = auto()          # Office building or business park
    FACTORY = auto()         # Factory, warehouse, or industrial site
    SCHOOL = auto()          # School, college, or educational facility
    SHOPS = auto()           # Retail shops or shopping area
    PUBLIC_BUILDING = auto() # Library, town hall, or other public building
    OPEN_SPACE = auto()      # Park, green space, or recreational open area
    MYSTERY = auto()         # Unrecognised land-use code in raw data
    WORSHIP = auto()         # Church, mosque, temple or other place of worship
    OTHER = auto()           # Catch-all for unclassified land uses
    HOSPITAL = auto()        # Hospital or large medical facility
    GP = auto()              # GP surgery, dentist, or other health service

    @classmethod
    def description(cls, member: Self):
        """
        Description:
            Returns the human-readable survey label for a given LandUse member.

        Input:
          - member (Self): one LandUse enum member.

        Output:
          - (str): the survey label, e.g. "Residential" or "GP/Dentist/Other health service".
        """
        descriptions = {
            cls.MISSING: "Missing",
            cls.NOT_ASKED: "Not asked",
            cls.RESIDENTIAL: "Residential",
            cls.OFFICE: "Office",
            cls.FACTORY: "Factory/warehouse",
            cls.SCHOOL: "School/College",
            cls.SHOPS: "Shops",
            cls.PUBLIC_BUILDING: "Public Buildings",
            cls.OPEN_SPACE: "Open space",
            cls.MYSTERY: "MYSTERY LAND USE",
            cls.WORSHIP: "Place of worship",
            cls.OTHER: "Other",
            cls.HOSPITAL: "Hospital",
            cls.GP: "GP/Dentist/Other health service",
        }

        return cls._check_description(descriptions[member])


class Mode(_PolarsEnum):
    """
    Description:
        Enumeration of transport modes used to make trips, as recorded in the survey.
        Members are power-of-two IntFlag values so they can be combined if needed,
        though in practice each trip record has a single mode.

        TRAM is defined in the class but not listed in the description dict because
        it does not appear in the LTDS data; its description falls through to the
        base class name() method.
    """
    MISSING = auto()    # Mode not recorded in the survey
    NOT_ASKED = auto()  # Respondent was not asked about mode
    WALK = auto()       # Walking, roller-blades, or micro-scooters
    CYCLE = auto()      # Cycling
    CAR = auto()        # Car driver (private car)
    MOTORCYCLE = auto() # Motorcycle or moped rider
    VAN = auto()        # Small van driver
    LORRY = auto()      # Lorry / heavy goods vehicle driver
    BUS = auto()        # Public bus service
    METRO = auto()      # Metro or light rail (e.g. London Underground, DLR)
    TRAIN = auto()      # National rail service
    TRAM = auto()       # Tram (not present in LTDS but reserved for future use)
    TAXI = auto()       # Taxi or private hire vehicle
    VEH_PASS = auto()   # Passenger in someone else's vehicle
    OTHER = auto()      # Unclassified or other mode

    @classmethod
    def description(cls, member: Self) -> str:
        """
        Description:
            Returns the human-readable survey label for a given Mode member.

        Input:
          - member (Self): one Mode enum member.

        Output:
          - (str): the survey label, e.g. "Car driver" or "Bus".
        """
        descriptions = {
            cls.MISSING: "Missing",
            cls.NOT_ASKED: "Not asked",
            cls.WALK: "Walk (/ roller-blades / scooters)",
            cls.CYCLE: "Cycle",
            cls.CAR: "Car driver",
            cls.MOTORCYCLE: "Motorcycle rider",
            cls.VAN: "Van (small) driver",
            cls.LORRY: "Lorry driver",
            cls.BUS: "Bus",
            cls.METRO: "Metro (/Light rail)",
            cls.TRAIN: "Rail",
            cls.TAXI: "Taxi",
            cls.VEH_PASS: "Vehicle passenger",
            cls.OTHER: "Other",
        }

        return cls._check_description(descriptions)[member]


# ── Schema definitions ────────────────────────────────────────────────────────
# These Polars Schema objects declare the expected column names and dtypes for
# each of the three data tables.  check_schema() uses them to validate DataFrames
# at load/save time, catching data-quality issues early.

# Schema for the household-person table: one row per person, with their home and
# (optional) work location IDs and coordinates. loc_work_loc_id == "-1" means
# the person has no fixed workplace.
HH_PERSON_SCHEMA = pl.Schema({
    "hh_id": pl.String,           # Unique household identifier, links to trips
    "person_id": pl.String,       # Unique person identifier within the household
    "year": pl.Int64,             # Survey year (allows multi-year datasets)
    "loc_work_loc_id": pl.String, # Location ID of usual workplace ("-1" if none)
    "loc_work_lat": pl.Float64,   # WGS84 latitude of workplace
    "loc_work_lon": pl.Float64,   # WGS84 longitude of workplace
    "loc_home_loc_id": pl.String, # Location ID of home address
    "loc_home_lat": pl.Float64,   # WGS84 latitude of home
    "loc_home_lon": pl.Float64,   # WGS84 longitude of home
})

# Schema for the trip (journey leg) table: one row per individual trip leg.
# Purpose is the purpose at the origin, purpose_dest at the destination.
# Times are stored as integers (minutes since midnight, or HHMM format).
TRIP_SCHEMA = pl.Schema({
    "hh_id": pl.String,                      # Household identifier (foreign key to HH_PERSON)
    "person_id": pl.String,                  # Person identifier within household
    "trip_id": pl.String,                    # Unique trip identifier
    "trip_number": pl.Int64,                 # Sequential trip index within the person's day
    "year": pl.Int64,                        # Survey year
    "mode": Mode.polars_enum(),              # Transport mode used (UInt32 Mode flag)
    "duration": pl.Int64,                    # Trip duration in minutes
    "distance": pl.Float64,                  # Trip distance in kilometres (or metres — check source)
    "purpose": Purpose.polars_enum(),        # Activity purpose at the origin location (UInt32 flag)
    "purpose_dest": Purpose.polars_enum(),   # Activity purpose at the destination location (UInt32 flag)
    "land_use": LandUse.polars_enum(),       # Land-use category at the origin (UInt32 flag)
    "start_time": pl.Int64,                  # Departure time (minutes since midnight or HHMM integer)
    "end_time": pl.Int64,                    # Arrival time (same format as start_time)
    "loc_origin_loc_id": pl.String,          # Location ID of the trip origin
    "loc_origin_lat": pl.Float64,            # WGS84 latitude of origin
    "loc_origin_lon": pl.Float64,            # WGS84 longitude of origin
    "loc_dest_loc_id": pl.String,            # Location ID of the trip destination
    "loc_destination_lat": pl.Float64,       # WGS84 latitude of destination
    "loc_destination_lon": pl.Float64,       # WGS84 longitude of destination
})

# Schema for the location metadata table: one row per unique location ID.
# Links raw location IDs to their administrative municipality information.
LOCATION_SCHEMA = pl.Schema({
    "loc_id": pl.String,            # Unique location identifier (e.g. zone or stop ID)
    "municipality_id": pl.String,   # ID of the municipality this location belongs to
    "municipality_name": pl.String, # Human-readable municipality name (e.g. "London Borough of Hackney")
})


class ActivityDataset:
    """
    Description:
        A container that bundles the three cleaned travel-survey tables — household/
        person info, trip records, and location metadata — into a single object.

        On construction each DataFrame is validated against its schema (HH_PERSON_SCHEMA,
        TRIP_SCHEMA, LOCATION_SCHEMA) so type errors are caught immediately rather than
        propagating silently downstream.

        The class also provides save/load methods for Parquet files so that processed
        datasets can be cached to disk and reloaded without re-running data cleaning.

    Attributes:
      - name (str): short dataset identifier (e.g. "ltds_2019"), used as the folder
            name and file prefix when saving.
      - hh_person_df (pl.DataFrame): household/person table (schema: HH_PERSON_SCHEMA).
      - trip_df (pl.DataFrame): trip records table (schema: TRIP_SCHEMA).
      - location_df (pl.DataFrame): location metadata table (schema: LOCATION_SCHEMA).
    """
    name: str             # Dataset identifier string (e.g. "ltds_2019")
    hh_person_df: pl.DataFrame   # Validated household-person table
    trip_df: pl.DataFrame        # Validated trip-records table
    location_df: pl.DataFrame    # Validated location-metadata table

    # Parquet filename suffixes used when saving; the dataset name is prepended
    _HHP_FILENAME = "hh_person_df.parquet"
    _TRIP_FILENAME = "trip_df.parquet"
    _LOCATION_FILNAME = "location_df.parquet"  # Note: intentional typo preserved from original

    def __init__(self, name: str, hh_person_df: pl.DataFrame, trip_df: pl.DataFrame, location_df: pl.DataFrame):
        """
        Description:
            Constructs an ActivityDataset, validating each DataFrame against its
            expected schema. Raises an error immediately if any column is missing
            or has the wrong data type.

        Input:
          - name (str): short identifier for this dataset, used in file naming.
          - hh_person_df (pl.DataFrame): household/person table to validate and store.
          - trip_df (pl.DataFrame): trip-records table to validate and store.
          - location_df (pl.DataFrame): location metadata table to validate and store.

        Output:
          - (ActivityDataset): the new dataset instance with all three validated tables.
        """
        self.name = name  # Store the dataset identifier for use in save/load paths
        # Validate each table against its schema before storing it
        self.hh_person_df = check_schema(hh_person_df, HH_PERSON_SCHEMA)
        self.trip_df = check_schema(trip_df, TRIP_SCHEMA)
        self.location_df = check_schema(location_df, LOCATION_SCHEMA)

    def save(self, path: Path | str, dir_name: str = None) -> Path:
        """
        Description:
            Writes all three DataFrames to Parquet files inside a sub-directory of
            `path`. The sub-directory is named after `dir_name` (or the dataset's
            own name if dir_name is not provided). Each file is prefixed with the
            dataset name (e.g. "ltds_2019_trip_df.parquet").

        Input:
          - path (Path | str): parent directory under which the dataset folder will
                be created. Created automatically if it does not exist.
          - dir_name (str | None): name for the sub-directory. Defaults to self.name.

        Output:
          - (Path): path to the sub-directory that was created/used for saving.
        """
        # Accept either a Path object or a plain string for convenience
        path = path if isinstance(path, Path) else Path(path)
        # Use the dataset's own name as the directory name if none is given
        dir_name = dir_name if dir_name is not None else self.name

        # Create the target directory (including any missing parent directories)
        dataset_dir = path / dir_name
        dataset_dir.mkdir(exist_ok=True, parents=True)

        # Build full file paths by prepending the dataset name to each filename
        hh_path = dataset_dir / self._add_file_prefix(self.name, self._HHP_FILENAME)
        trip_path = dataset_dir / self._add_file_prefix(self.name, self._TRIP_FILENAME)
        location_path = dataset_dir / self._add_file_prefix(self.name, self._LOCATION_FILNAME)

        # Write each DataFrame to its Parquet file
        self.hh_person_df.write_parquet(hh_path)
        self.trip_df.write_parquet(trip_path)
        self.location_df.write_parquet(location_path)

        return dataset_dir

    @classmethod
    def load(cls, path: Path | str, dir_name: str, name: str = None) -> Self:
        """
        Description:
            Reads a previously saved ActivityDataset from disk by loading the three
            Parquet files in the expected sub-directory and re-validating their schemas.

        Input:
          - path (Path | str): parent directory containing the dataset sub-directory.
          - dir_name (str): name of the sub-directory created by save().
          - name (str | None): dataset name to use for file prefixes. If None,
                dir_name is used as the name.

        Output:
          - (ActivityDataset): a reconstructed ActivityDataset with all three tables.

        Raises:
          - FileNotFoundError: if any expected Parquet file is missing.
          - ValueError: if any loaded DataFrame fails schema validation.
        """
        path = path if isinstance(path, Path) else Path(path)
        # Use dir_name as the name prefix if no separate name is provided
        name = dir_name if name is None else dir_name
        dataset_dir = path / dir_name  # Full path to the saved dataset directory

        # Reconstruct the file paths using the same naming convention as save()
        hh_path = dataset_dir / cls._add_file_prefix(name, cls._HHP_FILENAME)
        trip_path = dataset_dir / cls._add_file_prefix(name, cls._TRIP_FILENAME)
        location_path = dataset_dir / cls._add_file_prefix(name, cls._LOCATION_FILNAME)

        # Load and validate each table
        hh_person_df = check_schema(pl.read_parquet(hh_path), HH_PERSON_SCHEMA)
        trip_df = check_schema(pl.read_parquet(trip_path), TRIP_SCHEMA)
        location_df = check_schema(pl.read_parquet(location_path), LOCATION_SCHEMA)

        return ActivityDataset(name, hh_person_df, trip_df, location_df)

    @classmethod
    def exists_on_disk(cls, path: Path | str, dir_name: str, name: str = None) -> bool:
        """
        Description:
            Checks whether a previously saved ActivityDataset exists at the expected
            disk location without actually loading it. Useful for deciding whether to
            rebuild the dataset from raw files or load the cached version.

        Input:
          - path (Path | str): parent directory to check inside.
          - dir_name (str): sub-directory name that would contain the dataset files.
          - name (str | None): dataset name used for file prefixes. Defaults to dir_name.

        Output:
          - (bool): True if the sub-directory and all three Parquet files exist,
                False otherwise.
        """
        path = path if isinstance(path, Path) else Path(path)
        name = dir_name if name is None else dir_name  # Resolve name prefix
        dataset_dir = path / dir_name  # Expected directory path

        # Build the expected file paths using the same naming convention as save()
        hh_path = dataset_dir / cls._add_file_prefix(name, cls._HHP_FILENAME)
        trip_path = dataset_dir / cls._add_file_prefix(name, cls._TRIP_FILENAME)
        location_path = dataset_dir / cls._add_file_prefix(name, cls._LOCATION_FILNAME)

        # All four paths (directory + 3 files) must exist for the dataset to be usable
        return dataset_dir.exists() and hh_path.exists() and trip_path.exists() and location_path.exists()

    @classmethod
    def _add_file_prefix(cls, name: str, filename: str):
        """
        Description:
            Prepends the dataset name and an underscore to a base filename. If the
            name is an empty string, the filename is returned unchanged.
            Example: _add_file_prefix("ltds_2019", "trip_df.parquet")
                     -> "ltds_2019_trip_df.parquet"

        Input:
          - name (str): dataset name to use as prefix (can be empty string).
          - filename (str): base filename to prepend the name to.

        Output:
          - (str): the prefixed filename string, or the original filename if name is empty.
        """
        # Only add the underscore separator when name is non-empty
        name = name if not name else name + "_"
        return name + filename
