"""
London Travel Demand Survey (LTDS) data loader and encoder mappings.

This module provides the lookup tables that translate numeric LTDS survey codes
into the internal enum values used throughout the activitygraphs pipeline, and
the functions that read the raw Parquet files and produce cleaned
``ActivityDataset`` objects.

Key outputs:
  - ``LTDS_PURPOSES``     : dict mapping LTDS trip-purpose codes -> Purpose enum
  - ``LTDS_LAND_USES``    : dict mapping LTDS land-use codes -> LandUse enum
  - ``LTDS_MODES``        : dict mapping LTDS mode codes -> Mode enum
  - ``LTDS_MUNICIPALITY_IDS``   : dict mapping numeric LTDS borough IDs -> ONS codes
  - ``LTDS_MUNICIPALITY_NAMES`` : dict mapping numeric LTDS borough IDs -> borough names
  - ``read_and_process_ltds``   : top-level entry point that returns an ActivityDataset
"""

from pathlib import Path

from archive import exploration as dp
import polars as pl
from config import DataConfig

import activitygraphs.base

# List of raw LTDS purpose code strings, including special sentinel values:
#   "-2" = not applicable / data error
#   "-1" = not asked / question skipped
#   "1" through "21" = actual activity purpose categories
_LTDS_PURPOSES_KEYS = [
    "-2",
    "-1",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "19",
    "20",
    "21",
]
# Map each raw LTDS code string to its corresponding dp.Purpose enum member.
# The zip pairs the ordered key list with the ordered Purpose enum values.
LTDS_PURPOSES = {k: v for k, v in zip(_LTDS_PURPOSES_KEYS, dp.Purpose)}

# Same pattern for land-use codes: "-2"/"-1" are sentinels; "1"..."12" are real categories.
_LTDS_LAND_USES_KEYS = [
    "-2",
    "-1",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "11",
    "12",
]
# Map each raw LTDS land-use code string to its corresponding dp.LandUse enum member.
LTDS_LAND_USES = {k: v for k, v in zip(_LTDS_LAND_USES_KEYS, dp.LandUse)}

# Map raw LTDS transport mode codes to Mode enum members.
# Multiple LTDS codes collapse to the same Mode (e.g. codes 4, 6, 10, 12 are all VEH_PASS).
LTDS_MODES = {
    "-2": activitygraphs.mode.Mode.MISSING,
    "-1": activitygraphs.mode.Mode.NOT_ASKED,
    "1": activitygraphs.mode.Mode.WALK,
    "2": activitygraphs.mode.Mode.CYCLE,
    "3": activitygraphs.mode.Mode.CAR,
    "4": activitygraphs.mode.Mode.VEH_PASS,
    "5": activitygraphs.mode.Mode.MOTORCYCLE,
    "6": activitygraphs.mode.Mode.VEH_PASS,
    "9": activitygraphs.mode.Mode.VAN,
    "10": activitygraphs.mode.Mode.VEH_PASS,
    "11": activitygraphs.mode.Mode.LORRY,
    "12": activitygraphs.mode.Mode.VEH_PASS,
    "13": activitygraphs.mode.Mode.BUS,
    "14": activitygraphs.mode.Mode.BUS,
    "15": activitygraphs.mode.Mode.BUS,
    "16": activitygraphs.mode.Mode.BUS,
    "17": activitygraphs.mode.Mode.METRO,
    "18": activitygraphs.mode.Mode.METRO,
    "19": activitygraphs.mode.Mode.TRAIN,
    "20": activitygraphs.mode.Mode.TRAIN,
    "21": activitygraphs.mode.Mode.TAXI,
    "22": activitygraphs.mode.Mode.TAXI,
    "23": activitygraphs.mode.Mode.OTHER,
    "24": activitygraphs.mode.Mode.TRAIN,
}

# The LTDS year column stores "years since survey start" rather than calendar years.
# Adding this constant converts the relative year to a real calendar year.
SURVEY_START_YEAR = 2000

# Maps LTDS internal integer borough ID (1–52) to its ONS GSS code string.
# Values 51 and 52 are special categories for "outside Greater London".
LTDS_MUNICIPALITY_IDS = {
    1: "E09000007",
    2: "E09000001",
    3: "E09000012",
    4: "E09000013",
    5: "E09000014",
    6: "E09000019",
    7: "E09000020",
    8: "E09000022",
    9: "E09000023",
    10: "E09000025",
    11: "E09000028",
    12: "E09000030",
    13: "E09000032",
    14: "E09000033",
    15: "E09000002",
    16: "E09000003",
    17: "E09000004",
    18: "E09000005",
    19: "E09000006",
    20: "E09000008",
    21: "E09000009",
    22: "E09000010",
    23: "E09000011",
    24: "E09000015",
    25: "E09000016",
    26: "E09000017",
    27: "E09000018",
    28: "E09000021",
    29: "E09000024",
    30: "E09000026",
    31: "E09000027",
    32: "E09000029",
    33: "E09000031",
    34: "E07000107",
    35: "E07000207",
    36: "E07000072",
    37: "E07000208",
    38: "E07000098",
    39: "E07000210",
    40: "E07000211",
    41: "E07000212",
    42: "E07000111",
    43: "E06000060",
    44: "E07000213",
    45: "E07000240",
    46: "E07000215",
    47: "E07000102",
    48: "E06000034",
    49: "E07000103",
    50: "E07000217",
    51: "Outside (close)",
    52: "Outside (far)",
}

# Maps the same LTDS integer borough ID to the human-readable borough name string.
# Kept separate from LTDS_MUNICIPALITY_IDS so each can be used independently.
LTDS_MUNICIPALITY_NAMES = {
    1: "Camden",
    2: "City of London",
    3: "Hackney",
    4: "Hammersmith & Fulham",
    5: "Haringey",
    6: "Islington",
    7: "Kensington and Chelsea",
    8: "Lambeth",
    9: "Lewisham",
    10: "Newham",
    11: "Southwark",
    12: "Tower Hamlets",
    13: "Wandsworth",
    14: "Westminster",
    15: "Barking and Dagenham",
    16: "Barnet",
    17: "Bexley",
    18: "Brent",
    19: "Bromley",
    20: "Croydon",
    21: "Ealing",
    22: "Enfield",
    23: "Greenwich",
    24: "Harrow",
    25: "Havering",
    26: "Hillingdon",
    27: "Hounslow",
    28: "Kingston upon Thames",
    29: "Merton",
    30: "Redbridge",
    31: "Richmond upon Thames",
    32: "Sutton",
    33: "Waltham Forest",
    34: "Dartford",
    35: "Elmbridge",
    36: "Epping Forest",
    37: "Epsom and Ewell",
    38: "Hertsmere",
    39: "Mole Valley",
    40: "Reigate and Banstead",
    41: "Runnymede",
    42: "Sevenoaks",
    43: "South Bucks",
    44: "Spelthorne",
    45: "St Albans",
    46: "Tandridge",
    47: "Three Rivers",
    48: "Thurrock",
    49: "Watford",
    50: "Woking",
    51: "Outside Greater London (close)",
    52: "Outside Greater London (far)",
}


def _read_raw_data(data_cfg: DataConfig, project_root=None) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Description: Read the three raw LTDS Parquet files (household, person, trip) from disk
    and return them as Polars DataFrames with enforced string schemas for the ID columns.
    The data files are located under ``<project_root>/<data_cfg.paths.raw>/``.

    Input:
      - data_cfg (DataConfig): Configuration object whose ``.paths.raw`` attribute gives the
        relative path to the raw data directory, and whose ``.inputs.*`` attributes give the
        individual Parquet file names.
      - project_root (Path | None): Root of the project tree. Defaults to the current working
        directory (``Path(".")``) if not provided.

    Output:
      - (tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]):
          1. raw_household_df : one row per household; IDs cast to String.
          2. raw_person_df    : one row per person; IDs cast to String.
          3. raw_trip_df      : one row per trip; key columns cast to String.
    """
    project_root = project_root if project_root is not None else Path(".")

    data_path = project_root / data_cfg.paths.raw

    raw_household_df = dp.read_from_parquet(
        data_path / data_cfg.inputs.raw_household,
        schema={
            "hhid": pl.String,
        },
    )

    raw_person_df = dp.read_from_parquet(
        data_path / data_cfg.inputs.raw_person,
        schema={
            "phid": pl.String,
            "ppid": pl.String,
        },
    )

    raw_trip_df = dp.read_from_parquet(
        data_path / data_cfg.inputs.raw_trip,
        schema={
            "thid": pl.String,
            "tpid": pl.String,
            "ttid": pl.String,
            "topurpi": pl.String,
            "tdpurp": pl.String,
            "toland": pl.String,
            "tdland": pl.String,
            "tdbmmode": pl.String,
        },
    )

    return raw_household_df, raw_person_df, raw_trip_df


def is_ltds_entry_valid(colname: str, dtype="str") -> pl.Expr:
    """
    Description: Build a Polars boolean expression that is ``True`` when a column does NOT
    contain an LTDS sentinel "missing" value. LTDS uses ``-1`` ("not asked") and ``-2``
    ("not applicable / data error") as special codes rather than proper nulls.

    Input:
      - colname (str): Name of the column to test. Can be an actual column name or a
        Polars expression that resolves to a series.
      - dtype (str): ``"str"`` (default) treats sentinels as string literals ``"-1"`` /
        ``"-2"``; any other value uses integer literals ``-1`` / ``-2``.

    Output:
      - (pl.Expr): A boolean Polars expression. ``True`` = the entry is a real value;
        ``False`` = the entry is a sentinel code that should be ignored or set to null.
    """
    invalid_values = {"-1", "-2"} if dtype == "str" else {-1, -2}
    return ~pl.col(colname).is_in(invalid_values)


def propagate_invalid_entries(
    input_expr: pl.Expr,
    transformed_expr: pl.Expr,
    error_value=None,
    dtype="str",
) -> pl.Expr:
    """
    Description: Return a conditional Polars expression that applies ``transformed_expr``
    only when the source column (``input_expr``) contains a valid LTDS value, and falls
    back to ``error_value`` when the source contains a sentinel code (``-1`` or ``-2``).

    This prevents transformations (e.g. BNG-to-lat/lon conversion) from being applied to
    meaningless sentinel values, which would produce nonsensical coordinates.

    Input:
      - input_expr (pl.Expr): Expression identifying the raw LTDS column to validate.
        Passed directly to ``is_ltds_entry_valid``.
      - transformed_expr (pl.Expr): The expression to evaluate when the value is valid
        (e.g. the converted/renamed column value).
      - error_value: The fallback value to use for invalid entries. Defaults to ``None``
        (produces a Polars null).
      - dtype (str): Type hint forwarded to ``is_ltds_entry_valid``; ``"str"`` (default)
        or any other value for integer comparisons.

    Output:
      - (pl.Expr): A ``pl.when(...).then(...).otherwise(...)`` expression that propagates
        nulls for sentinel-coded rows.
    """
    return pl.when(is_ltds_entry_valid(input_expr, dtype=dtype)).then(transformed_expr).otherwise(error_value)


def combine_postcode_col(pc_out: str, pc_in: str) -> pl.Expr:
    """
    Description: Build a Polars expression that concatenates the LTDS outward postcode
    column and inward postcode column into a single full UK postcode string (e.g. ``"SW1A 1AA"``),
    or returns ``None`` when either part is a sentinel value (``-1`` / ``-2``).

    UK postcodes in LTDS are stored as two separate columns:
      - outward code (the part before the space, e.g. ``"SW1A"``)
      - inward code  (the part after the space, e.g. ``"1AA"``)

    Input:
      - pc_out (str): Name of the column containing the outward postcode component.
      - pc_in (str): Name of the column containing the inward postcode component.

    Output:
      - (pl.Expr): A string expression yielding ``"<pc_out> <pc_in>"`` for valid rows
        and ``null`` for rows where either component is a sentinel code.
    """
    postcode_non_null = is_ltds_entry_valid(pc_out) & is_ltds_entry_valid(pc_in)
    return pl.when(postcode_non_null).then(pl.col(pc_out) + " " + pl.col(pc_in)).otherwise(None)


def add_lat_lon_columns(
    df: pl.DataFrame,
    east: str,
    north: str,
    lat: str,
    lon: str,
    remove_cols=True,
):
    """
    Description: Convert British National Grid (BNG) easting/northing coordinate columns
    into WGS-84 latitude/longitude columns and append them to the DataFrame. Invalid
    sentinel values in the source columns are propagated as null rather than producing
    out-of-range coordinates.

    Input:
      - df (pl.DataFrame): The DataFrame containing the BNG coordinate columns.
      - east (str): Name of the column with easting values (integer metres in BNG).
      - north (str): Name of the column with northing values (integer metres in BNG).
      - lat (str): Name of the new latitude column to add (WGS-84 decimal degrees).
      - lon (str): Name of the new longitude column to add (WGS-84 decimal degrees).
      - remove_cols (bool): If ``True`` (default), drop the original ``east`` and ``north``
        columns from the returned DataFrame to keep it tidy.

    Output:
      - (pl.DataFrame): The input DataFrame with ``lat`` and ``lon`` columns appended
        (and ``east`` / ``north`` removed if ``remove_cols=True``).
    """
    lats, lons = dp.bng_to_lat_long(df, east, north)

    df_lat_lon = df.with_columns(
        propagate_invalid_entries(east, lats, dtype="int").alias(lat),
        propagate_invalid_entries(north, lons, dtype="int").alias(lon),
    )

    return df_lat_lon.drop([east, north]) if remove_cols else df_lat_lon


def _create_hh_person_df(raw_person_df: pl.DataFrame, raw_household_df: pl.DataFrame) -> pl.DataFrame:
    """
    Description: Build the cleaned household-person DataFrame by joining person-level records
    (work location, year) with household-level records (home location, year) and converting
    all BNG coordinates to WGS-84 lat/lon. The result conforms to the ``HH_PERSON_SCHEMA``
    defined in the exploration module.

    Each row in the output represents one person with their associated home and work locations.
    The join uses both ``hh_id`` and ``year`` to handle surveys spanning multiple years.

    Input:
      - raw_person_df (pl.DataFrame): Raw LTDS person table (columns include ``ppid``,
        ``phid``, ``pyearid``, work postcode columns, and BNG work coordinates).
      - raw_household_df (pl.DataFrame): Raw LTDS household table (columns include ``hhid``,
        ``hyearid``, home postcode columns, and BNG home coordinates).

    Output:
      - (pl.DataFrame): Cleaned and schema-validated household-person DataFrame with columns
        for ``hh_id``, ``person_id``, ``year``, home location (postcode + lat/lon),
        and work location (postcode + lat/lon).
    """
    person_df = raw_person_df.select(
        hh_id="phid",
        person_id="ppid",
        year=pl.col("pyearid") + SURVEY_START_YEAR,
        loc_work_loc_id=combine_postcode_col("pwspcout", "pwspcin"),
        easting="pwsose",
        northing="pwsosn",
    )

    person_df = add_lat_lon_columns(person_df, "easting", "northing", "loc_work_lat", "loc_work_lon")

    household_df = raw_household_df.select(
        hh_id="hhid",
        loc_home_loc_id=combine_postcode_col("hhpcout", "hhpcin"),
        year=pl.col("hyearid") + SURVEY_START_YEAR,
        easting="hhose",
        northing="hhosn",
    )

    household_df = add_lat_lon_columns(household_df, "easting", "northing", "loc_home_lat", "loc_home_lon")

    hh_person_df = person_df.join(household_df, on=["hh_id", "year"])
    return dp.check_schema(hh_person_df, dp.HH_PERSON_SCHEMA)


def _create_trip_df(raw_trip_df: pl.DataFrame, filter_null: bool = True) -> pl.DataFrame:
    """
    Description: Build the cleaned trip DataFrame from the raw LTDS trip table. Applies
    the following transformations:
      1. Rename columns to the canonical schema names (e.g. ``ttid`` -> ``trip_id``).
      2. Decode LTDS code columns into enum types (mode, purpose, land_use).
      3. Convert relative survey years to calendar years.
      4. Convert BNG coordinates for both trip origin and destination to WGS-84 lat/lon.
      5. Optionally drop rows with missing location data or negative duration/distance.

    Input:
      - raw_trip_df (pl.DataFrame): Raw LTDS trip table with LTDS-encoded columns.
      - filter_null (bool): If ``True`` (default), remove rows where origin or destination
        postcodes/coordinates are null, and rows with negative distance or duration values.
        Set to ``False`` to keep all rows including those with sentinel-coded locations.

    Output:
      - (pl.DataFrame): Cleaned and schema-validated trip DataFrame conforming to
        ``TRIP_SCHEMA``, with columns for IDs, year, mode, duration, distance, purpose,
        origin/destination location IDs, coordinates, land use, and start/end times.
    """
    trip_df = raw_trip_df.select(
        hh_id="thid",
        person_id="tpid",
        trip_id="ttid",
        trip_number="tseqno",
        year=pl.col("tyearid") + SURVEY_START_YEAR,
        mode=pl.col("tdbmmode").replace(LTDS_MODES).cast(activitygraphs.mode.Mode.polars_enum()),
        duration="tdurn",
        distance="tlenn",
        purpose=pl.col("topurpi").replace(LTDS_PURPOSES).cast(dp.Purpose.polars_enum()),
        purpose_dest=pl.col("tdpurp").replace(LTDS_PURPOSES).cast(dp.Purpose.polars_enum()),
        loc_origin_loc_id=combine_postcode_col("topcout", "topcin"),
        o_easting="toose",
        o_northing="toosn",
        loc_dest_loc_id=combine_postcode_col("tdpcout", "tdpcin"),
        d_easting="tdose",
        d_northing="tdosn",
        land_use=pl.col("toland").replace(LTDS_LAND_USES).cast(dp.LandUse.polars_enum()),
        start_time="tstime",
        end_time="tetime",
    )

    if filter_null:
        trip_df = trip_df.drop_nulls([
            "loc_origin_loc_id",
            "o_easting",
            "o_northing",
            "loc_dest_loc_id",
            "d_easting",
            "d_northing",
        ])

        trip_df = trip_df.filter(pl.col("distance") >= 0)
        trip_df = trip_df.filter(pl.col("duration") >= 0)

    trip_df = add_lat_lon_columns(trip_df, "o_easting", "o_northing", "loc_origin_lat", "loc_origin_lon")

    trip_df = add_lat_lon_columns(
        trip_df,
        "d_easting",
        "d_northing",
        "loc_destination_lat",
        "loc_destination_lon",
    )

    return dp.check_schema(trip_df, dp.TRIP_SCHEMA)


def _create_location_df(
    raw_person_df: pl.DataFrame, raw_household_df: pl.DataFrame, raw_trip_df: pl.DataFrame
) -> pl.DataFrame:
    """
    Description: Build a deduplicated location reference DataFrame by pooling every unique
    postcode-based location ID that appears anywhere in the dataset (household home locations,
    person work locations, trip origins, and trip destinations). Each location is enriched
    with the corresponding ONS GSS code and human-readable borough name.

    Sentinel-coded locations (LTDS codes ``-1`` / ``-2``) and rows with invalid borough IDs
    are excluded from the output.

    Input:
      - raw_person_df (pl.DataFrame): Raw LTDS person table (contains work postcode columns
        and work borough ``pwsaboro``).
      - raw_household_df (pl.DataFrame): Raw LTDS household table (contains home postcode
        columns and home borough ``hhaboro``).
      - raw_trip_df (pl.DataFrame): Raw LTDS trip table (contains origin and destination
        postcode columns and corresponding borough columns).

    Output:
      - (pl.DataFrame): Deduplicated location table with columns:
          - ``loc_id``           : full UK postcode string (e.g. ``"SW1A 1AA"``)
          - ``municipality_id``  : ONS GSS code for the borough (e.g. ``"E09000032"``)
          - ``municipality_name``: human-readable borough name (e.g. ``"Wandsworth"``)
    """
    person_locs = raw_person_df.select(
        combine_postcode_col("pwspcout", "pwspcin").alias("loc_id"), pl.col("pwsaboro").alias("ltds_muni_id")
    ).unique("loc_id")

    hh_locs = raw_household_df.select(
        combine_postcode_col("hhpcout", "hhpcin").alias("loc_id"), pl.col("hhaboro").alias("ltds_muni_id")
    ).unique("loc_id")

    trip_locs = pl.concat([
        raw_trip_df.select(
            combine_postcode_col("topcout", "topcin").alias("loc_id"), pl.col("toaboro").alias("ltds_muni_id")
        ),
        raw_trip_df.select(
            combine_postcode_col("tdpcout", "tdpcin").alias("loc_id"), pl.col("tdaboro").alias("ltds_muni_id")
        ),
    ]).unique()

    return (
        pl
        .concat([hh_locs, person_locs, trip_locs])
        .unique()
        .filter(is_ltds_entry_valid("loc_id") & is_ltds_entry_valid("ltds_muni_id", dtype="int"))
        .with_columns(
            pl.col("ltds_muni_id").cast(pl.Utf8).replace(LTDS_MUNICIPALITY_IDS).alias("municipality_id"),
            pl.col("ltds_muni_id").cast(pl.Utf8).replace(LTDS_MUNICIPALITY_NAMES).alias("municipality_name"),
        )
        .drop("ltds_muni_id")
    )


def read_and_process_ltds(cfg: DataConfig) -> dp.ActivityDataset:
    """
    Description: Top-level entry point for the LTDS data pipeline. Reads the three raw
    Parquet files, applies all cleaning and encoding transformations, and bundles the
    results into an ``ActivityDataset`` object that downstream code can use for graph
    construction and model training.

    This function is the recommended way to load LTDS data from scratch. The returned
    dataset can be serialised with ``dataset.save()`` and reloaded later to avoid
    repeating the (slow) processing step.

    Input:
      - cfg (DataConfig): Configuration object containing:
          - ``cfg.name``          : the dataset name string (used to label the ActivityDataset)
          - ``cfg.paths.raw``     : relative path to the directory with raw Parquet files
          - ``cfg.inputs.*``      : filenames for the household, person, and trip Parquet files

    Output:
      - (dp.ActivityDataset): A fully processed dataset object containing:
          - ``hh_person_df``  : cleaned household-person DataFrame
          - ``trip_df``       : cleaned trip DataFrame
          - ``location_df``   : deduplicated location reference DataFrame
          - ``name``          : dataset name string from ``cfg.name``
    """
    raw_household_df, raw_person_df, raw_trip_df = _read_raw_data(cfg)

    hh_person_df = _create_hh_person_df(raw_person_df, raw_household_df)
    trip_df = _create_trip_df(raw_trip_df)
    location_df = _create_location_df(raw_person_df, raw_household_df, raw_trip_df)

    return dp.ActivityDataset(cfg.name, hh_person_df, trip_df, location_df)
