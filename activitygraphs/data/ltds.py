from pathlib import Path

import exploration.dataprocessing as dp
import polars as pl
from config import DataConfig

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
LTDS_PURPOSES = {k: v for k, v in zip(_LTDS_PURPOSES_KEYS, dp.Purpose)}

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
LTDS_LAND_USES = {k: v for k, v in zip(_LTDS_LAND_USES_KEYS, dp.LandUse)}

LTDS_MODES = {
    "-2": dp.Mode.MISSING,
    "-1": dp.Mode.NOT_ASKED,
    "1": dp.Mode.WALK,
    "2": dp.Mode.CYCLE,
    "3": dp.Mode.CAR,
    "4": dp.Mode.VEH_PASS,
    "5": dp.Mode.MOTORCYCLE,
    "6": dp.Mode.VEH_PASS,
    "9": dp.Mode.VAN,
    "10": dp.Mode.VEH_PASS,
    "11": dp.Mode.LORRY,
    "12": dp.Mode.VEH_PASS,
    "13": dp.Mode.BUS,
    "14": dp.Mode.BUS,
    "15": dp.Mode.BUS,
    "16": dp.Mode.BUS,
    "17": dp.Mode.METRO,
    "18": dp.Mode.METRO,
    "19": dp.Mode.TRAIN,
    "20": dp.Mode.TRAIN,
    "21": dp.Mode.TAXI,
    "22": dp.Mode.TAXI,
    "23": dp.Mode.OTHER,
    "24": dp.Mode.TRAIN,
}

SURVEY_START_YEAR = 2000

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
    project_root = project_root if project_root is not None else Path(".")

    data_path = project_root / data_cfg.paths.raw

    raw_household_df = dp.read_from_parquet(
        data_path / data_cfg.files.raw_household,
        schema={
            "hhid": pl.String,
        },
    )

    raw_person_df = dp.read_from_parquet(
        data_path / data_cfg.files.raw_person,
        schema={
            "phid": pl.String,
            "ppid": pl.String,
        },
    )

    raw_trip_df = dp.read_from_parquet(
        data_path / data_cfg.files.raw_trip,
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
    invalid_values = {"-1", "-2"} if dtype == "str" else {-1, -2}
    return ~pl.col(colname).is_in(invalid_values)


def propagate_invalid_entries(
    input_expr: pl.Expr,
    transformed_expr: pl.Expr,
    error_value=None,
    dtype="str",
) -> pl.Expr:
    return pl.when(is_ltds_entry_valid(input_expr, dtype=dtype)).then(transformed_expr).otherwise(error_value)


def combine_postcode_col(pc_out: str, pc_in: str) -> pl.Expr:
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
    lats, lons = dp.bng_to_lat_long(df, east, north)

    df_lat_lon = df.with_columns(
        propagate_invalid_entries(east, lats, dtype="int").alias(lat),
        propagate_invalid_entries(north, lons, dtype="int").alias(lon),
    )

    return df_lat_lon.drop([east, north]) if remove_cols else df_lat_lon


def _create_hh_person_df(raw_person_df: pl.DataFrame, raw_household_df: pl.DataFrame) -> pl.DataFrame:
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
    trip_df = raw_trip_df.select(
        hh_id="thid",
        person_id="tpid",
        trip_id="ttid",
        trip_number="tseqno",
        year=pl.col("tyearid") + SURVEY_START_YEAR,
        mode=pl.col("tdbmmode").replace(LTDS_MODES).cast(dp.Mode.polars_enum()),
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
        pl.concat([hh_locs, person_locs, trip_locs])
        .unique()
        .filter(is_ltds_entry_valid("loc_id") & is_ltds_entry_valid("ltds_muni_id", dtype="int"))
        .with_columns(
            pl.col("ltds_muni_id").cast(pl.Utf8).replace(LTDS_MUNICIPALITY_IDS).alias("municipality_id"),
            pl.col("ltds_muni_id").cast(pl.Utf8).replace(LTDS_MUNICIPALITY_NAMES).alias("municipality_name"),
        )
        .drop("ltds_muni_id")
    )


def read_and_process_ltds(cfg: DataConfig) -> dp.ActivityDataset:
    raw_household_df, raw_person_df, raw_trip_df = _read_raw_data(cfg)

    hh_person_df = _create_hh_person_df(raw_person_df, raw_household_df)
    trip_df = _create_trip_df(raw_trip_df)
    location_df = _create_location_df(raw_person_df, raw_household_df, raw_trip_df)

    return dp.ActivityDataset(cfg.name, hh_person_df, trip_df, location_df)
