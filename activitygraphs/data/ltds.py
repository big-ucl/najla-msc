from pathlib import Path

import dataprocessing as dp
import polars as pl
from config import LTDSConfig

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


def _read_raw_data(
    cfg: LTDSConfig, project_root=None
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    project_root = project_root if project_root is not None else Path(".")

    data_path = project_root / cfg.paths.data_raw_ltds

    raw_household_df = dp.read_from_parquet(
        data_path / cfg.files.raw_household,
        schema={
            "hhid": pl.String,
        },
    )

    raw_person_df = dp.read_from_parquet(
        data_path / cfg.files.raw_person,
        schema={
            "phid": pl.String,
            "ppid": pl.String,
        },
    )

    raw_trip_df = dp.read_from_parquet(
        data_path / cfg.files.raw_trip,
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
    return (
        pl.when(is_ltds_entry_valid(input_expr, dtype=dtype))
        .then(transformed_expr)
        .otherwise(error_value)
    )


def combine_postcode_col(pc_out: str, pc_in: str) -> pl.Expr:
    postcode_non_null = is_ltds_entry_valid(pc_out) & is_ltds_entry_valid(pc_in)
    return (
        pl.when(postcode_non_null)
        .then(pl.col(pc_out) + " " + pl.col(pc_in))
        .otherwise(None)
    )


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


def create_hh_person_df(
    raw_person_df: pl.DataFrame, raw_household_df: pl.DataFrame
) -> pl.DataFrame:
    person_df = raw_person_df.select(
        hh_id="phid",
        person_id="ppid",
        year=pl.col("pyearid") + SURVEY_START_YEAR,
        loc_work_loc_id=combine_postcode_col("pwspcout", "pwspcin"),
        easting="pwsose",
        northing="pwsosn",
    )

    person_df = add_lat_lon_columns(
        person_df, "easting", "northing", "loc_work_lat", "loc_work_lon"
    )

    household_df = raw_household_df.select(
        hh_id="hhid",
        loc_home_loc_id=combine_postcode_col("hhpcout", "hhpcin"),
        year=pl.col("hyearid") + SURVEY_START_YEAR,
        easting="hhose",
        northing="hhosn",
    )

    household_df = add_lat_lon_columns(
        household_df, "easting", "northing", "loc_home_lat", "loc_home_lon"
    )

    hh_person_df = person_df.join(household_df, on=["hh_id", "year"])
    return dp.check_schema(hh_person_df, dp.HH_PERSON_SCHEMA)


def create_trip_df(raw_trip_df: pl.DataFrame) -> pl.DataFrame:
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
        purpose_dest=pl.col("tdpurp")
        .replace(LTDS_PURPOSES)
        .cast(dp.Purpose.polars_enum()),
        loc_origin_loc_id=combine_postcode_col("topcout", "topcin"),
        o_easting="toose",
        o_northing="toosn",
        loc_dest_loc_id=combine_postcode_col("tdpcout", "tdpcin"),
        d_easting="tdose",
        d_northing="tdosn",
        land_use=pl.col("toland")
        .replace(LTDS_LAND_USES)
        .cast(dp.LandUse.polars_enum()),
        start_time="tstime",
        end_time="tetime",
    )

    trip_df = add_lat_lon_columns(
        trip_df, "o_easting", "o_northing", "loc_origin_lat", "loc_origin_lon"
    )

    trip_df = add_lat_lon_columns(
        trip_df,
        "d_easting",
        "d_northing",
        "loc_destination_lat",
        "loc_destination_lon",
    )

    return dp.check_schema(trip_df, dp.TRIP_SCHEMA)


def read_and_parse_ltds(cfg: LTDSConfig, name: str = "LTDS") -> dp.ActivityDataset:
    raw_household_df, raw_person_df, raw_trip_df = _read_raw_data(cfg)

    hh_person_df = create_hh_person_df(raw_person_df, raw_household_df)
    trip_df = create_trip_df(raw_trip_df)

    return dp.ActivityDataset(name, hh_person_df, trip_df)
