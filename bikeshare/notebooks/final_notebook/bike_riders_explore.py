import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import re
    import marimo as mo
    import pandas as pd
    from pathlib import Path
    import polars as pl
    import json
    import geopandas as gpd

    return Path, mo, pd, pl, re


@app.cell
def _(Path):
    #bike_riders_dir = Path("/home/najla/dev/najla-msc/data/raw/THATS/bike_riders")
    bike_riders_dir = Path("/home/najla/dev/najla-msc/bikeshare/data/raw/bike_ridership/ridership")
    bike_riders_dir.exists()
    return (bike_riders_dir,)


@app.cell
def _(bike_riders_dir):
    csv_files = sorted(bike_riders_dir.glob("*.csv"))
    return (csv_files,)


@app.cell
def _(csv_files):
    len(csv_files)
    return


@app.cell
def _(csv_files):
    # Show the file names so we can confirm which years/months are included.
    [file.name for file in csv_files]
    return


@app.function
def clean_column_name(column: str) -> str:
    """
    Standardize column names across years.

    Examples:
    - 'Trip_Id' becomes 'trip_id'
    - 'Trip Id' becomes 'trip_id'
    - 'Start Station Name' becomes 'start_station_name'
    """
    return (
        column.strip()
        .lower()
        .replace(" ", "_")
    )


@app.cell
def _(csv_files, pd, re):
    #output_path = Path("/home/najla/dev/najla-msc/data/raw/THATS/bike_riders/bike_share_ridership.parquet")

    bike_rider_dfs = []

    for file in csv_files:
        try:
            df = pd.read_csv(file, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(file, encoding="cp1252")

        df.columns = [
            re.sub(
                r"_+",
                "_",
                clean_column_name(col)
                .replace("\ufeff", "")
                .replace("ï»¿", "")
                .replace("\xa0", "_")
                .strip()
                .strip("_"),
            )
            for col in df.columns
        ]

        df = df.loc[:, ~df.columns.duplicated()]
        df["source_file"] = file.name

        bike_rider_dfs.append(df)

    bike_riders = pd.concat(
        bike_rider_dfs,
        ignore_index=True,
        sort=False,
    )

    bike_riders = bike_riders[
        bike_riders["end_station_name"].notna()
        & (bike_riders["end_station_name"].str.strip() != "")
        & (bike_riders["end_station_name"].str.strip().str.upper() != "NULL")
    ].copy()

    bike_riders["trip_duration"] = (
        bike_riders["trip_duration"]
        .astype(str)
        .str.replace(",", "", regex=False)
    )

    #bike_riders.to_parquet(output_path, index=False)

    bike_riders.head(2)
    return (bike_riders,)


@app.cell
def _():
    """
    start_time = 2024-01-01 23:58
    end_time   = 2024-01-02 00:10

    there are 144,670 trips where:
    start_time date != end_time date
    """
    return


@app.cell
def _():
    """
    Cleaning steps:
    1. drop values where end_station_name empty
    2. agg on
    o	date of end_time
    o	start_station_id
    o	start_station_name
    o	start_lat
    o	start_lon
    o	start_CTUID
    o	end_station_id
    o	end_station_name
    o	end_lat
    o	end_lon
    o	end_CTUID
    trip_duration_avg, trips_count
    3. join with location

    """
    return


@app.cell
def _(bike_riders, pd):
    # 1. Drop rows where end_station_name is null or empty
    bike_riders_clean = bike_riders[
        bike_riders["end_station_name"].notna()
        & (bike_riders["end_station_name"].str.strip() != "")
    ].copy()

    # 2. Aggregate by end date, start station, and end station
    bike_riders_clean["end_time"] = pd.to_datetime(
        bike_riders_clean["end_time"],
        format="mixed",
        errors="coerce",
    )

    bike_riders_clean["end_date"] = bike_riders_clean["end_time"].dt.date

    bike_riders_agg = (
        bike_riders_clean
        .groupby(
            ["end_date", "start_station_name", "end_station_name"],
            as_index=False,
        )
        .agg(
            trip_duration_avg=("trip_duration", "mean"),
            trips_count=("trip_id", "unique"),
        )
    )

    bike_riders_agg[(bike_riders_agg["start_station_name"].isna())]
    return


@app.cell
def _():
    ##################################################### explore
    return


app._unparsable_cell(
    r"""
    bike_riders[~(bike_riders["end_station_id"].isna() & bike_riders["end_station_name"].isna()) | ].head()
    """,
    name="_"
)


@app.cell
def _(bike_riders):
    bike_riders[["source_file", "end_station_id", "end_station_name"]] \
        .drop_duplicates() \
        .sort_values(["end_station_name", "source_file", "end_station_id"]) \
        .reset_index(drop=True)
    return


@app.cell
def _(bike_riders):
    bike_riders[""]
    return


@app.cell
def _(bike_riders):
    bike_riders.count()
    return


@app.cell
def _(bike_riders):
    # Extract year from the source file name.
    bike_riders["year"] = (
        bike_riders["source_file"]
        .str.extract(r"(2022|2023|2024|2025|2026)")
        .astype("Int64")
    )
    return


@app.cell
def _():
    #######################################################################stats:
    return


@app.cell
def _(bike_riders):
    trip_count_per_year_station = (
        bike_riders
        .groupby(
            ["year", "start_station_id", "start_station_name"],
            dropna=False,
        )["trip_id"]
        .count()
        .to_frame("trip_count")
        .reset_index()
        .sort_values(
            ["year", "trip_count"],
            ascending=[True, False],
        )
    )

    trip_count_per_year_station
    return (trip_count_per_year_station,)


@app.cell
def _(trip_count_per_year_station):
    trip_count_per_year_station.sort_values(["trip_count"],ascending=[False]).head(20)
    return


@app.cell
def _(bike_riders):
    start_stations = (
        bike_riders[
            ["start_station_name"]
        ]
        .drop_duplicates()
        .sort_values("start_station_name")
        .reset_index(drop=True)
    )

    start_stations #1306
    return


@app.cell
def _(bike_riders):
    end_stations = (
        bike_riders[
            ["end_station_name"]
        ]
        .drop_duplicates()
        .sort_values("end_station_name")
        .reset_index(drop=True)
    )

    end_stations #1306
    return


@app.cell
def _(bike_riders):
    search_text = "Dufferin St / Finch Hydro Recreational Trail"
    matching_trips = bike_riders[bike_riders["start_station_name"].str.contains(search_text, case=False, na=False) | bike_riders["end_station_name"].str.contains(search_text, case=False, na=False)]
    matching_trips
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # confirm 2025 and 2026
    """)
    return


@app.cell
def _(Path):
    #bike_riders_dir = Path("/home/najla/dev/najla-msc/data/raw/THATS/bike_riders")
    bike_share_ridership = Path("/home/najla/dev/najlamsc/data/raw/THATS/bike_riders/bike_share_ridership.parquet")
    return


@app.cell
def _(Path, pl):
    from datetime import datetime

    bike_share_ridership = Path(
        "/home/najla/dev/najla-msc/bikeshare/data/raw/bike_ridership/ridership/bike_share_ridership.parquet"
    )

    end_time_dt = pl.coalesce(
        pl.col("start_time").str.to_datetime(format="%Y-%m-%d %H:%M:%S", strict=False),
        pl.col("start_time").str.to_datetime(format="%m/%d/%Y %H:%M", strict=False),
    )

    bike_riders_2025 = (
        pl.scan_parquet(bike_share_ridership)
        .with_columns(end_time_dt.alias("end_time_dt"))
        .filter(
            (pl.col("end_time_dt") >= datetime(2026, 3, 1)) &
            (pl.col("end_time_dt") < datetime(2027, 1, 1))
        )
        .collect()
    )

    bike_riders_2025.head(2)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
