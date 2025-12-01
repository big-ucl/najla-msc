import marimo

__generated_with = "0.17.7"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    import geopandas as gpd

    import marimo as mo
    import polars as pl
    import polars.selectors as cs
    from pathlib import Path

    from activitygraphs.config import load_config

    project_root = Path(mo.notebook_dir().parent)
    cfg = load_config(project_root)


@app.cell
def _():
    mo.md(r"""
    ## Project setup and data loading
    """)
    return


@app.cell
def _():
    user_trips_path = project_root / cfg.data.paths.raw / cfg.data.files.raw_trips
    gtfs_path = project_root / "data/raw/gtfs/gtfs_2022_switzerland"
    stops_path = gtfs_path / "stops.txt"
    boundaries_path = project_root / "data/raw/boundaries"
    subsectors_path = boundaries_path / "GEO_GIREC-SHP.shp"
    swiss_boundaries_path = boundaries_path / "swissboundaries3d_2025-04_2056_5728.shp"
    postcodes_path = boundaries_path / "ortschaftenverzeichnis_plz_2056.shp/AMTOVZ_SHP_LV95"
    french_path = boundaries_path / "codes_postaux_V5"
    return (
        french_path,
        gtfs_path,
        postcodes_path,
        stops_path,
        subsectors_path,
        user_trips_path,
    )


@app.cell
def _(french_path, postcodes_path, stops_path, subsectors_path):
    stops = pl.read_csv(
        stops_path,
        schema={
            "stop_id": pl.String,
            "stop_name": pl.String,
            "stop_lat": pl.Float32,
            "stop_lon": pl.Float32,
            "location_type": pl.Categorical,
            "parent_station": pl.String,
        },
    )

    subsectors_gdf = gpd.read_file(subsectors_path).to_crs("EPSG:4326")
    postcodes_gdf = gpd.read_file(postcodes_path, layer="AMTOVZ_ZIP").to_crs("EPSG:4326")
    localities_gdf = gpd.read_file(postcodes_path, layer="AMTOVZ_LOCALITY").to_crs("EPSG:4326")
    french_gdf = gpd.read_file(french_path).to_crs("EPSG:4326")
    return french_gdf, localities_gdf, postcodes_gdf, stops, subsectors_gdf


@app.cell
def _(
    french_gdf,
    localities_gdf,
    postcodes_gdf,
    stops,
    subsectors_gdf,
    user_trips_path,
):
    from activitygraphs.network import build_locations, match_loc_ids

    locations_gdf = build_locations(stops, subsectors_gdf, postcodes_gdf, localities_gdf, french_gdf)
    locations_df = pl.DataFrame(locations_gdf.drop(columns=["geometry"]))

    user_trips_df = pl.read_parquet(user_trips_path)
    user_trips_df = match_loc_ids(user_trips_df, locations_df, stops)

    _user_loc_ids = (
        pl.concat([user_trips_df["dep_loc_id"], user_trips_df["arr_loc_id"]]).unique().rename("loc_id").to_pandas()
    )
    locations_gdf = locations_gdf.merge(_user_loc_ids, on="loc_id")
    locations_df = pl.DataFrame(locations_gdf.drop(columns=["geometry"]))
    return (locations_df,)


@app.cell
def _():
    mo.md(r"""
    ## Network building
    """)
    return


@app.cell
def _():
    mo.md(r"""
    1. Filter all active PT stops in the user trips
    2. Find all trips connecting at least two active PT stops
    3.
    """)
    return


@app.cell
def _(gtfs_path, stops):
    # 1. Select trip_ids of trips with stops in the trips list


    def parse_gtfs_date(*cols: str) -> pl.Expr:
        return pl.col(*cols).cast(pl.String).str.to_date("%Y%m%d")


    _stop_id_to_loc_id = stops.select(
        "stop_id", pl.when(pl.col("parent_station") == "").then("stop_id").otherwise("parent_station").alias("loc_id")
    )

    stop_times_df = pl.scan_csv(gtfs_path / "stop_times.txt").join(_stop_id_to_loc_id.lazy(), on="stop_id", how="left")
    trips_df = pl.scan_csv(gtfs_path / "trips.txt")
    routes_df = pl.read_csv(gtfs_path / "routes.txt")
    agency_df = pl.read_csv(gtfs_path / "agency.txt")
    calendar_df = pl.read_csv(gtfs_path / "calendar.txt").with_columns(parse_gtfs_date("start_date", "end_date"))
    calendar_dates_df = pl.read_csv(gtfs_path / "calendar_dates.txt").with_columns(parse_gtfs_date("date"))

    stop_times_df
    return (
        agency_df,
        calendar_dates_df,
        calendar_df,
        routes_df,
        stop_times_df,
        trips_df,
    )


@app.cell
def _():
    return


@app.cell
def _(agency_df, locations_df, routes_df, stop_times_df, trips_df):
    def filter_active_stop_times(
        stop_times_df: pl.LazyFrame, trips_df: pl.LazyFrame, routes_df: pl.DataFrame
    ) -> pl.LazyFrame:
        route_type_to_mode = {
            101: "mode_train",
            101: "mode_train",
            102: "mode_train",
            103: "mode_train",
            106: "mode_train",
            109: "mode_train",
            117: "mode_train",
            700: "mode_bus",
            900: "mode_tram",
            1000: "mode_bateau_navette",
        }

        # Find all trip_ids that connect at least 2 active locations
        active_trip_ids = (
            stop_times_df.filter(pl.col("loc_id").is_in(locations_df["loc_id"].implode()))
            .filter((pl.len() >= 2).over("trip_id"))
            .unique("trip_id")
        )

        # Find the corresponding active routes
        active_route_ids = active_trip_ids.join(trips_df, on="trip_id").unique("route_id").select("route_id")
        active_routes_df = (
            routes_df.lazy()
            .join(active_route_ids, on="route_id")
            .drop("route_long_name", "route_desc")
            .with_columns(pl.col("route_type").replace_strict(route_type_to_mode, return_dtype=pl.Categorical))
            .join(agency_df.lazy().select("agency_id", "agency_name"), on="agency_id")
        )

        # Return the stop_times of all trips on the corresponding active routes
        trips_on_active_route_df = trips_df.select("route_id", "service_id", "trip_id").join(
            active_routes_df.select("route_id"), on="route_id"
        )

        return stop_times_df.join(trips_on_active_route_df, on="trip_id").with_columns(
            pl.col("stop_sequence").rank("dense").over("trip_id", order_by="stop_sequence")
        )


    active_stop_times = filter_active_stop_times(stop_times_df, trips_df, routes_df)
    return (active_stop_times,)


@app.cell
def _(active_stop_times):
    # Join stop_times with itself to form a PT trip table


    def create_pt_trip_df(active_stop_times: pl.LazyFrame) -> pl.LazyFrame:
        join_cols = ["trip_id", "service_id", "route_id"]
        departures = active_stop_times.select(*join_cols, pl.exclude(join_cols).name.prefix("orig_"))
        arrivals = active_stop_times.select(*join_cols, pl.exclude(join_cols).name.prefix("dest_"))

        return (
            departures.join(
                arrivals,
                left_on=[*join_cols, pl.col("orig_stop_sequence")],
                right_on=[*join_cols, pl.col("dest_stop_sequence") - 1],
            )
            .drop("trip_id_right", "service_id_right", "route_id_right")
            .with_columns(cs.ends_with("time").str.to_time("%T", strict=False))
            .drop_nulls()
            .with_columns(
                orig_dwell_time_min=(pl.col("orig_departure_time") - pl.col("orig_arrival_time")).dt.total_minutes(
                    fractional=True
                ),
                travel_time_min=(pl.col("dest_arrival_time") - pl.col("orig_departure_time")).dt.total_minutes(
                    fractional=True
                ),
            )
        )


    pt_trips = create_pt_trip_df(active_stop_times)
    return (pt_trips,)


@app.cell
def _(calendar_dates_df, calendar_df, pt_trips):
    def compute_avg_headways(
        pt_trips: pl.LazyFrame, calendar_df: pl.DataFrame, calendar_dates_df: pl.DataFrame, edge_id: list[str]
    ) -> pl.LazyFrame:
        sample_date = pl.date(2022, 5, 18)
        sample_weekday = "wednesday"

        day_start = pl.time(6, 0, 0)
        day_end = pl.time(21, 0, 0)

        exceptions = calendar_dates_df.filter(date=sample_date)
        services_added = exceptions.filter(exception_type=1).select("service_id")
        services_removed = exceptions.filter(exception_type=2).select("service_id")

        sample_services = (
            calendar_df.filter(pl.col(sample_weekday) == 1)
            .select("service_id")
            .join(services_added, on="service_id", how="left")
            .join(services_removed, on="service_id", how="anti")
        )

        return (
            pt_trips.join(sample_services.lazy(), on="service_id")
            .filter((day_start <= pl.col("orig_departure_time")) & (pl.col("orig_departure_time") <= day_end))
            .group_by(edge_id)
            .agg(
                avg_headway_min=pl.col("orig_departure_time")
                .sort()
                .diff(null_behavior="drop")
                .dt.total_minutes(fractional=True)
                .mean()
            )
        )


    edge_ids = ["route_id", "orig_loc_id", "dest_loc_id"]
    headways_df = compute_avg_headways(pt_trips, calendar_df, calendar_dates_df, edge_ids)
    return edge_ids, headways_df


@app.cell
def _(edge_ids, headways_df, pt_trips):
    edge_df = (
        pt_trips.group_by(edge_ids)
        .agg(
            first_departure_time=pl.col("orig_departure_time").min(),
            last_departure_time=pl.col("orig_departure_time").max(),
            avg_dwell_time_min=pl.col("orig_dwell_time_min").mean(),
            avg_travel_time=pl.col("travel_time_min").mean(),
        )
        .join(headways_df, on=edge_ids, how="left")
    ).collect()

    edge_df
    return


@app.cell
def _():
    mo.md(r"""
    TODO DO THE ROUTES AND THEN HEY PRESTO GRAPH
    """)
    return


if __name__ == "__main__":
    app.run()
