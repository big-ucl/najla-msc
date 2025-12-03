from dataclasses import dataclass

import polars as pl
import polars.selectors as cs

from activitygraphs.network import PT_EDGE_LIST_SCHEMA
from activitygraphs.utils import check_schema

ROUTE_TYPE_TO_MODE_MAP = {
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

AGENCY_NAME_MAP = {
    "Schweizerische Bundesbahnen SBB": "SBB",
    "Schweizerische Südostbahn (sob)": "SOB",
    "Transports Publics Genevois": "TPG",
    "BLS AG (bls)": "BLS",
    "Transports publics fribourgeois": "TPF",
    "Transports Publics de l'agglomération d'Annemasse": "TAC",
}

MODE_TO_SHORT_NAME_MAP = {
    "mode_train": "Train : ",
    "mode_tram": "Tram  : ",
    "mode_bus": "Bus  : ",
    "mode_bateau_navette": "Boat : ",
}


SAMPLE_DATE = pl.date(2022, 5, 18)
SAMPLE_WEEKDAY = "wednesday"
SAMPLE_DAY_START = pl.time(6, 0, 0)
SAMPLE_DAY_END = pl.time(21, 0, 0)


@dataclass(frozen=True)
class GTFSInputs:
    stops_df: pl.DataFrame
    stop_times_df: pl.LazyFrame
    trips_df: pl.LazyFrame
    routes_df: pl.DataFrame
    agency_df: pl.DataFrame
    calendar_df: pl.DataFrame
    calendar_dates_df: pl.DataFrame


def build_pt_network_edges(locations_df: pl.DataFrame, gtfs: GTFSInputs) -> pl.DataFrame:
    edge_ids = ["route_id", "orig_loc_id", "dest_loc_id"]

    active_stop_times = filter_active_stop_times(locations_df, gtfs)
    pt_trips = create_pt_trip_df(active_stop_times)
    headways_df = compute_avg_headways(pt_trips, gtfs, edge_ids)

    edge_df = (
        pt_trips.group_by(edge_ids)
        .agg(
            first_departure_time=pl.col("orig_departure_time").min(),
            last_departure_time=pl.col("orig_departure_time").max(),
            avg_dwell_time_min=pl.col("orig_dwell_time_min").mean(),
            avg_travel_time_min=pl.col("travel_time_min").mean(),
        )
        .join(headways_df, on=edge_ids, how="left")
    ).collect()

    edge_df = add_route_attributes_to_edges(edge_df, gtfs)
    edge_df = edge_df.select(PT_EDGE_LIST_SCHEMA.keys())

    return check_schema(edge_df, PT_EDGE_LIST_SCHEMA)


def filter_active_stop_times(locations_df: pl.DataFrame, gtfs: GTFSInputs) -> pl.LazyFrame:
    # Find all trip_ids that connect at least 2 locations in the list of active locations
    active_trip_ids = (
        gtfs.stop_times_df.filter(pl.col("loc_id").is_in(locations_df["loc_id"].implode()))
        .filter((pl.len() >= 2).over("trip_id"))
        .unique("trip_id")
    )

    # Find the corresponding active routes
    active_route_ids = active_trip_ids.join(gtfs.trips_df, on="trip_id").unique("route_id").select("route_id")

    # Find the stop_times of all trips on the corresponding active routes
    trips_on_active_route_df = gtfs.trips_df.select("route_id", "service_id", "trip_id").join(
        active_route_ids, on="route_id"
    )

    active_stop_times = gtfs.stop_times_df.join(trips_on_active_route_df, on="trip_id").with_columns(
        pl.col("stop_sequence").rank("dense").over("trip_id", order_by="stop_sequence")
    )

    return active_stop_times


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


def compute_avg_headways(pt_trips: pl.LazyFrame, gtfs: GTFSInputs, edge_id: list[str]) -> pl.LazyFrame:
    exceptions = gtfs.calendar_dates_df.filter(date=SAMPLE_DATE)
    services_added = exceptions.filter(exception_type=1).select("service_id")
    services_removed = exceptions.filter(exception_type=2).select("service_id")

    sample_services = (
        gtfs.calendar_df.filter(pl.col(SAMPLE_WEEKDAY) == 1)
        .select("service_id")
        .join(services_added, on="service_id", how="left")
        .join(services_removed, on="service_id", how="anti")
    )

    return (
        pt_trips.join(sample_services.lazy(), on="service_id")
        .filter((SAMPLE_DAY_START <= pl.col("orig_departure_time")) & (pl.col("orig_departure_time") <= SAMPLE_DAY_END))
        .group_by(edge_id)
        .agg(
            avg_headway_min=pl.col("orig_departure_time")
            .sort()
            .diff(null_behavior="drop")
            .dt.total_minutes(fractional=True)
            .mean()
        )
    )


def add_route_attributes_to_edges(edge_df: pl.DataFrame, gtfs: GTFSInputs) -> pl.DataFrame:
    agencies = gtfs.agency_df.select("agency_id", "agency_name")

    route_mode = pl.col("route_type").replace_strict(
        ROUTE_TYPE_TO_MODE_MAP, default="mode_unknown", return_dtype=pl.Categorical
    )
    route_mode_desc = pl.col("route_mode").cast(pl.String).replace(MODE_TO_SHORT_NAME_MAP)
    agency_name = pl.lit("  (") + pl.col("agency_name").replace(AGENCY_NAME_MAP) + pl.lit(")")
    route_name = pl.col("route_mode_desc") + pl.col("route_short_name") + agency_name

    active_routes_df = (
        gtfs.routes_df.join(agencies, on="agency_id")
        .with_columns(route_mode=route_mode)
        .with_columns(route_mode_desc=route_mode_desc)
        .select("route_id", "route_mode", route_name=route_name)
    )

    return edge_df.join(active_routes_df, on="route_id")
