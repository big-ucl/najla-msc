"""
Shared enumerations and data-frame schemas for the activitygraphs package.

This module defines:
  - PTNodeType  : controls whether a public-transport graph uses one node per route
                  or one node per stop.
  - Mode        : categorical string enum for travel modes (walk, bus, train, etc.).
  - Purpose     : categorical string enum for trip purposes (home, work, shop, etc.).
  - Schema dicts/objects for locations, user journeys, users, and network edge lists.
  - CRS         : the coordinate reference system used throughout the project (WGS-84).

All schemas are used as validation contracts via ``activitygraphs.utils.check_schema``.
"""

from enum import StrEnum

import polars as pl


class PTNodeType(StrEnum):
    """
    Description: Controls the granularity of nodes in the public-transport (PT)
    graph representation.  When ONE_PER_ROUTE, each transit route gets a single
    graph node; when ONE_PER_STOP, every individual stop gets its own node.

    Input:
      - (none — this is an enum, no constructor arguments needed)

    Output:
      - (PTNodeType): one of the two string-valued enum members below
    """

    # Each route in the GTFS feed is collapsed into a single graph node.
    ONE_PER_ROUTE = "ONE_PER_ROUTE"
    # Every physical stop in the GTFS feed gets its own graph node.
    ONE_PER_STOP = "ONE_PER_STOP"


class Mode(StrEnum):
    """
    Description: Enumeration of all travel modes that can appear in a journey leg.
    The string values (e.g. "mode_bus") are used directly as categorical column
    values in the USER_JOURNEY_SCHEMA leg_mode column, and as keys in colour maps
    for map visualisation.

    Input:
      - (none — enum, no constructor arguments)

    Output:
      - (Mode): one of the named members, usable as a plain string anywhere a mode
                label is expected
    """

    OTHER = "mode_other"           # Any mode that does not fit another category
    UNKNOWN = "mode_unknown"       # Mode was not recorded or could not be determined
    BOAT = "mode_boat"             # Ferry or other water-based transport
    BUS = "mode_bus"               # Urban or suburban bus service
    COACH = "mode_coach"           # Long-distance coach / intercity bus
    WALK = "mode_walk"             # Walking (on foot)
    CYCLE = "mode_cycle"           # Bicycle (conventional or e-bike)
    MOTORCYCLE = "mode_motorcycle" # Motorbike or moped
    TAXI = "mode_taxi"             # Taxi or ride-hailing service
    TRAIN = "mode_train"           # Heavy rail or suburban train
    TRAMWAY = "mode_tramway"       # Tram or light rail
    VEH_PASS = "mode_vehicle_passenger"  # Passenger in a private car (not driver)
    CAR = "mode_car"               # Driver of a private car


class Purpose(StrEnum):
    """
    Description: Enumeration of trip purposes that describe WHY a traveller made a
    journey.  The string values (e.g. "purp_work") are stored in the dep_purpose and
    arr_purpose columns of USER_JOURNEY_SCHEMA and are used as categorical features
    for model training.

    Input:
      - (none — enum, no constructor arguments)

    Output:
      - (Purpose): one of the named members, usable as a plain string anywhere a
                   purpose label is expected
    """

    OTHER = "purp_other"                          # Any purpose that does not fit another category
    UNKNOWN = "purp_unknown"                      # Purpose was not recorded or could not be determined
    HOME = "purp_home"                            # Returning to or being at the respondent's home
    WORK_MAIN = "purp_work"                       # Main / primary place of work
    WORK_OTHER = "purp_work_other"                # Secondary work location or work-related errand
    STUDY = "purp_study"                          # School, university, or other educational institution
    VISIT = "purp_visit"                          # Visiting friends or family
    ESCORT = "purp_escort"                        # Escorting another person (e.g. dropping a child at school)
    PERSONAL = "purp_personal"                    # Personal business (doctor, bank, admin, etc.)
    SHOP = "purp_shopping"                        # Grocery, retail, or other shopping
    ENTERTAINMENT = "purp_entertainment"          # Cinema, restaurant, sports event, etc.
    LEISURE_OTHER = "purp_leisure_other"          # Other leisure activities not covered above
    LONG_DISTANCE_TRIP = "purp_long_distance_trip"  # Trip covering a long distance (e.g. overnight travel)


# ---------------------------------------------------------------------------
# Locations schema
# ---------------------------------------------------------------------------
# LOCATIONS_SCHEMA defines the expected column names and dtypes for any
# GeoDataFrame that holds the spatial locations (stops, zones, etc.) used in
# the transport network.  The "object" dtype is GeoPandas/Pandas terminology
# for a string/mixed column.
LOCATIONS_SCHEMA = {
    "loc_id": "object",    # Unique string identifier for the location (e.g. "stop_123")
    "loc_name": "object",  # Human-readable name of the location (e.g. "Gare Cornavin")
    "type": "object",      # Category string: "subsector", "public_transport", etc.
    "lon": "float64",      # Longitude in decimal degrees (WGS-84)
    "lat": "float64",      # Latitude in decimal degrees (WGS-84)
    "geometry": "geometry",  # Shapely geometry object (point or polygon)
}

# Convenience list of column names derived directly from LOCATIONS_SCHEMA.
# Used when code needs to select or reorder columns to match the schema.
LOCATIONS_COLUMNS = list(LOCATIONS_SCHEMA.keys())

# ---------------------------------------------------------------------------
# User journey schema
# ---------------------------------------------------------------------------
# USER_JOURNEY_SCHEMA is a Polars Schema (ordered dict of column -> dtype) that
# describes a single journey LEG made by a survey respondent.  One journey
# (journey_id) can contain multiple sequential legs (leg_id = 0, 1, 2, ...).
USER_JOURNEY_SCHEMA = pl.Schema({
    "user_id": pl.String,          # Unique identifier for the survey respondent
    "journey_id": pl.String,       # Unique identifier for a trip / journey chain
    "leg_id": pl.Int8,             # Sequential index of this leg within the journey (0-indexed)
    "leg_mode": pl.Categorical(),  # Travel mode for this leg; values are Mode enum strings
    "leg_line": pl.String,         # Transit line or route name used for this leg (empty for walk/car)
    "duration": pl.Duration(),     # Travel time for this leg as a Polars Duration
    "dep_day": pl.Date,            # Calendar date of departure for this leg
    "dep_time": pl.Time,           # Clock time of departure (wall-clock, local timezone)
    "dep_purpose": pl.Categorical(),  # Activity purpose at the departure location; Purpose enum string
    "dep_loc_id": pl.String,       # loc_id of the departure location (matches LOCATIONS_SCHEMA)
    "arr_loc_id": pl.String,       # loc_id of the arrival location (matches LOCATIONS_SCHEMA)
    "arr_purpose": pl.Categorical(),  # Activity purpose at the arrival location; Purpose enum string
})

# ---------------------------------------------------------------------------
# User (respondent) schema
# ---------------------------------------------------------------------------
# USER_SCHEMA describes the per-respondent socio-demographic attributes used as
# contextual features when constructing per-person activity graphs.
USER_SCHEMA = pl.Schema({
    "user_id": pl.String,          # Unique identifier for the survey respondent (matches USER_JOURNEY_SCHEMA)
    "home_loc_id": pl.String,      # loc_id of the respondent's home location
    "hh_num_adults": pl.UInt32,    # Number of adults in the respondent's household
    "hh_num_children": pl.UInt32,  # Number of children in the respondent's household
})

# ---------------------------------------------------------------------------
# Generic edge-list schema (walk and link edges share this layout)
# ---------------------------------------------------------------------------
# EDGE_LIST_SCHEMA is the minimal schema for a directed edge in the transport
# network.  Each row represents one directed connection between two locations.
EDGE_LIST_SCHEMA = pl.Schema({
    "orig_loc_id": pl.String,      # loc_id of the origin (tail) of the directed edge
    "dest_loc_id": pl.String,      # loc_id of the destination (head) of the directed edge
    "travel_time_min": pl.Float64, # Expected travel time along this edge in minutes
})

# Walking edges (pedestrian connections between nearby locations) use the
# same three-column schema as the generic edge list.
WALK_EDGE_LIST_SCHEMA = EDGE_LIST_SCHEMA

# Link edges (synthetic connections used to join different network layers,
# e.g. walking from a zone centroid to a transit stop) also share this schema.
LINK_EDGE_LIST_SCHEMA = EDGE_LIST_SCHEMA

# ---------------------------------------------------------------------------
# Public-transport edge schema
# ---------------------------------------------------------------------------
# PT_EDGE_LIST_SCHEMA describes a directed connection between two transit stops
# (or zones) that is operated by a specific transit route.  The extra columns
# beyond the generic EDGE_LIST_SCHEMA carry service-level information derived
# from GTFS data.
PT_EDGE_LIST_SCHEMA = pl.Schema({
    "orig_loc_id": pl.String,              # loc_id of the boarding stop
    "route_id": pl.String,                 # GTFS route_id that operates this connection
    "dest_loc_id": pl.String,              # loc_id of the alighting stop
    "first_departure_time": pl.Time,       # Earliest departure time of this service during the day
    "last_departure_time": pl.Time,        # Latest departure time of this service during the day
    "avg_dwell_time_min": pl.Float64,      # Average time the vehicle spends at the origin stop (minutes)
    "travel_time_min": pl.Float64,         # Average in-vehicle travel time from origin to destination (minutes)
    "daily_trip_count": pl.UInt32,         # Number of individual trips (vehicle runs) on this route per day
    "avg_headway_min": pl.Float64,         # Average time between successive departures (minutes)
    "route_mode": pl.Categorical(),        # Mode of the route; values are Mode enum strings (e.g. "mode_bus")
    "route_name": pl.String,               # Human-readable name of the route (e.g. "Line 15")
})

# ---------------------------------------------------------------------------
# Transfer edge schema
# ---------------------------------------------------------------------------
# TRANSFER_EDGE_LIST_SCHEMA describes a passenger transfer between two transit
# routes, possibly at the same physical location (internal transfer) or at
# different stops (external transfer requiring walking).
TRANSFER_EDGE_LIST_SCHEMA = pl.Schema({
    "orig_loc_id": pl.String,      # loc_id of the stop where the passenger alights from the first route
    "orig_route_id": pl.String,    # GTFS route_id of the first (incoming) route
    "dest_loc_id": pl.String,      # loc_id of the stop where the passenger boards the second route
    "dest_route_id": pl.String,    # GTFS route_id of the second (outgoing) route
    "travel_time_min": pl.Float64, # Time required to complete the transfer (walking + waiting), in minutes
})

# ---------------------------------------------------------------------------
# Global coordinate reference system
# ---------------------------------------------------------------------------
# CRS is the EPSG code string for WGS-84 geographic coordinates (longitude /
# latitude in decimal degrees).  All GeoDataFrames in this project are stored
# in this CRS.  When computing distances or centroids, geometries are first
# re-projected to a local UTM zone and then converted back to WGS-84.
CRS = "EPSG:4326"
