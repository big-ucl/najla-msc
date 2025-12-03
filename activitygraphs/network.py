import folium
import geopandas as gpd
import polars as pl

CRS = "EPSG:4326"
NA_LON, NA_LAT = 6.1709475192397605, 46.24348817355701

LOCATIONS_SCHEMA = {
    "loc_id": "object",
    "loc_name": "object",
    "type": "object",
    "lon": "float64",
    "lat": "float64",
    "geometry": "geometry",
}
LOCATIONS_COLUMNS = LOCATIONS_SCHEMA.keys()

USER_JOURNEY_SCHEMA = pl.Schema({
    "user_id": pl.String,
    "journey_id": pl.String,
    "leg_id": pl.Int8,
    "leg_mode": pl.Categorical(),
    "leg_line": pl.String,
    "dep_day": pl.Date,
    "dep_time": pl.Time,
    "dep_purpose": pl.Categorical(),
    "dep_loc_id": pl.String,
    "arr_loc_id": pl.String,
    "arr_purpose": pl.Categorical(),
})

PT_EDGE_LIST_SCHEMA = pl.Schema({
    "orig_loc_id": pl.String,
    "route_id": pl.String,
    "dest_loc_id": pl.String,
    "first_departure_time": pl.Time,
    "last_departure_time": pl.Time,
    "avg_dwell_time_min": pl.Float64,
    "avg_travel_time_min": pl.Float64,
    "avg_headway_min": pl.Float64,
    "route_mode": pl.Categorical(),
    "route_name": pl.String,
})


def explore_location_affluence(
    trips_df: pl.DataFrame, locations_gdf: gpd.GeoDataFrame, loc_id_column: str
) -> folium.Map:
    stops = locations_gdf.merge(
        trips_df[loc_id_column].value_counts(name="num_visits").to_pandas(),
        left_on="loc_id",
        right_on=loc_id_column,
        how="right",
    )
    stops.set_geometry(gpd.points_from_xy(stops["lon"], stops["lat"], crs="EPSG:4326"), inplace=True)

    return stops.explore(
        column="num_visits",
        cmap="viridis_r",
        tiles="Cartodb Positron",
        scheme="NaturalBreaks",
        k=10,
        tooltip=["loc_name", "num_visits"],
        marker_kwds={"radius": 5},
    )
