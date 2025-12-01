from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import polars.selectors as cs
from rapidfuzz import fuzz, process

from activitygraphs.config import Config

CRS = "EPSG:4326"
CANTONS_LAYER = "swissBOUNDARIES3D_1_5_TLM_KANTONSGEBIET"
MUNICIPALITY_LAYER = ""
NA_LON, NA_LAT = 6.1709475192397605, 46.24348817355701
LOCATIONS_COLUMNS = ["loc_id", "loc_name", "type", "lon", "lat", "geometry"]

LOCATION_REGEXES = {
    "subsector": r"([\s\S]+) - sous_secteur\s*$",
    "municipality_swiss": r"([\s\S]+) - (\d\d\d\d)\s*$",
    "municipality_french": r"([\s\S]+) - (\d\d\d\d\d)\s*$",
    "na": r"NA",
}

STOP_NAME_MAPPING = {
    "Domicile": "NA",
    "Home": "NA",
    "Other": "NA",
    "Cuvat, Les Voisins": "CRUSEILLES - 74350",
    "Genève-Cornavin": "Genève",
    "Lucerne": "Luzern",
    "Saint-Julien, SNCF": "Saint-Julien-en-Genevois, SNCF",
    "Viège": "Visp",
    "Collex-Bossy - 1239": "Collex - 1239",
    "Perly-Certoux - 1258": "Perly - 1258",
    "LÉAZ - 1200": "Léaz, Village",
    "Port Noir/Genève-Plage, lac": "Genève-Port Noir (lac)",
    "Pregny-Chambésy - 1292": "Chambésy - 1292",
    "Pâquis, lac": "Genève-Pâquis (lac)",
    "Lancy - 1212": "Grand-Lancy - 1212",
    "Pringy": "Pringy - 1663",
    "Cluses": "CLUSES - 74300",
    "Signy-Avenex - 1274": "Signy, Le Glassey",
    "De-Chateaubriand, lac": "Genève-De-Châteaubriand (lac)",
    "Eaux-Vives, lac": "Genève-Eaux-Vives (lac)",
    "Saint-Cergue - 1265": "La Cure",
    "Saint-Cergue - 1264": "St-Cergue",
    "Vernier, Etang-Place": "Vernier, Etang Place",
    "Vernier, CHôtelaine": "Vernier, Châtelaine",
}

FUZZY_MATCH_THRESHOLD = 65


def load_files(cfg: Config) -> tuple[pl.DataFrame, gpd.GeoDataFrame]:
    raw_dir = Path(cfg.data.paths.raw).parent  # TODO Fix this in config.yaml
    gtfs_dir = raw_dir / "gtfs" / "gtfs_2022_switzerland"
    boundaries_dir = raw_dir / "boundaries" / "swissboundaries3d_2025-04_2056_5728.shp"

    stops_df = pl.read_csv(
        gtfs_dir / "stops.txt",
        schema={
            "stop_id": pl.String,
            "stop_name": pl.String,
            "stop_lat": pl.Float32,
            "stop_lon": pl.Float32,
            "location_type": pl.Categorical,
            "parent_station": pl.String,
        },
    )

    cantons_gdf = gpd.read_file(boundaries_dir, layer=CANTONS_LAYER).to_crs(CRS)

    return stops_df, cantons_gdf


def build_locations(
    stops: pl.DataFrame,
    subsectors: gpd.GeoDataFrame,
    postcodes: gpd.GeoDataFrame,
    localities: gpd.GeoDataFrame,
    french_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    special_locations = _build_special_locations()
    pt_locations = _build_pt_locations(stops)
    subsector_locations = _build_subsector_locations(subsectors)
    municipality_swiss_locations = _build_municipality_swiss_locations(postcodes, localities)
    municipality_french_locations = _build_municipality_french_locations(french_gdf)

    return gpd.GeoDataFrame(
        pd.concat(
            [
                special_locations,
                pt_locations,
                subsector_locations,
                municipality_swiss_locations,
                municipality_french_locations,
            ],
            ignore_index=True,
        ),
        crs=CRS,
    )


def _build_special_locations() -> gpd.GeoDataFrame:
    na_location = {
        "loc_id": "NA",
        "loc_name": "NA",
        "type": "na",
        "lon": NA_LON,
        "lat": NA_LAT,
    }

    return gpd.GeoDataFrame(
        [na_location],
        geometry=gpd.points_from_xy([NA_LON], [NA_LAT]),
        crs=CRS,
    )


def _build_pt_locations(stops: pl.DataFrame) -> gpd.GeoDataFrame:
    pt_locations = stops.select(
        loc_id="stop_id", loc_name="stop_name", type=pl.lit("public_transport"), lon="stop_lon", lat="stop_lat"
    )

    return gpd.GeoDataFrame(
        pt_locations.to_pandas(), geometry=gpd.points_from_xy(pt_locations["lon"], pt_locations["lat"], crs=CRS)
    )[LOCATIONS_COLUMNS]


def _build_subsector_locations(subsectors_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    subsector_locations = _add_lon_lat_from_centroid(subsectors_gdf, index_col="OBJECTID")

    subsector_locations["loc_id"] = "subsector-" + subsector_locations["OBJECTID"].astype(str)
    subsector_locations["type"] = "subsector"

    subsector_locations = subsector_locations.rename(columns={"NOM": "loc_name"})
    return subsector_locations[LOCATIONS_COLUMNS]


def _build_municipality_swiss_locations(
    postcodes_gdf: gpd.GeoDataFrame, localities_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    postcodes_gdf = postcodes_gdf[["FK_LOCALIT", "ZIP_ID", "ZIP4", "geometry"]]
    postcodes_gdf = _add_lon_lat_from_centroid(postcodes_gdf, index_col="ZIP_ID")
    municipality_locations = localities_gdf.drop(columns=["geometry"]).merge(
        postcodes_gdf, left_on="LOCALITYID", right_on="FK_LOCALIT"
    )

    municipality_locations["loc_name"] = municipality_locations["NAME"] + " - " + municipality_locations["ZIP4"]
    municipality_locations["loc_id"] = (
        "CH-" + municipality_locations["ZIP4"] + "-" + municipality_locations["INDEXNAME"]
    )
    municipality_locations["type"] = "municipality_swiss"

    municipality_locations = municipality_locations.drop_duplicates(subset="loc_id", keep="first").reset_index()
    return municipality_locations[LOCATIONS_COLUMNS]


def _build_municipality_french_locations(french_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    french_locations = _add_lon_lat_from_centroid(french_gdf, index_col="ID")
    french_locations["loc_id"] = "FR-" + french_locations["ID"]
    french_locations["loc_name"] = french_locations["LIB"].str.upper() + " - " + french_locations["ID"]
    french_locations["type"] = "municipality_french"

    return french_locations[LOCATIONS_COLUMNS]


def _add_lon_lat_from_centroid(
    gdf: gpd.GeoDataFrame, index_col: str, lon_name="lon", lat_name="lat"
) -> gpd.GeoDataFrame:
    gdf = gdf.copy()

    projected_crs = gdf.estimate_utm_crs()
    centroids = gdf.to_crs(projected_crs).set_index(index_col).centroid.to_crs(CRS)
    centroids = gpd.GeoDataFrame(centroids, columns=["centroid"])

    gdf = gdf.join(centroids, on=index_col)
    gdf[lon_name] = gdf["centroid"].x
    gdf[lat_name] = gdf["centroid"].y

    return gdf.drop(columns=["centroid"])


def build_stop_names_to_loc_id_mapping(stops_df: pl.DataFrame) -> pl.DataFrame:
    df = stops_df.with_columns(
        pl.when(pl.col("stop_id").str.starts_with("Parent"))
        .then("stop_id")
        .otherwise("parent_station")
        .alias("parent_station")
    )

    with_parents = (
        df.filter(pl.col("parent_station") != "")
        .group_by("parent_station")
        .agg(pl.col("stop_name").first().alias("loc_name"))
        .rename({"parent_station": "loc_id"})
        .select("loc_name", "loc_id")
    )

    without_parents = df.filter(pl.col("parent_station") == "", ~pl.col("stop_id").str.starts_with("Parent")).select(
        pl.col("stop_name").alias("loc_name"),
        pl.col("stop_id").alias("loc_id"),
    )

    return pl.concat([with_parents, without_parents])


def match_loc_ids(trips_df: pl.DataFrame, locations_df: pl.DataFrame, stops_df: pl.DataFrame) -> pl.DataFrame:
    stop_names_to_id_df = build_stop_names_to_loc_id_mapping(stops_df)

    trips_df = manual_patch_stop_names(trips_df, "lieu_depart_trajet", "lieu_arrivee_trajet")
    matched_df = match_loc_id_on_strict_stop_name(trips_df, stop_names_to_id_df)

    columns = [
        ("dep_loc_id", "lieu_depart_trajet", "dep_match_type"),
        ("arr_loc_id", "lieu_arrivee_trajet", "arr_match_type"),
    ]

    for loc_id_col, stop_name_col, match_type_col in columns:
        matched_df = (
            matched_df.pipe(match_loc_id_on_subsector, locations_df, loc_id_col, stop_name_col, match_type_col)
            .pipe(match_loc_id_on_municipality_swiss, locations_df, loc_id_col, stop_name_col, match_type_col)
            .pipe(match_loc_id_on_municipality_french, locations_df, loc_id_col, stop_name_col, match_type_col)
            .pipe(match_loc_id_on_na, locations_df, loc_id_col, stop_name_col, match_type_col)
            .pipe(match_loc_id_on_fuzzy_stop_names, stop_names_to_id_df, loc_id_col, stop_name_col, match_type_col)
            .pipe(match_null_loc_ids_to_na, loc_id_col, match_type_col)
        )

    return matched_df


def manual_patch_stop_names(trips_df: pl.DataFrame, *stop_name_cols: str) -> pl.DataFrame:
    return trips_df.with_columns(pl.col(stop_name_cols).replace(STOP_NAME_MAPPING))


def match_loc_id_on_strict_stop_name(trips_df: pl.DataFrame, stop_names_to_id_df: pl.DataFrame) -> pl.DataFrame:
    return (
        trips_df.join(
            stop_names_to_id_df.select(pl.all().name.prefix("dep_"), dep_match_type=pl.lit("strict_stop_name")),
            left_on=pl.col("lieu_depart_trajet").str.to_lowercase(),
            right_on=pl.col("dep_loc_name").str.to_lowercase(),
            how="left",
        )
        .join(
            stop_names_to_id_df.select(pl.all().name.prefix("arr_"), arr_match_type=pl.lit("strict_stop_name")),
            left_on=pl.col("lieu_arrivee_trajet").str.to_lowercase(),
            right_on=pl.col("arr_loc_name").str.to_lowercase(),
            how="left",
        )
        .drop(cs.ends_with("loc_name"))
    )


def match_loc_id_on_subsector(
    df: pl.DataFrame, all_locations_df: pl.DataFrame, loc_id_col: str, stop_name_col: str, match_type_col: str
):
    regex = LOCATION_REGEXES["subsector"]
    subsectors = all_locations_df.filter(pl.col("type") == "subsector").select("loc_id", "loc_name")
    does_regex_match_expr = pl.col(loc_id_col).is_null() & ~pl.col("loc_id").is_null()

    return (
        df.with_columns(pl.col(stop_name_col).str.extract(regex, 1).alias("match"))
        .join(
            subsectors,
            left_on=pl.col("match").str.strip_chars().str.to_lowercase(),
            right_on=pl.col("loc_name").str.to_lowercase(),
            how="left",
        )
        .with_columns(
            pl.when(does_regex_match_expr).then("loc_id").otherwise(loc_id_col).alias(loc_id_col),
            pl.when(does_regex_match_expr).then(pl.lit("subsector")).otherwise(match_type_col).alias(match_type_col),
        )
        .drop("match", "loc_id", "loc_name")
    )


def match_loc_id_on_municipality_swiss(
    df: pl.DataFrame, all_locations_df: pl.DataFrame, loc_id_col: str, stop_name_col: str, match_type_col: str
):
    regex = LOCATION_REGEXES["municipality_swiss"]
    municipalities = all_locations_df.filter(pl.col("type") == "municipality_swiss").select("loc_id", "loc_name")
    does_regex_match_expr = (
        pl.col(loc_id_col).is_null() & ~pl.col("loc_id").is_null() & pl.col(stop_name_col).str.contains(regex)
    )

    return (
        df.join(
            municipalities,
            left_on=stop_name_col,
            right_on="loc_name",
            how="left",
        )
        .with_columns(
            pl.when(does_regex_match_expr).then("loc_id").otherwise(loc_id_col).alias(loc_id_col),
            pl.when(does_regex_match_expr)
            .then(pl.lit("municipality_swiss"))
            .otherwise(match_type_col)
            .alias(match_type_col),
        )
        .drop("loc_id")
    )


def match_loc_id_on_municipality_french(
    df: pl.DataFrame, all_locations_df: pl.DataFrame, loc_id_col: str, stop_name_col: str, match_type_col: str
):
    regex = LOCATION_REGEXES["municipality_french"]
    municipalities = all_locations_df.filter(pl.col("type") == "municipality_french").select("loc_id", "loc_name")
    does_regex_match_expr = pl.col(loc_id_col).is_null() & ~pl.col("loc_id").is_null()

    return (
        df.join(
            municipalities,
            left_on=pl.col(stop_name_col).str.extract(regex, 2),
            right_on=pl.col("loc_id").str.strip_prefix("FR-"),
            how="left",
        )
        .with_columns(
            pl.when(does_regex_match_expr).then("loc_id").otherwise(loc_id_col).alias(loc_id_col),
            pl.when(does_regex_match_expr)
            .then(pl.lit("municipality_french"))
            .otherwise(match_type_col)
            .alias(match_type_col),
        )
        .drop("loc_id", "loc_name")
    )


def match_loc_id_on_na(
    df: pl.DataFrame, all_locations_df: pl.DataFrame, loc_id_col: str, stop_name_col: str, match_type_col: str
) -> pl.DataFrame:
    regex = LOCATION_REGEXES["na"]
    municipalities = all_locations_df.filter(pl.col("type") == "na").select("loc_id", "loc_name")
    does_regex_match_expr = (
        pl.col(loc_id_col).is_null() & ~pl.col("loc_id").is_null() & pl.col(stop_name_col).str.contains(regex)
    )

    return (
        df.join(
            municipalities,
            left_on=pl.col(stop_name_col),
            right_on=pl.col("loc_name"),
            how="left",
        )
        .with_columns(
            pl.when(does_regex_match_expr).then("loc_id").otherwise(loc_id_col).alias(loc_id_col),
            pl.when(does_regex_match_expr).then(pl.lit("na")).otherwise(match_type_col).alias(match_type_col),
        )
        .drop("loc_id")
    )


def match_loc_id_on_fuzzy_stop_names(
    df: pl.DataFrame, stop_names_to_id_df: pl.DataFrame, loc_id_col: str, stop_name_col: str, match_type_col: str
) -> pl.DataFrame:
    unmatched_stop_names = pl.concat([
        df.filter(pl.col("dep_loc_id").is_null())["lieu_depart_trajet"].rename("stop_name"),
        df.filter(pl.col("arr_loc_id").is_null())["lieu_arrivee_trajet"].rename("stop_name"),
    ]).unique()
    unmatched_stop_names = pl.DataFrame(unmatched_stop_names)

    stop_names_list: list[str] = stop_names_to_id_df["loc_name"].str.to_lowercase().to_list()
    stop_ids_list = stop_names_to_id_df["loc_id"].to_list()

    def fuzzy_match(stop_name: str):
        best_name, best_score, best_idx = process.extractOne(stop_name, stop_names_list, scorer=fuzz.ratio)

        return {
            "closest_stop_name": best_name,
            "closest_stop_id": stop_ids_list[best_idx],
            "score": best_score,
        }

    stripped_stop_names = pl.col("stop_name").str.to_lowercase().str.strip_chars("0123456789- ")
    matched_stop_names = (
        unmatched_stop_names.with_columns(stripped_stop_names.map_elements(fuzzy_match).alias("result"))
        .unnest("result")
        .filter(pl.col("score") > FUZZY_MATCH_THRESHOLD)
        .select("stop_name", pl.col("closest_stop_id").alias("loc_id"))
    )

    is_fuzzy_match_success = pl.col(loc_id_col).is_null() & pl.col("loc_id").is_not_null()

    return (
        df.join(matched_stop_names, left_on=stop_name_col, right_on="stop_name", how="left")
        .with_columns(
            pl.when(is_fuzzy_match_success).then("loc_id").otherwise(loc_id_col).alias(loc_id_col),
            pl.when(is_fuzzy_match_success)
            .then(pl.lit("fuzzy_stop_name"))
            .otherwise(match_type_col)
            .alias(match_type_col),
        )
        .drop("loc_id")
    )


def match_null_loc_ids_to_na(df: pl.DataFrame, loc_id_col: str, match_type_col: str) -> pl.DataFrame:
    is_unmatched = pl.col(loc_id_col).is_null()

    return df.with_columns(
        pl.when(is_unmatched).then(pl.lit("NA")).otherwise(loc_id_col).alias(loc_id_col),
        pl.when(is_unmatched).then(pl.lit("na")).otherwise(match_type_col).alias(match_type_col),
    )


def plot(*gdfs: gpd.GeoDataFrame | tuple[gpd.GeoDataFrame, dict], cfg: Config):
    fig, ax = plt.subplots()
    for i, arg in enumerate(gdfs):
        if isinstance(arg, gpd.GeoDataFrame):
            g, p = arg, {}
        else:
            g, p = arg

        g.plot(ax=ax, **p)

    fig.savefig(Path(cfg.paths.figures) / "geoplot.png", bbox_inches="tight")


def test(cfg: Config):
    stops_df, cantons_gdf = load_files(cfg)

    geneva_gdf = cantons_gdf[cantons_gdf["NAME"] == "Genève"]
    geneva_shp = geneva_gdf.iloc[0]["geometry"]

    stops_gdf = gpd.GeoDataFrame(
        stops_df.to_pandas(), geometry=gpd.points_from_xy(stops_df["stop_lon"], stops_df["stop_lat"], crs="EPSG:4326")
    )

    geneva_stops = stops_gdf[stops_gdf.intersects(geneva_shp)]
    print(
        pl.DataFrame(geneva_stops.drop(columns=["geometry"])).with_columns(
            pl.col("parent_station").str.strip_prefix("Parent")
        )
    )

    plot(geneva_gdf, (geneva_stops, {"c": "orange", "markersize": 3, "alpha": 0.7}), cfg=cfg)
