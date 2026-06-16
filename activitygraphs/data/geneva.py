"""
Geneva TPG survey loader and NetworkData subclass.

This module handles loading and parsing the Geneva Transport Public Genevois (TPG)
travel survey. The survey records individual trip legs with origin/destination location
names (free text) that must be matched to a standardised location vocabulary.

The location hierarchy used has four types:
  - ``subsector``: 325+ fine-grained urban zones in the Geneva canton (main spatial unit).
  - ``municipality_swiss`` / ``municipality_geneva``: Swiss postal code areas.
  - ``municipality_french``: French postal code areas across the border.
  - ``public_transport``: PT stops from GTFS (used to resolve PT-named origins/destinations).
  - ``na``: Sentinel location for trips with unknown endpoints.

Location matching uses a cascade of strategies (strict name match → subsector regex →
Swiss/French municipality regex → NA regex → fuzzy string match → fallback to NA).

Key classes/functions:
  - ``GenevaInputs``: Frozen dataclass holding all parsed raw input files.
  - ``GenevaData``: NetworkData subclass for Geneva with caching support.
  - ``load_files``: Reads all raw Geneva files from disk.
  - ``build_geneva_data``: Orchestrates parsing from raw files to standardised GenevaData.
  - ``match_loc_ids``: Cascade location-ID matching pipeline for raw trip location strings.
"""

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd
import polars as pl
from polars import selectors as cs
from rapidfuzz import fuzz, process

from activitygraphs import utils
from activitygraphs.base import CRS, LOCATIONS_COLUMNS, LOCATIONS_SCHEMA, USER_JOURNEY_SCHEMA, Mode, Purpose
from activitygraphs.config import DataConfig, GenevaDataConfig
from activitygraphs.data.gtfs import GTFSInputs
from activitygraphs.network import (
    NetworkData,
    build_special_locations,
)
from activitygraphs.utils import DataFrameStore, add_lon_lat_from_centroid, check_schema

# Official Swiss canton number for Geneva, used to filter boundaries to the Geneva canton.
GENEVA_CANTON_CODE = 25

# Regular expressions used to identify the location type from the raw survey text.
# Each pattern matches the suffix format used in the survey's location name strings.
# Group 1 captures the human-readable name; group 2 (if present) captures the postal code.
LOCATION_REGEXES = {
    "subsector": r"([\s\S]+) - sous_secteur\s*$",         # E.g. "Jonction - sous_secteur"
    "municipality_swiss": r"([\s\S]+) - (\d\d\d\d)\s*$",  # E.g. "Genève - 1200" (4-digit Swiss ZIP)
    "municipality_french": r"([\s\S]+) - (\d\d\d\d\d)\s*$", # E.g. "Annemasse - 74100" (5-digit French)
    "na": r"NA",                                           # Exact string "NA" for unknown locations
}

# Manual corrections for survey stop names that don't match GTFS stop names exactly.
# Keys are the raw survey string; values are the corrected/canonical form used in GTFS.
# "Domicile" and "Home" both map to "NA" (home location handled separately).
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

# Mapping from raw Geneva survey mode strings (French) to the internal Mode enum.
# The raw survey uses French mode labels; this dict translates them to the canonical Mode values.
MODE_MAPPING = {
    "mode_autre": Mode.OTHER,                # Any other / unclassified mode
    "mode_bateau_navette": Mode.BOAT,        # Ferry / lake shuttle boat
    "mode_bus": Mode.BUS,                    # City bus or regional bus
    "mode_car_interurbain": Mode.COACH,      # Long-distance coach
    "mode_marche_à_pied": Mode.WALK,         # Walking
    "mode_moto_scooter": Mode.MOTORCYCLE,    # Motorcycle or scooter
    "mode_taxi_vtc": Mode.TAXI,              # Taxi or rideshare (VTC)
    "mode_train": Mode.TRAIN,                # Regional or intercity train
    "mode_tramway": Mode.TRAMWAY,            # Tram / light rail
    "mode_trottinette": Mode.CYCLE,          # Electric scooter / kick scooter (treated as cycle)
    "mode_velo": Mode.CYCLE,                 # Bicycle
    "mode_voiture_conducteur": Mode.CAR,     # Car (driver)
    "mode_voiture_passager": Mode.VEH_PASS,  # Car (passenger)
}

# Mapping from raw Geneva survey purpose strings (French) to the internal Purpose enum.
# Each French purpose code corresponds to a standardised activity purpose.
PURPOSE_MAPPING = {
    "od_lieu_achat": Purpose.SHOP,                        # Shopping
    "od_lieu_autre": Purpose.OTHER,                       # Other / unclassified
    "od_lieu_autrelieutravail": Purpose.WORK_OTHER,       # Work at a location other than main workplace
    "od_lieu_autreloisir": Purpose.LEISURE_OTHER,         # Other leisure activity
    "od_lieu_domicile": Purpose.HOME,                     # Home
    "od_lieu_etude": Purpose.STUDY,                       # School / university
    "od_lieu_resto": Purpose.ENTERTAINMENT,               # Restaurant / bar / entertainment
    "od_lieu_travail": Purpose.WORK_MAIN,                 # Main workplace
    "od_lieu_visite": Purpose.VISIT,                      # Visiting family or friends
    "od_lieu_voyage_longue_distance": Purpose.LONG_DISTANCE_TRIP,  # Long-distance travel
}

# Minimum fuzzy match score (0-100) for a stop name to be accepted as a valid match.
# Scores below this threshold are treated as non-matches (location falls back to NA).
FUZZY_MATCH_THRESHOLD = 65


@dataclass(frozen=True)
class GenevaInputs:
    """
    Description: Immutable container (frozen dataclass) holding all raw input DataFrames
    and GeoDataFrames parsed from disk for the Geneva TPG survey. Passed to
    ``build_geneva_data`` and stored on the ``GenevaData`` instance for reference.

    Attributes:
      - raw_journeys_df (pl.DataFrame): Raw survey trip-leg records as read from the
        parquet file, before any parsing or standardisation.
      - subsectors_gdf (gpd.GeoDataFrame): Geneva subsector polygon boundaries.
        Used to build the ``subsector`` location type and the network graph nodes.
      - postcodes_gdf (gpd.GeoDataFrame): Swiss postal code polygon boundaries.
        Used to build ``municipality_swiss`` locations.
      - localities_gdf (gpd.GeoDataFrame): Swiss locality (commune) boundaries.
        Used together with postcodes to assign Swiss municipality loc_ids.
      - swiss_boundaries_gdf (gpd.GeoDataFrame): Swiss administrative canton boundaries.
        Used to determine which municipalities lie inside the Geneva canton.
      - french_gdf (gpd.GeoDataFrame): French postal code polygon boundaries across the border.
        Used to build ``municipality_french`` locations.
      - gtfs (GTFSInputs): Parsed GTFS public transport feed tables (stops, routes, etc.).
        Used to build the ``public_transport`` location type and match PT stop names.
    """

    raw_journeys_df: pl.DataFrame   # Raw trip-leg records from the survey parquet file

    subsectors_gdf: gpd.GeoDataFrame      # Geneva subsector polygons
    postcodes_gdf: gpd.GeoDataFrame       # Swiss postal code polygons
    localities_gdf: gpd.GeoDataFrame      # Swiss locality/commune polygons
    swiss_boundaries_gdf: gpd.GeoDataFrame  # Swiss canton administrative boundaries
    french_gdf: gpd.GeoDataFrame          # French postal code polygons

    gtfs: GTFSInputs  # All GTFS feed tables for the Geneva public transport network


class GenevaData(NetworkData, DataFrameStore):
    """
    Description: Concrete NetworkData subclass for the Geneva TPG travel survey.
    Inherits the filterable network view from ``NetworkData`` and disk-caching behaviour
    from ``DataFrameStore``. Stores a reference to the raw ``GenevaInputs`` and the GTFS
    feed for downstream use (e.g. PT edge building).

    The class supports both in-memory construction (via ``build_geneva_data``) and
    disk-cached loading (via ``GenevaData.load``).
    """

    def __init__(
        self,
        inputs: GenevaInputs,
        locations_gdf: gpd.GeoDataFrame,
        user_journeys_df: pl.DataFrame,
        filters: list[str] | None = None,
    ):
        """
        Description: Initialise GenevaData by delegating to NetworkData.__init__ and
        storing the Geneva-specific raw inputs and GTFS reference.

        Input:
          - inputs (GenevaInputs): All raw parsed input data for this dataset.
          - locations_gdf (gpd.GeoDataFrame): Standardised location table (all types).
          - user_journeys_df (pl.DataFrame): Standardised trip-leg table.
          - filters (list[str] | None): Optional location-type filter to apply. Passed to
            ``NetworkData.__init__``.

        Output:
          - (None): Initialises the instance; no return value.
        """
        super().__init__(
            user_journeys_df,
            locations_gdf,
            filters,
        )

        # Store raw inputs for reference (e.g. boundaries for spatial joins)
        self.inputs = inputs
        # Shortcut to the GTFS tables (used by PT layer builders downstream)
        self.gtfs = self.inputs.gtfs

    def _copy(self, filters: list[str] | None = None):
        """
        Description: Create a new GenevaData instance sharing the same raw data
        but with a different location-type filter. Called by ``NetworkData.with_filter``.

        Input:
          - filters (list[str] | None): New filter list to apply.

        Output:
          - (GenevaData): New instance with the filter applied.
        """
        return GenevaData(self.inputs, self._locations_gdf, self._user_journeys_df, filters)

    @classmethod
    def load(cls, cfg: GenevaDataConfig, project_root: Path | None = None, name: str | None = None) -> "GenevaData":
        """
        Description: Load GenevaData from a Parquet cache if available, otherwise build
        it from scratch and save it to disk. Always loads raw GTFS and boundary files
        (these are not cached as they are fast to parse).

        Input:
          - cfg (GenevaDataConfig): Geneva-specific configuration with file paths.
          - project_root (Path | None): Project root directory.
          - name (str | None): Optional sub-directory name for the cache. Defaults to class name.

        Output:
          - (GenevaData): Loaded (or freshly built) GenevaData instance.
        """
        # Determine the cache directory path from the config
        project_root, data_dir = cls._dirs(cfg, project_root, name)
        # Always load the raw input files (GTFS, boundaries etc.) since they are needed
        # both for cache construction and as metadata on the instance
        gva_inputs = load_files(cfg, project_root)

        if data_dir.exists():
            # Cache hit: load pre-processed locations and journeys from parquet files
            locations_gdf = gpd.read_parquet(data_dir / "locations_gdf.parquet")
            user_journeys_df = pl.read_parquet(data_dir / "user_journeys_df.parquet", schema=USER_JOURNEY_SCHEMA)

            return cls(gva_inputs, locations_gdf, user_journeys_df)
        else:
            # Cache miss: build from raw inputs and save to disk
            data = build_geneva_data(gva_inputs)
            data.save(cfg, project_root, name)

            return build_geneva_data(gva_inputs)

    def save(self, cfg: GenevaDataConfig, project_root: Path | None = None, name: str | None = None):
        """
        Description: Persist the processed locations GeoDataFrame and user journeys
        DataFrame to Parquet files for fast reloading on subsequent runs.

        Input:
          - cfg (GenevaDataConfig): Geneva configuration with processed data path.
          - project_root (Path | None): Project root directory.
          - name (str | None): Optional cache sub-directory name.

        Output:
          - (None): Writes files to disk; no return value.
        """
        # Resolve the cache directory path
        project_root, data_dir = self._dirs(cfg, project_root, name)

        # Create the directory (and any parents) if it doesn't exist
        data_dir.mkdir(parents=True, exist_ok=True)
        # Save the filtered locations GeoDataFrame as Parquet (preserves geometry)
        self.locations_gdf.to_parquet(data_dir / "locations_gdf.parquet")
        # Save the user journey DataFrame as Parquet (preserves all column types)
        self.user_journeys_df.write_parquet(data_dir / "user_journeys_df.parquet")


def load_files(cfg: GenevaDataConfig, project_root: Path | None = None) -> GenevaInputs:
    """
    Description: Read all raw input files for the Geneva dataset from disk and return them
    packaged in a ``GenevaInputs`` container. This includes the survey trip-leg parquet,
    all geographic boundary shapefiles, and all GTFS CSV tables.

    Input:
      - cfg (GenevaDataConfig): Configuration specifying file paths for all inputs.
      - project_root (Path | None): Root directory from which relative paths are resolved.
        Defaults to the current working directory.

    Output:
      - (GenevaInputs): Frozen dataclass containing all parsed input DataFrames and
        GeoDataFrames ready for use in ``build_geneva_data``.
    """

    def parse_gtfs_date(*cols: str) -> pl.Expr:
        """
        Description: Helper expression that parses GTFS date columns (stored as integers
        in YYYYMMDD format) into Polars Date values.

        Input:
          - *cols (str): One or more column name strings to parse.

        Output:
          - (pl.Expr): A Polars expression that casts the columns to string and then
            parses as date with the format ``"%Y%m%d"``.
        """
        return pl.col(*cols).cast(pl.String).str.to_date("%Y%m%d")

    # Resolve the project root (falls back to "." if not specified)
    project_root: Path = project_root if project_root is not None else Path(".")

    # Directory containing the raw survey parquet file
    raw_path = project_root / cfg.paths.raw

    # Directory containing geographic boundary shapefiles
    boundaries_path = project_root / cfg.inputs.boundaries.directory
    # Config object with individual shapefile names
    boundaries_files = cfg.inputs.boundaries

    # Directory containing GTFS CSV files
    gtfs_path = project_root / cfg.paths.gtfs
    # Config object with individual GTFS file names
    gtfs_files = cfg.inputs.gtfs

    # Load the raw survey trip-leg records (all trips, all users, all legs)
    raw_journeys_df = pl.read_parquet(raw_path / cfg.inputs.raw_journeys)

    # Load all geographic boundary shapefiles and reproject to the project CRS
    subsectors_gdf = gpd.read_file(boundaries_path / boundaries_files.geneva_subsectors).to_crs(CRS)
    postcodes_gdf = gpd.read_file(boundaries_path / boundaries_files.swiss_postcodes).to_crs(CRS)
    localities_gdf = gpd.read_file(boundaries_path / boundaries_files.swiss_localities).to_crs(CRS)
    boundaries_gdf = gpd.read_file(boundaries_path / boundaries_files.swiss_boundaries).to_crs(CRS)
    french_gdf = gpd.read_file(boundaries_path / boundaries_files.french_postcodes).to_crs(CRS)

    # Load the GTFS stops table and derive a canonical loc_id:
    # If the stop has a parent station, use the parent station's ID as loc_id;
    # otherwise use the stop's own ID. This groups platforms under one loc_id.
    stops_df = pl.read_csv(
        gtfs_path / gtfs_files.stops,
        schema={
            "stop_id": pl.String,
            "stop_name": pl.String,
            "stop_lat": pl.Float32,
            "stop_lon": pl.Float32,
            "location_type": pl.Categorical(),
            "parent_station": pl.String,
        },
    ).with_columns(loc_id=pl.when(pl.col("parent_station") == "").then("stop_id").otherwise("parent_station"))

    # Build a mapping from individual stop_id to its canonical loc_id (parent station or self)
    stop_id_to_loc_id = stops_df.select("stop_id", "loc_id").lazy()
    # Load stop times lazily (large file); join with loc_id mapping for downstream use
    stop_times_df = pl.scan_csv(gtfs_path / gtfs_files.stop_times).join(stop_id_to_loc_id, on="stop_id", how="left")
    # Load trips lazily (large file)
    trips_df = pl.scan_csv(gtfs_path / gtfs_files.trips)
    # Load routes, agency, and calendar tables eagerly (smaller files)
    routes_df = pl.read_csv(gtfs_path / gtfs_files.routes)
    agency_df = pl.read_csv(gtfs_path / gtfs_files.agency)
    # Parse GTFS date integer columns to proper Date type
    calendar_df = pl.read_csv(gtfs_path / gtfs_files.calendar).with_columns(parse_gtfs_date("start_date", "end_date"))
    calendar_dates_df = pl.read_csv(gtfs_path / gtfs_files.calendar_dates).with_columns(parse_gtfs_date("date"))

    # Load the GTFS transfers table (walk times between connected stops)
    transfers_df = pl.read_csv(
        gtfs_path / gtfs_files.transfers,
        schema={
            "from_stop_id": pl.String,
            "to_stop_id": pl.String,
            "transfer_type": pl.Categorical(),
            "min_transfer_time": pl.Int64,
            # All other fields are empty in the 2022 Swiss GTFS timetable, and `transfer_type` is always 2
        },
    )

    # Bundle all GTFS tables into a single container
    gtfs = GTFSInputs(
        stops_df, stop_times_df, trips_df, routes_df, agency_df, calendar_df, calendar_dates_df, transfers_df
    )

    # Bundle everything into the immutable GenevaInputs container
    return GenevaInputs(
        raw_journeys_df, subsectors_gdf, postcodes_gdf, localities_gdf, boundaries_gdf, french_gdf, gtfs
    )


def build_geneva_data(inputs: GenevaInputs) -> GenevaData:
    """
    Description: Orchestrate the full parsing pipeline from raw Geneva inputs to a
    standardised ``GenevaData`` instance. Steps:
      1. Build the full location vocabulary (all types: subsectors, municipalities, PT stops).
      2. Clean null/empty values from the raw journeys DataFrame.
      3. Match raw free-text location names to canonical loc_ids using a cascade of strategies.
      4. Rename and parse columns to conform to USER_JOURNEY_SCHEMA.
      5. Filter locations to only those referenced in at least one user journey.

    Input:
      - inputs (GenevaInputs): All raw parsed files from ``load_files``.

    Output:
      - (GenevaData): Standardised GenevaData instance with validated schemas.
    """
    # Create locations with all PT stops
    # This builds the full location vocabulary used to match raw trip location strings
    locations_gdf = build_geneva_locations(
        inputs.gtfs.stops_df,
        inputs.subsectors_gdf,
        inputs.postcodes_gdf,
        inputs.localities_gdf,
        inputs.swiss_boundaries_gdf,
        inputs.french_gdf,
    )

    # Match user trips to existing locations then parse other columns
    # Step 1: Remove rows with missing date/time/purpose values
    user_journeys_df = _handle_null_values(inputs.raw_journeys_df)
    # Step 2: Match free-text location strings to canonical loc_ids
    user_journeys_df = match_loc_ids(user_journeys_df, utils.gdf_to_polars(locations_gdf), inputs.gtfs.stops_df)
    # Step 3: Select, rename, and parse columns to match USER_JOURNEY_SCHEMA
    user_journeys_df = user_journeys_df.select(
        user_id=pl.col("id_utilisateur").cast(pl.String),          # Survey respondent ID
        journey_id=pl.col("id_deplacement").cast(pl.String),       # Trip ID (groups legs)
        leg_id=pl.col("id_trajet").cast(pl.Int8),                  # Leg number within trip
        leg_mode=pl.col("mode").cast(pl.Categorical).replace(MODE_MAPPING),  # Translated mode
        leg_line=pl.col("ligne_trajet").cast(pl.String),            # PT line name (if applicable)
        duration=pl.duration(microseconds=0),                       # Duration not available in Geneva
        dep_day=pl.col("jour_depart").str.to_date("%+"),            # Departure date
        dep_time=pl.col("date").str.split(" - ").list.first().str.to_time("%R"),  # Departure time (HH:MM)
        dep_purpose=pl.col("motif_depart").replace_strict(PURPOSE_MAPPING).cast(pl.Categorical),
        dep_loc_id="dep_loc_id",    # Already matched loc_id for origin
        arr_loc_id="arr_loc_id",    # Already matched loc_id for destination
        arr_purpose=pl.col("motif_arrivee").replace_strict(PURPOSE_MAPPING).cast(pl.Categorical),
    )
    # Validate the final journey table schema
    user_journeys_df = check_schema(user_journeys_df, USER_JOURNEY_SCHEMA)

    # Filter out locations not in user journeys
    # Many PT stops and municipalities may have been created but never appear in any trip
    user_loc_ids = (
        pl
        .concat([user_journeys_df["dep_loc_id"], user_journeys_df["arr_loc_id"]])
        .unique()
        .rename("loc_id")
        .to_pandas()
    )
    # Inner merge: keep only locations that are referenced by at least one journey
    locations_gdf = locations_gdf.merge(user_loc_ids, on="loc_id")
    locations_gdf = check_schema(locations_gdf, LOCATIONS_SCHEMA)

    return GenevaData(inputs, locations_gdf, user_journeys_df)


def build_geneva_locations(
    stops: pl.DataFrame,
    subsectors: gpd.GeoDataFrame,
    postcodes: gpd.GeoDataFrame,
    localities: gpd.GeoDataFrame,
    swiss_boundaries: gpd.GeoDataFrame,
    french_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Description: Build the full Geneva location vocabulary by concatenating all location types
    (NA sentinel, PT stops, subsectors, Swiss municipalities, French municipalities) into a
    single GeoDataFrame. This vocabulary is used to match raw trip location strings and
    as the input to the network graph construction.

    Input:
      - stops (pl.DataFrame): GTFS stops table (from ``GenevaInputs.gtfs.stops_df``).
      - subsectors (gpd.GeoDataFrame): Geneva subsector polygon boundaries.
      - postcodes (gpd.GeoDataFrame): Swiss postal code polygons.
      - localities (gpd.GeoDataFrame): Swiss locality/commune polygons.
      - swiss_boundaries (gpd.GeoDataFrame): Swiss canton administrative boundaries.
      - french_gdf (gpd.GeoDataFrame): French postal code polygons.

    Output:
      - (gpd.GeoDataFrame): Combined location table conforming to LOCATIONS_SCHEMA.
        One row per location with columns: loc_id, loc_name, type, lon, lat, geometry.
    """
    # Build each location type separately then concatenate
    special_locations = build_special_locations()                    # Single "NA" sentinel location
    pt_locations = _build_pt_locations(stops)                       # One location per PT stop/station
    subsector_locations = _build_subsector_locations(subsectors)    # One per Geneva subsector
    municipality_swiss_locations = _build_municipality_swiss_locations(postcodes, localities, swiss_boundaries)
    municipality_french_locations = _build_municipality_french_locations(french_gdf)

    # Concatenate all types; reset_index to get a clean 0-based integer index
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


def _build_pt_locations(stops: pl.DataFrame) -> gpd.GeoDataFrame:
    """
    Description: Convert the GTFS stops table into a GeoDataFrame of public-transport
    locations with a Point geometry at each stop's coordinate.

    Input:
      - stops (pl.DataFrame): GTFS stops table (with stop_id, stop_name, stop_lon, stop_lat).

    Output:
      - (gpd.GeoDataFrame): Locations of type ``"public_transport"`` conforming to
        LOCATIONS_COLUMNS. One row per stop (using stop_id as loc_id).
    """
    # Select and rename columns to match the standard location schema
    pt_locations = stops.select(
        loc_id="stop_id", loc_name="stop_name", type=pl.lit("public_transport"), lon="stop_lon", lat="stop_lat"
    )

    # Create a Point geometry from lon/lat columns and return only LOCATIONS_COLUMNS
    return gpd.GeoDataFrame(
        pt_locations.to_pandas(), geometry=gpd.points_from_xy(pt_locations["lon"], pt_locations["lat"], crs=CRS)
    )[LOCATIONS_COLUMNS]


def _build_subsector_locations(subsectors_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Description: Convert the Geneva subsector polygon GeoDataFrame into a location table
    with centroid-derived lon/lat coordinates.

    Input:
      - subsectors_gdf (gpd.GeoDataFrame): Geneva subsector polygons with columns
        OBJECTID (unique integer ID) and NOM (French name).

    Output:
      - (gpd.GeoDataFrame): Locations of type ``"subsector"`` with loc_id in the form
        ``"subsector-<OBJECTID>"``, conforming to LOCATIONS_COLUMNS.
    """
    # Compute centroid lon/lat from polygon geometry; OBJECTID is the unique identifier
    subsector_locations = add_lon_lat_from_centroid(subsectors_gdf, index_col="OBJECTID")

    # Build canonical loc_id: "subsector-" prefix + integer OBJECTID
    subsector_locations["loc_id"] = "subsector-" + subsector_locations["OBJECTID"].astype(str)
    # Set the location type
    subsector_locations["type"] = "subsector"

    # Rename French column NOM to loc_name
    subsector_locations = subsector_locations.rename(columns={"NOM": "loc_name"})
    return subsector_locations[LOCATIONS_COLUMNS]


def _build_municipality_swiss_locations(
    postcodes_gdf: gpd.GeoDataFrame, localities_gdf: gpd.GeoDataFrame, boundaries_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Description: Build Swiss municipality locations by joining postcodes with locality names
    and classifying each as either ``"municipality_geneva"`` (inside the Geneva canton) or
    ``"municipality_swiss"`` (elsewhere in Switzerland). The Geneva canton is identified by
    both name matching and a spatial within-check.

    Input:
      - postcodes_gdf (gpd.GeoDataFrame): Swiss postal code polygon boundaries.
      - localities_gdf (gpd.GeoDataFrame): Swiss locality/commune polygons with NAME and LOCALITYID.
      - boundaries_gdf (gpd.GeoDataFrame): Swiss canton administrative boundaries with KANTONSNUM.

    Output:
      - (gpd.GeoDataFrame): Locations of type ``"municipality_swiss"`` or
        ``"municipality_geneva"`` conforming to LOCATIONS_COLUMNS.
    """
    # Project to local UTM for accurate within() spatial check
    projected_crs = localities_gdf.estimate_utm_crs()
    localities_gdf = localities_gdf.copy().to_crs(projected_crs)

    # Extract Geneva canton boundary polygons using the canton number
    geneva_boundaries_gdf = boundaries_gdf[boundaries_gdf["KANTONSNUM"] == GENEVA_CANTON_CODE]
    # Merge all Geneva canton polygons into one shape for within() test
    geneva_shape = geneva_boundaries_gdf.geometry.to_crs(projected_crs).union_all()
    # List of locality names that are known to be in Geneva
    # (includes manually added edge cases not captured by the boundary)
    geneva_locality_names = list(geneva_boundaries_gdf["NAME"]) + [
        "Athenaz (Avusy)",
        "La Croix-de-Rozon",
        "Collex",
        "Grand-Lancy",
        "Petit-Lancy",
        "Perly",
        "Chambésy",
        "Carouge GE",
        "Corsier GE",
    ]
    # Flag localities by name match (True if locality is in the Geneva name list)
    localities_gdf["in_geneva"] = localities_gdf["NAME"].isin(geneva_locality_names)
    # Also flag by spatial containment (handles edge cases not in the name list)
    localities_gdf["in_geneva"] = localities_gdf["in_geneva"] | localities_gdf.within(geneva_shape)

    # Keep only the columns needed from postcodes and compute centroid coordinates
    postcodes_gdf = postcodes_gdf[["FK_LOCALIT", "ZIP_ID", "ZIP4", "geometry"]]
    postcodes_gdf = add_lon_lat_from_centroid(postcodes_gdf, index_col="ZIP_ID")
    # Join localities with postcodes on the FK_LOCALIT / LOCALITYID relationship
    municipality_locations = localities_gdf.drop(columns=["geometry"]).merge(
        postcodes_gdf, left_on="LOCALITYID", right_on="FK_LOCALIT"
    )

    # Build loc_name: "LocalityName - ZIP4" e.g. "Genève - 1200"
    municipality_locations["loc_name"] = municipality_locations["NAME"] + " - " + municipality_locations["ZIP4"]
    # Build loc_id: "CH-ZIP4-IndexName" e.g. "CH-1200-GENEVE"
    municipality_locations["loc_id"] = (
        "CH-" + municipality_locations["ZIP4"] + "-" + municipality_locations["INDEXNAME"]
    )

    # Classify by Geneva membership
    in_geneva = municipality_locations["in_geneva"]
    municipality_locations.loc[~in_geneva, "type"] = "municipality_swiss"   # Outside Geneva
    municipality_locations.loc[in_geneva, "type"] = "municipality_geneva"   # Inside Geneva

    # Remove duplicate loc_ids (can arise from multiple postcodes per locality)
    municipality_locations = municipality_locations.drop_duplicates(subset="loc_id", keep="first").reset_index()
    return municipality_locations[LOCATIONS_COLUMNS]


def _build_municipality_french_locations(french_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Description: Convert the French postal code boundary GeoDataFrame into municipality
    locations across the French border from Geneva.

    Input:
      - french_gdf (gpd.GeoDataFrame): French postal code polygons with columns
        ID (postal code), LIB (commune name), geometry.

    Output:
      - (gpd.GeoDataFrame): Locations of type ``"municipality_french"`` conforming to
        LOCATIONS_COLUMNS. loc_id has the form ``"FR-<ID>"``.
    """
    # Compute centroid lon/lat for each French postal code polygon
    french_locations = add_lon_lat_from_centroid(french_gdf, index_col="ID")
    # Build loc_id: "FR-" prefix + 5-digit French postal code
    french_locations["loc_id"] = "FR-" + french_locations["ID"]
    # Build loc_name: uppercase commune name + " - " + postal code
    french_locations["loc_name"] = french_locations["LIB"].str.upper() + " - " + french_locations["ID"]
    french_locations["type"] = "municipality_french"

    return french_locations[LOCATIONS_COLUMNS]


def build_stop_names_to_loc_id_mapping(stops_df: pl.DataFrame) -> pl.DataFrame:
    """
    Description: Build a two-column mapping table from stop names (as they appear in the
    survey) to canonical loc_ids (parent station IDs or stop IDs for standalone stops).
    This table is used as a lookup during the strict and fuzzy stop-name matching steps.

    Input:
      - stops_df (pl.DataFrame): GTFS stops table with columns: stop_id, stop_name,
        parent_station, loc_id (derived from stops loading).

    Output:
      - (pl.DataFrame): Two-column table: loc_name (String), loc_id (String).
        One row per unique stop name, using the parent station's ID as loc_id
        when available, or the stop's own ID for standalone stops.
    """
    # Normalise parent_station: if stop_id starts with "Parent", it IS the parent station
    df = stops_df.with_columns(
        pl
        .when(pl.col("stop_id").str.starts_with("Parent"))
        .then("stop_id")                    # This stop is itself a parent station
        .otherwise("parent_station")        # Otherwise use the declared parent_station
        .alias("parent_station")
    )

    # Stops that have a parent station: group by parent to get one name per station
    with_parents = (
        df
        .filter(pl.col("parent_station") != "")   # Only stops with a parent
        .group_by("parent_station")
        .agg(pl.col("stop_name").first().alias("loc_name"))  # Use first child's name
        .rename({"parent_station": "loc_id"})      # Parent station ID becomes loc_id
        .select("loc_name", "loc_id")
    )

    # Standalone stops: neither a parent nor a child
    without_parents = df.filter(pl.col("parent_station") == "", ~pl.col("stop_id").str.starts_with("Parent")).select(
        pl.col("stop_name").alias("loc_name"),
        pl.col("stop_id").alias("loc_id"),
    )

    # Combine both groups into a single lookup table
    return pl.concat([with_parents, without_parents])


def match_loc_ids(user_journeys_df: pl.DataFrame, locations_df: pl.DataFrame, stops_df: pl.DataFrame) -> pl.DataFrame:
    """
    Description: Orchestrate the full cascade location-ID matching pipeline for raw Geneva
    trip records. For both departure and arrival location columns, applies matching
    strategies in order of precision, stopping as soon as a loc_id is assigned:

      1. Manual name patches (STOP_NAME_MAPPING corrections)
      2. Strict case-insensitive stop name match against GTFS
      3. Subsector regex match (``" - sous_secteur"`` suffix)
      4. Swiss municipality regex match (4-digit ZIP suffix)
      5. French municipality regex match (5-digit postal code suffix)
      6. NA regex match (literal "NA" string)
      7. Fuzzy stop name match (rapidfuzz, threshold 65)
      8. Final fallback: any remaining null → "NA"

    Input:
      - user_journeys_df (pl.DataFrame): Raw survey trip records after null handling.
        Must have columns: ``lieu_depart_trajet`` (origin name), ``lieu_arrivee_trajet``
        (destination name).
      - locations_df (pl.DataFrame): Full location vocabulary from ``build_geneva_locations``.
        Used for subsector, municipality, and NA matching steps.
      - stops_df (pl.DataFrame): GTFS stops table for stop-name matching.

    Output:
      - (pl.DataFrame): Same as input but with additional columns: dep_loc_id, arr_loc_id,
        dep_match_type, arr_match_type. Unmatched locations get loc_id="NA".
    """
    # Pre-build the stop name → loc_id lookup table
    stop_names_to_id_df = build_stop_names_to_loc_id_mapping(stops_df)

    # Step 1: Apply manual corrections for known problematic stop names
    user_journeys_df = manual_patch_stop_names(user_journeys_df, "lieu_depart_trajet", "lieu_arrivee_trajet")
    # Step 2: Strict stop-name match for both departure and arrival at once
    matched_df = match_loc_id_on_strict_stop_name(user_journeys_df, stop_names_to_id_df)

    # Column configuration: (loc_id output column, raw name input column, match type column)
    columns = [
        ("dep_loc_id", "lieu_depart_trajet", "dep_match_type"),   # Departure
        ("arr_loc_id", "lieu_arrivee_trajet", "arr_match_type"),  # Arrival
    ]

    # Apply remaining matching steps to both departure and arrival columns
    for loc_id_col, stop_name_col, match_type_col in columns:
        matched_df = (
            matched_df
            .pipe(match_loc_id_on_subsector, locations_df, loc_id_col, stop_name_col, match_type_col)
            .pipe(match_loc_id_on_municipality_swiss, locations_df, loc_id_col, stop_name_col, match_type_col)
            .pipe(match_loc_id_on_municipality_french, locations_df, loc_id_col, stop_name_col, match_type_col)
            .pipe(match_loc_id_on_na, locations_df, loc_id_col, stop_name_col, match_type_col)
            .pipe(match_loc_id_on_fuzzy_stop_names, stop_names_to_id_df, loc_id_col, stop_name_col, match_type_col)
            .pipe(match_null_loc_ids_to_na, loc_id_col, match_type_col)
        )

    return matched_df


def manual_patch_stop_names(user_journeys_df: pl.DataFrame, *stop_name_cols: str) -> pl.DataFrame:
    """
    Description: Apply the manual STOP_NAME_MAPPING corrections to location name columns.
    Replaces known problematic names (e.g. "Domicile" → "NA", "Genève-Cornavin" → "Genève")
    before the automated matching steps run.

    Input:
      - user_journeys_df (pl.DataFrame): Raw trip records with location name columns.
      - *stop_name_cols (str): Names of the columns to apply the mapping to.
        Typically ``"lieu_depart_trajet"`` and ``"lieu_arrivee_trajet"``.

    Output:
      - (pl.DataFrame): Same as input with the specified columns patched.
    """
    return user_journeys_df.with_columns(pl.col(stop_name_cols).replace(STOP_NAME_MAPPING))


def match_loc_id_on_strict_stop_name(user_journeys_df: pl.DataFrame, stop_names_to_id_df: pl.DataFrame) -> pl.DataFrame:
    """
    Description: Attempt a case-insensitive exact match of departure and arrival location
    names against the GTFS stop-name vocabulary. This is the highest-confidence matching
    step and runs first, before any regex or fuzzy strategies.

    Input:
      - user_journeys_df (pl.DataFrame): Trip records (after manual patching).
        Must have columns: lieu_depart_trajet, lieu_arrivee_trajet.
      - stop_names_to_id_df (pl.DataFrame): Stop name → loc_id lookup table from
        ``build_stop_names_to_loc_id_mapping``.

    Output:
      - (pl.DataFrame): Same as input with new columns: dep_loc_id, dep_match_type,
        arr_loc_id, arr_match_type. Unmatched rows have null in the loc_id columns.
    """
    return (
        user_journeys_df
        # Left join for departure: match lieu_depart_trajet (lowercased) to stop loc_name
        .join(
            stop_names_to_id_df.select(pl.all().name.prefix("dep_"), dep_match_type=pl.lit("strict_stop_name")),
            left_on=pl.col("lieu_depart_trajet").str.to_lowercase(),
            right_on=pl.col("dep_loc_name").str.to_lowercase(),
            how="left",
        )
        # Left join for arrival: same approach
        .join(
            stop_names_to_id_df.select(pl.all().name.prefix("arr_"), arr_match_type=pl.lit("strict_stop_name")),
            left_on=pl.col("lieu_arrivee_trajet").str.to_lowercase(),
            right_on=pl.col("arr_loc_name").str.to_lowercase(),
            how="left",
        )
        # Drop temporary loc_name columns added by the join (only keep loc_id)
        .drop(cs.ends_with("loc_name"))
    )


def match_loc_id_on_subsector(
    user_journeys_df: pl.DataFrame,
    all_locations_df: pl.DataFrame,
    loc_id_col: str,
    stop_name_col: str,
    match_type_col: str,
):
    """
    Description: Try to match unmatched location strings to subsector loc_ids by extracting
    the subsector name from the ``" - sous_secteur"`` suffix pattern and joining against
    the subsector vocabulary. Only updates rows where ``loc_id_col`` is still null.

    Input:
      - user_journeys_df (pl.DataFrame): Trip records with potentially null loc_id_col.
      - all_locations_df (pl.DataFrame): Full location vocabulary.
      - loc_id_col (str): Name of the column to write the matched loc_id to.
      - stop_name_col (str): Name of the column containing the raw location name string.
      - match_type_col (str): Column to record the match type (set to ``"subsector"``).

    Output:
      - (pl.DataFrame): Same as input but with subsector-matched rows filled in loc_id_col.
    """
    # Regex pattern to extract subsector name from suffix format
    regex = LOCATION_REGEXES["subsector"]
    # Only look at subsector-type locations
    subsectors = all_locations_df.filter(pl.col("type") == "subsector").select("loc_id", "loc_name")
    # Condition: loc_id is still null (not yet matched) AND a join match was found
    does_regex_match_expr = pl.col(loc_id_col).is_null() & ~pl.col("loc_id").is_null()

    return (
        user_journeys_df
        .with_columns(pl.col(stop_name_col).str.extract(regex, 1).alias("match"))
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
    user_journeys_df: pl.DataFrame,
    all_locations_df: pl.DataFrame,
    loc_id_col: str,
    stop_name_col: str,
    match_type_col: str,
):
    """
    Description: Match unmatched location strings to Swiss municipality loc_ids by direct
    loc_name join (format ``"LocalityName - XXXX"`` with a 4-digit ZIP code suffix).
    Only updates rows where ``loc_id_col`` is still null and the regex pattern is matched.

    Input:
      - user_journeys_df (pl.DataFrame): Trip records with potentially null loc_id_col.
      - all_locations_df (pl.DataFrame): Full location vocabulary.
      - loc_id_col (str): Name of the column to write the matched loc_id to.
      - stop_name_col (str): Name of the column containing the raw location name string.
      - match_type_col (str): Column to record the match type (set to ``"municipality_swiss"``).

    Output:
      - (pl.DataFrame): Same as input but with Swiss municipality rows filled in loc_id_col.
    """
    # 4-digit ZIP regex to confirm this is a Swiss municipality string
    regex = LOCATION_REGEXES["municipality_swiss"]
    # Only match against Swiss and Geneva municipality types
    municipalities = all_locations_df.filter(
        pl.col("type").is_in(["municipality_swiss", "municipality_geneva"])
    ).select("loc_id", "loc_name")
    # Condition: not yet matched, a join hit was found, AND the string contains the regex
    does_regex_match_expr = (
        pl.col(loc_id_col).is_null() & ~pl.col("loc_id").is_null() & pl.col(stop_name_col).str.contains(regex)
    )

    return (
        user_journeys_df
        .join(
            municipalities,
            left_on=stop_name_col,
            right_on="loc_name",
            how="left",
        )
        .with_columns(
            pl.when(does_regex_match_expr).then("loc_id").otherwise(loc_id_col).alias(loc_id_col),
            pl
            .when(does_regex_match_expr)
            .then(pl.lit("municipality_swiss"))
            .otherwise(match_type_col)
            .alias(match_type_col),
        )
        .drop("loc_id")
    )


def match_loc_id_on_municipality_french(
    user_journeys_df: pl.DataFrame,
    all_locations_df: pl.DataFrame,
    loc_id_col: str,
    stop_name_col: str,
    match_type_col: str,
):
    """
    Description: Match unmatched location strings to French municipality loc_ids by extracting
    the 5-digit French postal code from the location string and joining against the French
    municipality vocabulary. Only updates rows where ``loc_id_col`` is still null.

    Input:
      - user_journeys_df (pl.DataFrame): Trip records with potentially null loc_id_col.
      - all_locations_df (pl.DataFrame): Full location vocabulary.
      - loc_id_col (str): Name of the column to write the matched loc_id to.
      - stop_name_col (str): Name of the column containing the raw location name string.
      - match_type_col (str): Column to record the match type (set to ``"municipality_french"``).

    Output:
      - (pl.DataFrame): Same as input but with French municipality rows filled in loc_id_col.
    """
    # 5-digit postal code regex to extract the French postal code from the location string
    regex = LOCATION_REGEXES["municipality_french"]
    municipalities = all_locations_df.filter(pl.col("type") == "municipality_french").select("loc_id", "loc_name")
    # Condition: not yet matched and a join hit was found
    does_regex_match_expr = pl.col(loc_id_col).is_null() & ~pl.col("loc_id").is_null()

    return (
        user_journeys_df
        .join(
            municipalities,
            left_on=pl.col(stop_name_col).str.extract(regex, 2),
            right_on=pl.col("loc_id").str.strip_prefix("FR-"),
            how="left",
        )
        .with_columns(
            pl.when(does_regex_match_expr).then("loc_id").otherwise(loc_id_col).alias(loc_id_col),
            pl
            .when(does_regex_match_expr)
            .then(pl.lit("municipality_french"))
            .otherwise(match_type_col)
            .alias(match_type_col),
        )
        .drop("loc_id", "loc_name")
    )


def match_loc_id_on_na(
    user_journeys_df: pl.DataFrame,
    all_locations_df: pl.DataFrame,
    loc_id_col: str,
    stop_name_col: str,
    match_type_col: str,
) -> pl.DataFrame:
    """
    Description: Match location strings that literally equal "NA" to the special NA
    sentinel location. Only updates rows where ``loc_id_col`` is null and the raw
    location string matches the "NA" regex exactly.

    Input:
      - user_journeys_df (pl.DataFrame): Trip records with potentially null loc_id_col.
      - all_locations_df (pl.DataFrame): Full location vocabulary (must include an "na" type).
      - loc_id_col (str): Name of the column to write the matched loc_id to.
      - stop_name_col (str): Name of the column containing the raw location name string.
      - match_type_col (str): Column to record the match type (set to ``"na"``).

    Output:
      - (pl.DataFrame): Same as input with NA-string rows filled in loc_id_col.
    """
    # Matches the literal string "NA" in the location name
    regex = LOCATION_REGEXES["na"]
    municipalities = all_locations_df.filter(pl.col("type") == "na").select("loc_id", "loc_name")
    # Condition: not yet matched, a join hit found, AND the string contains "NA"
    does_regex_match_expr = (
        pl.col(loc_id_col).is_null() & ~pl.col("loc_id").is_null() & pl.col(stop_name_col).str.contains(regex)
    )

    return (
        user_journeys_df
        .join(
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
    user_journeys_df: pl.DataFrame,
    stop_names_to_id_df: pl.DataFrame,
    loc_id_col: str,
    stop_name_col: str,
    match_type_col: str,
) -> pl.DataFrame:
    """
    Description: Use fuzzy string matching (rapidfuzz) to match remaining unmatched location
    strings to GTFS stop names. This handles typos, abbreviations, and minor name differences.
    Only matches with a score above ``FUZZY_MATCH_THRESHOLD`` (65) are accepted.

    Input:
      - user_journeys_df (pl.DataFrame): Trip records with potentially null loc_id_col.
      - stop_names_to_id_df (pl.DataFrame): Stop name → loc_id lookup table.
      - loc_id_col (str): Name of the column to write the matched loc_id to.
      - stop_name_col (str): Name of the column containing the raw location name string.
      - match_type_col (str): Column to record the match type (set to ``"fuzzy_stop_name"``).

    Output:
      - (pl.DataFrame): Same as input with fuzzy-matched rows filled in loc_id_col.
    """
    # Collect all unique unmatched location names (from both dep and arr columns)
    # These are the names that all previous strategies failed to match
    unmatched_stop_names = pl.concat([
        user_journeys_df.filter(pl.col("dep_loc_id").is_null())["lieu_depart_trajet"].rename("stop_name"),
        user_journeys_df.filter(pl.col("arr_loc_id").is_null())["lieu_arrivee_trajet"].rename("stop_name"),
    ]).unique()
    unmatched_stop_names = pl.DataFrame(unmatched_stop_names)

    # Pre-build lists for rapidfuzz: all GTFS stop names (lowercased) and their loc_ids
    stop_names_list: list[str] = stop_names_to_id_df["loc_name"].str.to_lowercase().to_list()
    stop_ids_list = stop_names_to_id_df["loc_id"].to_list()

    def fuzzy_match(stop_name: str):
        """
        Description: Find the best fuzzy match for a single stop name string.

        Input:
          - stop_name (str): A lowercased, stripped location name string to match.

        Output:
          - (dict): Keys: closest_stop_name (str), closest_stop_id (str), score (float 0-100).
        """
        # noinspection PyTypeChecker
        best_name, best_score, best_idx = process.extractOne(stop_name, stop_names_list, scorer=fuzz.ratio)

        return {
            "closest_stop_name": best_name,       # The best-matching GTFS stop name
            "closest_stop_id": stop_ids_list[best_idx],  # Its loc_id
            "score": best_score,                  # Similarity score (0-100)
        }

    # Strip leading/trailing digits and separators before matching (improves recall)
    stripped_stop_names = pl.col("stop_name").str.to_lowercase().str.strip_chars("0123456789- ")
    # Apply fuzzy_match to each unique unmatched name; keep only above-threshold matches
    matched_stop_names = (
        unmatched_stop_names
        .with_columns(stripped_stop_names.map_elements(fuzzy_match).alias("result"))
        .unnest("result")                                  # Expand the dict into columns
        .filter(pl.col("score") > FUZZY_MATCH_THRESHOLD)  # Only keep high-confidence matches
        .select("stop_name", pl.col("closest_stop_id").alias("loc_id"))
    )

    # Condition: loc_id is still null but a fuzzy match was found
    is_fuzzy_match_success = pl.col(loc_id_col).is_null() & pl.col("loc_id").is_not_null()

    return (
        user_journeys_df
        # Join the fuzzy-match results onto the full journey table
        .join(matched_stop_names, left_on=stop_name_col, right_on="stop_name", how="left")
        .with_columns(
            pl.when(is_fuzzy_match_success).then("loc_id").otherwise(loc_id_col).alias(loc_id_col),
            pl
            .when(is_fuzzy_match_success)
            .then(pl.lit("fuzzy_stop_name"))
            .otherwise(match_type_col)
            .alias(match_type_col),
        )
        .drop("loc_id")  # Remove the temporary join column
    )


def match_null_loc_ids_to_na(user_journeys_df: pl.DataFrame, loc_id_col: str, match_type_col: str) -> pl.DataFrame:
    """
    Description: Final fallback step: any location that is still null after all matching
    strategies is assigned the "NA" sentinel loc_id. Ensures no null values remain in the
    loc_id columns before schema validation.

    Input:
      - user_journeys_df (pl.DataFrame): Trip records with potentially null loc_id_col.
      - loc_id_col (str): Column that may still have null values to fill.
      - match_type_col (str): Column to record match type (set to ``"na"`` for fallbacks).

    Output:
      - (pl.DataFrame): Same as input with all null loc_id values replaced by ``"NA"``.
    """
    # True for any row where the loc_id was never matched by any strategy
    is_unmatched = pl.col(loc_id_col).is_null()

    return user_journeys_df.with_columns(
        pl.when(is_unmatched).then(pl.lit("NA")).otherwise(loc_id_col).alias(loc_id_col),  # Assign NA sentinel
        pl.when(is_unmatched).then(pl.lit("na")).otherwise(match_type_col).alias(match_type_col),
    )


def _read_raw_data(data_cfg: DataConfig, project_root=None) -> pl.DataFrame:
    """
    Description: Read the raw Geneva survey parquet file with an explicit schema to ensure
    correct column types. Parses the departure date string into a proper Date column.
    This is an internal helper used before the main parsing pipeline.

    Input:
      - data_cfg (DataConfig): Configuration containing the raw data path and file name.
      - project_root: Project root directory; defaults to current working directory.

    Output:
      - (pl.DataFrame): Raw Geneva trip-leg records with typed columns.
    """
    # Resolve the project root
    project_root = project_root if project_root is not None else Path(".")
    data_path = project_root / data_cfg.paths.raw

    raw_geneva_df = utils.read_from_parquet(
        data_path / data_cfg.inputs.raw_journeys,
        schema={
            "id_utilisateur": pl.String,        # Survey respondent identifier
            "id_deplacement": pl.Categorical,    # Trip identifier (groups legs)
            "id_trajet": pl.Int8,                # Leg number within trip
            "jour_depart": pl.String,            # Departure date (string, to be parsed)
            "motif_depart": pl.Categorical,      # Activity purpose at departure
            "motif_arrivee": pl.Categorical,     # Activity purpose at arrival
            "date": pl.String,                   # Date + time range string
            "lieu_depart_trajet": pl.String,     # Departure location free-text name
            "lieu_arrivee_trajet": pl.String,    # Arrival location free-text name
            "mode": pl.Categorical,              # Transport mode (French label)
            "ligne_trajet": pl.String,           # PT line name (if applicable)
        },
    )

    # Parse the departure date from ISO-format string to a proper Polars Date
    raw_geneva_df = raw_geneva_df.with_columns(pl.col("jour_depart").str.to_date(format="%+", strict=False))

    return raw_geneva_df


def _handle_null_values(raw_geneva_df: pl.DataFrame) -> pl.DataFrame:
    """
    Description: Clean the raw Geneva survey DataFrame by removing unusable rows and
    imputing missing values with sensible defaults. Applied before any location matching.

    Steps:
      1. Drop rows with null departure date (can't be used without a date).
      2. Drop rows with empty strings in key columns (date, purpose fields).
      3. Impute empty location strings to ``"NA"`` (will be matched to NA sentinel).
      4. Impute empty mode to ``Mode.UNKNOWN``.
      5. Impute empty line name to ``"UNKNOWN"`` for PT modes or ``"NA"`` otherwise.

    Input:
      - raw_geneva_df (pl.DataFrame): Raw trip-leg records as loaded from parquet.

    Output:
      - (pl.DataFrame): Cleaned trip-leg records with no null dates and imputed values.
    """
    # Remove null rows
    # Rows without a departure date can't be assigned to any day and are useless
    cols_remove_null_rows = ["jour_depart"]
    removed_rows_df = raw_geneva_df.drop_nulls(subset=cols_remove_null_rows)

    # Remove empty rows from certain columns
    # Empty strings in these columns indicate completely missing/invalid records
    cols_remove_empty_rows = ["jour_depart", "date", "motif_depart", "motif_arrivee"]
    removed_rows_df = removed_rows_df.filter(pl.all_horizontal(pl.col(cols_remove_empty_rows) != ""))

    # Impute empty rows to `NA` category for location columns
    # Empty location names will later be matched to the "NA" sentinel location
    cols_impute_empty_to_na = ["lieu_depart_trajet", "lieu_arrivee_trajet"]
    imputed_na_rows_df = removed_rows_df.with_columns(pl.col(cols_impute_empty_to_na).replace(old="", new="NA"))

    # Impute empty values to unknown for mode column
    # A trip with no recorded mode gets the UNKNOWN category
    cols_impute_empty_to_unknown = ["mode"]
    imputed_unknown_rows_df = imputed_na_rows_df.with_columns(
        pl.col(cols_impute_empty_to_unknown).replace(old="", new=Mode.UNKNOWN)
    )

    # Impute empty values of `ligne_trajet` to UNKNOWN or NA for ligne column depending on if mode is applicable
    # PT modes (bus, tram, boat) should have a line; if missing, mark as UNKNOWN
    applicable_modes = ["mode_bus", "mode_tramway", "mode_bateau_navette"]
    imputed_line_df = imputed_unknown_rows_df.with_columns(
        pl
        .when((pl.col("ligne_trajet") == "") & pl.col("mode").is_in(applicable_modes))
        .then(pl.lit("UNKNOWN"))      # PT mode with no line → UNKNOWN
        .otherwise("ligne_trajet")
        .alias("ligne_trajet")
    ).with_columns(
        # Any remaining empty ligne_trajet (non-PT modes) → NA
        pl.when(pl.col("ligne_trajet") == "").then(pl.lit("NA")).otherwise("ligne_trajet").alias("ligne_trajet")
    )

    return imputed_line_df
