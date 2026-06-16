"""
Module: notebooks/thats/anonymise.py

Description:
    A Marimo interactive notebook for anonymising the THATS (Toronto Household Activity
    Travel Survey) dataset before sharing.

    The anonymisation pipeline performs the following steps in order:
      Step 1 – Remove non-useful columns and personally identifiable information (PII):
               TTS IDs, contact details, ethnicity, postal code, census geography codes.
      Step 2 – Reshuffle (anonymise) primary key IDs: hh_id, person_id, app_id, trip_id,
               and leg_id are replaced with zero-padded integer surrogates that preserve
               referential integrity across tables.
      Step 3 – Remove children (THATS age < 18) from individuals, activities and trips.
      Step 4 – Replace raw latitude/longitude coordinates with Census Tract (CT) and
               Dissemination Area (DA) identifiers using a spatial join against Statistics
               Canada boundary files.
      Step 5 – Write the anonymised tables to Parquet files.

    Input tables: HHDem (households), IndDem (individuals), Trips, HrAct (hourly activities).
    Output directory: data/raw/THATS/

Dependencies:
    - Marimo, Polars, GeoPandas
    - Statistics Canada 2021 Census boundary shapefiles (CT and DA)
"""

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import geopandas as gpd
    import polars as pl

    from pathlib import Path

    project_root = Path(mo.notebook_dir().parent.parent)


@app.cell
def _():
    """
    Description:
        Imports dataclass and asdict from the standard library.  dataclass is used to
        define the THATSData container below; asdict enables iterating over its fields
        as a plain dictionary when applying pipeline steps to all four tables at once.

    Output:
      - asdict (function): converts a dataclass instance to a dict of its fields.
      - dataclass (decorator): class decorator that auto-generates __init__ etc.
    """
    from dataclasses import dataclass, asdict

    return asdict, dataclass


@app.cell(hide_code=True)
def _():
    """
    Description: Render the notebook title heading for the THATS anonymisation notebook.

    Output:
      - (none): renders a Markdown heading in the notebook UI.
    """
    mo.md(r"""
    # Anonymising the THATS survey
    """)
    return


@app.cell
def _():
    """
    Description:
        Defines the paths to the original (pre-anonymisation) THATS survey data on the
        researcher's local machine.  These paths are only valid on the machine where the
        raw survey data is stored and are hard-coded intentionally (the raw data is never
        committed to the repository).

    Output:
      - raw_demos_path (Path): absolute path to the Demographics Excel files.
      - raw_trips_path (Path): absolute path to the Trips and Activities files.
    """
    # Root directory of the original (pre-anonymisation) THATS survey data on the researcher's machine
    raw_path = Path(
        "/home/luca/Documents/THATS/THATS Survey Data for Luca/Main Survey (GTHA)"
    )
    # Sub-directory containing household and individual demographic Excel files
    raw_demos_path = raw_path / "Demographics"
    # Sub-directory containing trip and hourly-activity Excel/CSV files
    raw_trips_path = raw_path / "Trips and Activities"
    return raw_demos_path, raw_trips_path


@app.cell
def _(dataclass, raw_demos_path, raw_trips_path):
    """
    Description: Load the four raw THATS survey tables (household demographics, individual
    demographics, GPS trips, and hourly activities) as Polars LazyFrames, define the
    ``THATSData`` dataclass container, and bundle all four tables into ``raw_thats``.

    Input:
      - dataclass (decorator): used to define the THATSData container.
      - raw_demos_path (Path): directory containing the Demographics Excel files.
      - raw_trips_path (Path): directory containing the Trips and Activities files.

    Output:
      - THATSData (class): the dataclass grouping the four survey tables.
      - raw_thats (THATSData): the four raw tables bundled for pipeline processing.
    """
    # Read each table as a LazyFrame so processing is deferred until needed.
    # Schema overrides ensure ID columns are read as strings (not auto-cast to integers).
    _indivs = pl.read_excel(
        raw_demos_path / "IndDem_October 25.xlsx",
        schema_overrides={
            "THATS HHID": pl.String,      # household ID — keep as string for safe joining
            "THATS PersonID": pl.String,  # person ID
            "THATS AppID": pl.String,     # smartphone app ID
        },
    ).lazy()

    _hhs = pl.read_excel(
        raw_demos_path / "HHDem_October 25 With LU and Transit Access Data.xlsx",
        schema_overrides={
            "THATS HHID": pl.String,  # household ID
        },
    ).lazy()
    _trips = pl.read_excel(raw_trips_path / "Trips.xlsx").lazy()

    # HrAct contains hourly activities; HHID and PersonID are stored as integers in the
    # CSV but we cast them to strings for consistent joining with the other tables.
    _activs = (
        pl
        .read_csv(
            raw_trips_path / "HrAct_Full_November 07.csv",
        )
        .with_columns(
            pl.col("THATS HHID", "THATS PersonID").cast(pl.Int64).cast(pl.String)
        )
        .lazy()
    )

    @dataclass
    class THATSData:
        """
        Description:
            A simple data container that groups the four THATS survey tables together.
            Using a dataclass makes it easy to pass all four tables as a single argument
            and to iterate over them using dataclasses.asdict().

        Input (constructor):
          - hhs (pl.LazyFrame): household demographics table.
          - indivs (pl.LazyFrame): individual demographics table.
          - trips (pl.LazyFrame): GPS-tracked trip legs table.
          - activs (pl.LazyFrame): hourly activity diary table.
        """
        hhs: pl.LazyFrame    # one row per household
        indivs: pl.LazyFrame  # one row per individual
        trips: pl.LazyFrame   # one row per trip leg
        activs: pl.LazyFrame  # one row per hourly activity record

    # Bundle the four raw tables into a single container for pipeline processing
    raw_thats = THATSData(_hhs, _indivs, _trips, _activs)
    return THATSData, raw_thats


@app.cell(hide_code=True)
def _():
    """
    Description: Render the Step 1 section heading — removing PII and non-informative columns.

    Output:
      - (none): renders a Markdown heading in the notebook UI.
    """
    mo.md(r"""
    ## Step 1 - Remove non-useful columns and personal information and sensitive attributes
    """)
    return


@app.cell
def _():
    """
    Description:
        Defines the lists of column names to drop from each THATS table during Step 1.
        Columns are dropped if they are personally identifiable (email, phone, name),
        duplicated from another table (TTS IDs), administratively derived (census codes),
        or not relevant to the research (geographic join artefacts like Shape_Leng).

    Output:
      - activ_drop_cols (list[str]): columns to drop from the hourly activities table.
      - hhs_drop_cols (list[str]): columns to drop from the households table.
      - indiv_drop_cols (list[str]): columns to drop from the individuals table.
    """
    indiv_drop_cols = [
        "TTS HHID",
        "TTS PersonID",
        "TTS PersonNumber",
        "THATS Survey Email",
        "THATS age - 0.8",
        "THATS agerange",
        "THATS agerangedesc",
        "THATS age(-0.8)range",
        "TTS agerange",
        "TTS agerangedesc",
        "TTS agecategory",
        "TTS agecategorydesc",
    ]

    hhs_drop_cols = [
        "TTS HHID",
        "ContactEmail",
        "TTS ContactName",
        "TTS ContactPhone",
        "THATS Ethnicity",
        "THATS HomePostalCode",
        "PR",
        "CDuid",
        "CSDname",
        "CCScode",
        "SAC",
        "CTname",
        "ER",
        "DPL",
        "FED13uid",
        "POP_CNTR_RA",
        "dauid",
        "DisBlock",
        "LAT",
        "LONG",
        "Comm_Name",
        "H_DMT",
        "PO",
        "QI",
        "PopulationDensity",
        "LandAreainSquareKm",
        "cbd_latitude",
        "cbd_longitude",
        "distance_to_cbd",
        "Distance_to_CBDPER1000",
        "geometry",
        "index_right",
        "FID",
        "OBJECTID",
        "Name",
        "Shape_Leng",
        "Shape_Area",
        "Shape__Are",
        "Shape__Len",
        "HasAccessToTransit",
    ]

    activ_drop_cols = [
        "THATS Survey Email",
    ]
    return activ_drop_cols, hhs_drop_cols, indiv_drop_cols


@app.cell
def _(THATSData, activ_drop_cols, hhs_drop_cols, indiv_drop_cols):
    """
    Description:
        Defines the drop_non_useful_columns pipeline step (Step 1 of anonymisation).
        Uses the column lists defined in the previous cell to remove PII and
        non-informative columns from the three affected tables.

    Input:
      - THATSData (class): the data container dataclass.
      - activ_drop_cols, hhs_drop_cols, indiv_drop_cols (list[str]): column lists.

    Output:
      - drop_non_useful_columns (function): pipeline step function, exported to the
            main pipeline cell.
    """
    def drop_non_useful_columns(thats: THATSData) -> THATSData:
        """
        Description:
            Removes PII and non-informative columns from the household, individual, and
            activity tables.  The trips table is not modified by this step.

        Input:
          - thats (THATSData): the raw THATS data container.

        Output:
          - (THATSData): a new THATSData with the specified columns dropped from each table.
        """
        hhs = thats.hhs.drop(hhs_drop_cols)       # drop household PII and geography
        indivs = thats.indivs.drop(indiv_drop_cols)  # drop individual PII and derived age columns
        activs = thats.activs.drop(activ_drop_cols)  # drop survey email from activities

        return THATSData(hhs, indivs, thats.trips, activs)

    return (drop_non_useful_columns,)


@app.cell(hide_code=True)
def _():
    """
    Description: Render the Step 2 section heading — reshuffling primary key IDs.

    Output:
      - (none): renders a Markdown heading in the notebook UI.
    """
    mo.md(r"""
    ## Step 2: Reshuffle IDs
    """)
    return


@app.cell
def _():
    """
    Description:
        Defines the id_equivalences mapping that tells the ID-reshuffling pipeline
        which column in each table corresponds to each semantic ID concept.

        Structure: { id_concept: { table_name: column_name_or_list } }

        For example, "hh_id" appears as "THATS HHID" in the hhs, indivs, and activs
        tables.  The mapping is used by generate_id_map and replace_ids_in_cols to
        ensure all references to the same household are updated consistently.

    Output:
      - id_equivalences (dict): the full ID concept -> table -> column(s) mapping.
    """
    id_equivalences = {
        "hh_id": {
            "indivs": "THATS HHID",
            "hhs": "THATS HHID",
            "activs": "THATS HHID",
        },
        "person_id": {
            "indivs": "THATS PersonID",
            "activs": "THATS PersonID",
        },
        "app_id": {
            "indivs": "THATS AppID",
            "trips": "user_uuid",
            "activs": "THATS AppID",
        },
        "trip_id": {
            "trips": "trip_id",
            "activs": [
                "1stStrtTripID",
                "2ndStrtTripID",
                "3rdStrtTripID",
                "4thStrtTripID",
                "5thStrtTripID",
                "6thStrtTripID",
                "7thStrtTripID",
                "1stEndTripID",
                "2ndEndTripID",
                "3rdEndTripID",
                "4thEndTripID",
                "5thEndTripID",
                "6thEndTripID",
                "7thEndTripID",
            ],
        },
        "leg_id": {
            "trips": "section_id",
        },
    }
    return (id_equivalences,)


@app.cell
def _(THATSData, asdict):
    """
    Description:
        Defines four functions that together implement Step 2 (ID reshuffling):
          - extract_ids: collects all unique ID values from a given table column.
          - generate_id_map: builds a (original -> zero-padded surrogate) mapping table.
          - replace_ids_in_cols: applies the mapping to one or more columns in a table.
          - reshuffle_ids: orchestrates the above for all five ID concepts.

    Input:
      - THATSData (class): the data container dataclass (needed for type hints).
      - asdict (function): converts a THATSData to a plain dict for iteration.

    Output:
      - reshuffle_ids (function): the top-level Step 2 pipeline function, exported to
            the main pipeline cell.
    """
    def extract_ids(
        df: pl.LazyFrame, cols: str | list[str], id_col: str
    ) -> pl.DataFrame:
        """
        Description:
            Extracts all unique non-null values of one or more ID columns from a LazyFrame,
            returning them as a single-column DataFrame.  Used to collect the universe of IDs
            that need to be remapped.

        Input:
          - df (pl.LazyFrame): the table to extract IDs from.
          - cols (str | list[str]): column name(s) in df that contain the IDs.
          - id_col (str): the output column name for the extracted IDs.

        Output:
          - (pl.DataFrame): a single-column DataFrame with unique non-null IDs.
        """
        return (
            df
            .select(pl.concat_list(pl.col(cols).cast(str)).explode().alias(id_col))
            .unique()
            .drop_nulls()
        )

    def generate_id_map(
        thats: THATSData, id_col: str, equivalences: dict[str, str | list[str]]
    ) -> pl.LazyFrame:
        """
        Description:
            Builds a mapping table from original ID values to zero-padded anonymous
            surrogate integers.  The mapping is consistent across all tables that
            share the same ID concept.

        Input:
          - thats (THATSData): the THATS data container to extract all original IDs from.
          - id_col (str): the semantic ID name (e.g. "hh_id", "person_id").
          - equivalences (dict): maps table names to the column name(s) in that table that
                hold this ID.

        Output:
          - (pl.LazyFrame): a two-column LazyFrame: original id_col and new <id_col>_new.
        """
        # Collect all unique IDs appearing in any of the tables for this ID concept
        all_ids = pl.concat([
            extract_ids(df, equivalences[df_name], id_col)
            for df_name, df in asdict(thats).items()
            if df_name in equivalences
        ]).unique()

        # Zero-pad to 8 digits — long enough for all household/person counts in THATS
        max_num_digits = 8

        # Assign a sequential index to each ID and format as a zero-padded string
        id_map = all_ids.with_columns(
            pl
            .row_index(name=f"{id_col}_new")  # sequential integer starting at 0
            .cast(str)
            .str.pad_start(max_num_digits, "0")  # e.g. "00000042"
        )

        return id_map

    def replace_ids_in_cols(
        df_name: str,
        df: pl.LazyFrame,
        id_map: pl.LazyFrame,
        id_col: str,
        new_id_col: str,
        df_id_cols: str | list[str],
    ) -> pl.LazyFrame:
        """
        Description:
            Replaces original ID values in one or more columns of a table with the
            corresponding anonymous surrogate values from id_map.

        Input:
          - df_name (str): the name of the table being processed (e.g. "hhs", "trips").
          - df (pl.LazyFrame): the table to update.
          - id_map (pl.LazyFrame): the mapping produced by generate_id_map.
          - id_col (str): the original ID column name (used as join key).
          - new_id_col (str): the new surrogate ID column name in id_map (e.g. "hh_id_new").
          - df_id_cols (str | list[str]): the column name(s) in df to replace.

        Output:
          - (pl.LazyFrame): the updated table with original IDs replaced by surrogates.
        """
        # Normalise to list for uniform iteration
        df_id_cols = [df_id_cols] if isinstance(df_id_cols, str) else df_id_cols

        for df_id_col in df_id_cols:
            # Ensure the ID column is string type before joining
            df = df.with_columns(pl.col(df_id_col).cast(str))
            # Left-join with the ID map to bring in the new surrogate ID
            df = (
                df
                .join(id_map, left_on=df_id_col, right_on=id_col, how="left")
                .with_columns(pl.col(new_id_col).alias(df_id_col).cast(str))
                .drop(new_id_col)  # remove the intermediate surrogate column
            )

            # For the trip_id columns in the activs table we keep the original column
            # name (e.g. "1stStrtTripID") rather than renaming to "trip_id"
            if (id_col, df_name) != ("trip_id", "activs"):
                df = df.rename({df_id_col: id_col})

        return df

    def reshuffle_ids(
        thats: THATSData, id_equivalences: dict[str, dict[str, str | list[str]]]
    ) -> THATSData:
        """
        Description:
            Iterates over all ID concepts (hh_id, person_id, app_id, trip_id, leg_id)
            and applies the full ID remapping pipeline to every table that contains
            each ID concept, ensuring referential integrity is preserved across tables.

        Input:
          - thats (THATSData): the THATS data container before ID anonymisation.
          - id_equivalences (dict): the global mapping of ID concept names to per-table
                column names (defined in the id_equivalences cell above).

        Output:
          - (THATSData): a new THATSData with all ID columns replaced by surrogates.
        """
        # Convert to a plain dict so we can update individual table LazyFrames
        new_thats = asdict(thats)

        for id_col, equivalences in id_equivalences.items():
            # Generate the (original -> surrogate) mapping for this ID concept
            new_id_col = f"{id_col}_new"
            id_map = generate_id_map(thats, id_col, equivalences)

            for df_name, df in new_thats.items():
                if df_name in equivalences:
                    # This table contains the current ID concept; replace it
                    df_id_cols = equivalences[df_name]
                    df = replace_ids_in_cols(
                        df_name, df, id_map, id_col, new_id_col, df_id_cols
                    )

                new_thats[df_name] = df  # store the (possibly updated) table back

        return THATSData(**new_thats)

    return (reshuffle_ids,)


@app.cell(hide_code=True)
def _():
    """
    Description: Render the Step 3 section heading — removing children from the dataset.

    Output:
      - (none): renders a Markdown heading in the notebook UI.
    """
    mo.md(r"""
    ## Step 3: Remove children from dataset
    """)
    return


@app.cell
def _(THATSData):
    """
    Description:
        Defines the remove_children pipeline step (Step 3).  Filters individuals,
        activities, and trips to retain only adults (age >= 18).

    Input:
      - THATSData (class): the data container dataclass.

    Output:
      - remove_children (function): the Step 3 pipeline function, exported to the
            main pipeline cell.
    """
    def remove_children(thats: THATSData) -> THATSData:
        """
        Description:
            Removes individuals under 18 years of age from all tables.  Children are
            excluded because the research focuses on adult travel behaviour.

        Input:
          - thats (THATSData): the THATS data container (after ID reshuffling).

        Output:
          - (THATSData): a new THATSData with children removed from indivs, activs, and trips.
        """
        # Keep only adult individuals (age >= 18 in the survey)
        adult_indivs = thats.indivs.filter(pl.col("THATS age") >= 18)

        # Extract the person_id and app_id columns of adults for semi-join filtering
        adult_person_ids = adult_indivs.select("person_id")  # used to filter activities
        adult_app_ids = adult_indivs.select("app_id")        # used to filter trips

        # Keep only activities and trips that belong to adult individuals
        adult_activs = thats.activs.join(adult_person_ids, on="person_id")
        adult_trips = thats.trips.join(adult_app_ids, on="app_id")

        # Households remain unchanged — they are not age-filtered
        return THATSData(thats.hhs, adult_indivs, adult_trips, adult_activs)

    return (remove_children,)


@app.cell(hide_code=True)
def _():
    """
    Description: Render the Step 4 section heading — replacing coordinates with census zone IDs.

    Output:
      - (none): renders a Markdown heading in the notebook UI.
    """
    mo.md(r"""
    ## Step 4: Replace latitudes and longitudes with Census Tracts / DAs
    """)
    return


@app.cell
def _():
    """
    Description:
        Defines the coordinate_columns mapping used in Step 4 to identify which
        (longitude, latitude) column pairs in each table need to be replaced with
        census zone IDs.

        Structure: { table_name: [ [lon_col, lat_col], ... ] }

        The households table has one coordinate pair (home location) and the trips
        table has four pairs (start/end of each trip leg, both at section and leg level).

    Output:
      - coordinate_columns (dict): table -> list of [longitude_col, latitude_col] pairs.
    """
    coordinate_columns = {
        "hhs": [
            ["THATS HomeLon", "THATS HomeLat"],
        ],
        "trips": [
            ["start_loc_lon", "start_loc_lat"],
            ["end_loc_lon", "end_loc_lat"],
            ["start_loc_lon_section", "start_loc_lat_section"],
            ["end_loc_lon_section", "end_loc_lat_section"],
        ],
    }
    return (coordinate_columns,)


@app.cell(hide_code=True)
def _():
    """
    Description: Render a link to the Statistics Canada 2021 Census boundary data source.

    Output:
      - (none): renders a Markdown link in the notebook UI.
    """
    mo.md(r"""
    - Dissemination Area (DA) & Census Tract (CT) boundaries: [link](https://www12.statcan.gc.ca/census-recensement/2021/geo/sip-pis/boundary-limites/index2021-eng.cfm?year=21)
    """)
    return


@app.cell
def _():
    """
    Description:
        Defines the path to the Statistics Canada 2021 Census boundary shapefiles and
        the coordinate reference system string used throughout the spatial join steps.

    Output:
      - CRS (str): "EPSG:4326" — the WGS-84 lat/lon CRS expected by GeoPandas and sjoin.
      - boundaries_dir (Path): directory containing the CT and DA boundary zip files.
    """
    # Directory where the downloaded Statistics Canada boundary shapefiles are stored
    boundaries_dir = project_root / "data/external/boundaries/canada"
    CRS = "EPSG:4326"  # standard geographic CRS for the coordinate columns in THATS
    return CRS, boundaries_dir


@app.cell
def _(boundaries_dir):
    """
    Description:
        Loads the Census Tract (CT) and Dissemination Area (DA) boundary shapefiles
        from Statistics Canada.  Only the unique zone identifier column is kept to
        minimise memory usage.  The identifier column is renamed to "zone_id" for
        consistency across both spatial join calls.

    Input:
      - boundaries_dir (Path): directory containing the boundary zip files.

    Output:
      - ct (gpd.GeoDataFrame): Census Tract polygons with a "zone_id" column (CTUID).
      - da (gpd.GeoDataFrame): Dissemination Area polygons with a "zone_id" column (DAUID).
    """
    # Census Tract boundaries — each CT has a unique CTUID (6-digit code)
    ct = gpd.read_file(
        boundaries_dir / "lct_000b21a_e.zip", columns=["CTUID"]
    ).rename(columns={"CTUID": "zone_id"})
    # Dissemination Area boundaries — finer resolution than CTs (8-digit DAUID)
    da = gpd.read_file(
        boundaries_dir / "lda_000b21a_e.zip", columns=["DAUID"]
    ).rename(columns={"DAUID": "zone_id"})
    return ct, da


@app.cell
def _(CRS, THATSData, asdict, coordinate_columns, ct, da):
    """
    Description:
        Defines the two Step 4 functions:
          - add_zone_id_from_coords: spatially joins a single (lon, lat) column pair
            to census zone polygons, appending the matched zone ID as a new column.
          - replace_coords_with_zones: applies add_zone_id_from_coords for both CT
            and DA to every coordinate column pair in every affected table, then drops
            the original coordinate columns.

    Input:
      - CRS (str): WGS-84 CRS string for GeoPandas reprojection.
      - THATSData (class): the data container dataclass.
      - asdict (function): converts THATSData to a plain dict.
      - coordinate_columns (dict): table -> list of (lon, lat) column pairs.
      - ct (gpd.GeoDataFrame): Census Tract boundary polygons.
      - da (gpd.GeoDataFrame): Dissemination Area boundary polygons.

    Output:
      - replace_coords_with_zones (function): the top-level Step 4 pipeline function,
            exported to the main pipeline cell.
    """
    def add_zone_id_from_coords(
        df: pl.LazyFrame, zones: pl.LazyFrame, lon: str, lat: str, zone_name: str
    ) -> pl.LazyFrame:
        """
        Description:
            Appends a census zone identifier column to a LazyFrame by spatially joining
            the coordinate columns (lon, lat) against a census boundary GeoDataFrame.
            The resulting zone ID column is named by replacing "Lon"/"lon" in the
            longitude column name with zone_name.

        Input:
          - df (pl.LazyFrame): the table containing coordinate columns.
          - zones (gpd.GeoDataFrame): census zone polygons with a "zone_id" column.
          - lon (str): name of the longitude column in df.
          - lat (str): name of the latitude column in df.
          - zone_name (str): suffix for the new column name (e.g. "CT" or "DA").

        Output:
          - (pl.LazyFrame): the table with a new zone ID column added.
        """
        # Derive the new column name by replacing "Lon"/"lon" in the longitude column name
        zone_name = lon.replace("Lon", zone_name).replace("lon", zone_name)
        # Estimate the UTM CRS for the zone boundaries to use metric distance during sjoin
        zones_crs = zones.estimate_utm_crs()

        # Materialise the lat/lon columns for GeoPandas processing
        coords = df.select(lon, lat).collect().to_pandas()
        # Build a GeoDataFrame of point geometries and reproject to the UTM CRS
        coords = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(coords[lon], coords[lat]), crs=CRS
        ).to_crs(zones_crs)

        # Spatially join each point to the containing census zone polygon
        zones = coords.sjoin(zones.to_crs(zones_crs), how="left").to_crs(CRS)
        # Extract the matched zone IDs as a Polars Series with the derived column name
        zones = pl.Series(zone_name, zones["zone_id"])

        return df.with_columns(zones)

    def replace_coords_with_zones(thats: THATSData) -> THATSData:
        """
        Description:
            Replaces raw latitude/longitude coordinate columns in the households and trips
            tables with Census Tract (CT) and Dissemination Area (DA) identifiers.
            After replacement, the original coordinate columns are dropped.

        Input:
          - thats (THATSData): the THATS data container (after children have been removed).

        Output:
          - (THATSData): a new THATSData where coordinates have been replaced by zone IDs.
        """
        new_thats = asdict(thats)

        for df_name, coord_cols in coordinate_columns.items():
            for col_pair in coord_cols:
                lon, lat = col_pair  # unpack the (longitude, latitude) column name pair

                new_thats[df_name] = (
                    new_thats[df_name]
                    # Add Census Tract ID column
                    .pipe(add_zone_id_from_coords, ct, lon, lat, "CT")
                    # Add Dissemination Area ID column
                    .pipe(add_zone_id_from_coords, da, lon, lat, "DA")
                    # Drop the original coordinate columns to remove precise locations
                    .drop(lon, lat)
                )

        return THATSData(**new_thats)

    return (replace_coords_with_zones,)


@app.cell(hide_code=True)
def _():
    """
    Description: Render the Step 5 section heading — writing anonymised data to Parquet files.

    Output:
      - (none): renders a Markdown heading in the notebook UI.
    """
    mo.md(r"""
    ## Step 5: Write results to file
    """)
    return


@app.cell
def _(
    drop_non_useful_columns,
    id_equivalences,
    raw_thats,
    remove_children,
    replace_coords_with_zones,
    reshuffle_ids,
):
    """
    Description:
        Runs the complete five-step anonymisation pipeline in sequence:
          Step 1: drop_non_useful_columns — remove PII and non-informative columns.
          Step 2: reshuffle_ids — replace all primary keys with anonymous surrogates.
          Step 3: remove_children — filter out individuals under 18.
          Step 4: replace_coords_with_zones — replace lat/lon with CT and DA IDs.
        The final thats object is displayed to verify the pipeline completed correctly.

    Input:
      - drop_non_useful_columns, reshuffle_ids, remove_children, replace_coords_with_zones:
            the four pipeline step functions defined in earlier cells.
      - id_equivalences (dict): ID concept -> table -> column mapping for Step 2.
      - raw_thats (THATSData): the four raw THATS tables loaded in the earlier cell.

    Output:
      - thats (THATSData): the fully anonymised data, displayed and ready for export.
    """
    thats = drop_non_useful_columns(raw_thats)      # Step 1: remove PII columns
    thats = reshuffle_ids(thats, id_equivalences)    # Step 2: anonymise primary keys
    thats = remove_children(thats)                   # Step 3: keep only adults
    thats = replace_coords_with_zones(thats)         # Step 4: replace coordinates with zones

    thats
    return (thats,)


@app.cell
def _(thats):
    """
    Description:
        Writes the four anonymised THATS tables to Parquet files in the raw data
        directory.  Using sink_parquet (streaming write) avoids materialising the full
        LazyFrame in memory before writing.

    Input:
      - thats (THATSData): the fully anonymised data container.

    Output:
      - output_dir (Path): the directory where the four Parquet files were written,
            exported so the verification cell below can read from it.
    """
    output_dir = project_root / "data/raw/THATS"  # output location for anonymised files

    # Write each table as a Parquet file; sink_parquet streams the LazyFrame to disk
    thats.hhs.sink_parquet(output_dir / "hhs.parquet")
    thats.indivs.sink_parquet(output_dir / "indivs.parquet")
    thats.trips.sink_parquet(output_dir / "trips.parquet")
    thats.activs.sink_parquet(output_dir / "activs.parquet")
    return (output_dir,)


@app.cell
def _(output_dir):
    """
    Description:
        Verification cell — reads the anonymised activities table back from disk and
        displays it so the user can confirm the anonymisation pipeline succeeded and the
        Parquet file was written correctly.

    Input:
      - output_dir (Path): directory containing the written Parquet files.

    Output:
      - (pl.DataFrame displayed in notebook; nothing returned to other cells)
    """
    pl.read_parquet(output_dir / "activs.parquet")
    return


@app.cell
def _():
    """
    Description:
        Empty placeholder cell — reserved for future additions to the notebook.
        Returns nothing and performs no computation.
    """
    return


if __name__ == "__main__":
    app.run()
