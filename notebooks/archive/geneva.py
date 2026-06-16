"""
Module: geneva.py

Description:
    A Marimo interactive notebook for exploratory data analysis (EDA) on the raw Geneva
    (TPG / LTDS) travel-survey dataset.

    The notebook explores:
      - Null, empty, and "NA" value patterns in key trip columns.
      - User behaviour (number of travel days per user, trip-mode distributions).
      - Construction of the Geneva locations GeoDataFrame by combining GTFS stops,
        canton subsectors, Swiss postal codes, and French postal boundaries.
      - Matching raw trip origin/destination strings to internal loc_ids.
      - Geographic visualisation of stop affluence (how often each location is used).

    This notebook is exploratory and is not required for the main training pipeline.

Dependencies:
    - activitygraphs library (GenevaData, build_geneva_locations, match_loc_ids)
    - Marimo, Polars, GeoPandas, Altair
"""

import marimo

__generated_with = "0.17.7"
app = marimo.App(width="full")


@app.cell
def _():
    """
    Description: Entry cell that imports all top-level libraries used throughout the
    notebook: pathlib for path manipulation, Marimo for reactive UI, Polars for fast
    DataFrame operations, and Polars selectors for column-type-based filtering.

    Input:
      - (none)

    Output:
      - Path (type): ``pathlib.Path`` class for constructing file paths.
      - cs (module): ``polars.selectors`` for selecting columns by dtype in expressions.
      - mo (module): Marimo module; exported for use by all subsequent cells.
      - pl (module): Polars DataFrame library.
    """
    from pathlib import Path

    import marimo as mo
    import polars as pl
    import polars.selectors as cs

    return Path, cs, mo, pl


@app.cell
def _(Path, mo):
    """
    Description: Resolve the project root directory (one level above the notebooks folder)
    and load the project configuration from the YAML file found there.

    Input:
      - Path (type): pathlib.Path for building the project root path.
      - mo (module): Marimo (provides ``notebook_dir()`` to locate the current notebook).

    Output:
      - cfg: the loaded configuration object with data paths and input file names.
      - project_root (Path): absolute path to the project root directory; used by later
        cells to construct absolute paths to raw data files.
    """
    from activitygraphs.config import load_config

    project_root = Path(mo.notebook_dir().parent)
    cfg = load_config(project_root)
    return cfg, project_root


@app.cell(hide_code=True)
def _(mo):
    """
    Description: Display a Marimo Markdown section heading for the "Exploratory Data
    Analysis" portion of the notebook. Subsequent cells investigate null values, user
    behaviour, and location matching. Pure presentation; no computation.

    Input:
      - mo (module): Marimo (for ``mo.md``).

    Output:
      - (none): renders the section heading in the notebook UI.
    """
    mo.md(r"""
    ## Exploratory Data Analysis
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    """
    Description: Display a Marimo Markdown introductory note indicating that the next
    cell reads the raw Geneva trip data into memory. Pure presentation; no computation.

    Input:
      - mo (module): Marimo (for ``mo.md``).

    Output:
      - (none): renders the introductory note in the notebook UI.
    """
    mo.md(r"""
    We read the raw data into memory.
    """)
    return


@app.cell
def _(cfg, pl, project_root):
    """
    Description: Load the raw Geneva (TPG) trip records from a Parquet file into a Polars
    DataFrame. Each row represents one GPS-tracked trip segment recorded by the TPG survey.

    Input:
      - cfg: project configuration (provides raw data directory path and input file name).
      - pl (module): Polars for reading Parquet.
      - project_root (Path): root directory of the project.

    Output:
      - df (pl.DataFrame): the raw trips DataFrame with columns for user ID, departure and
        arrival locations, mode, route line, date, and activity purposes.
    """
    # Build the absolute path to the raw trips Parquet file using paths from the config.
    raw_path = project_root / cfg.data.paths.raw / cfg.data.inputs.raw_trips

    # df: the full raw trips DataFrame; each row is one GPS-tracked trip segment.
    df = pl.read_parquet(raw_path)
    df
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    """
    Description: Display a Marimo Markdown sub-heading for the "Null and NA values"
    investigation section. The cells that follow check for true nulls, empty strings,
    and the literal string "NA" across all trip columns. Pure presentation.

    Input:
      - mo (module): Marimo (for ``mo.md``).

    Output:
      - (none): renders the sub-heading in the notebook UI.
    """
    mo.md(r"""
    ### Null and NA values
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    """
    Description: Display a Marimo Markdown note prompting the investigation of null values
    in the raw trips DataFrame. The next cell calls ``df.null_count()``. Pure presentation.

    Input:
      - mo (module): Marimo (for ``mo.md``).

    Output:
      - (none): renders the prompt text in the notebook UI.
    """
    mo.md(r"""
    Let's first investigate the dataframe for null values
    """)
    return


@app.cell
def _(df):
    """
    Description: Count the number of true null values in each column of the raw trips
    DataFrame as a first data quality check. LTDS/TPG data often uses sentinel strings
    (e.g. "NA", "") rather than null, so this check may return zeros even for columns
    with missing values.

    Input:
      - df (pl.DataFrame): the raw trips DataFrame.

    Output:
      - (none): renders the null count row in the notebook UI.
    """
    df.null_count()
    return


@app.cell(hide_code=True)
def _(mo):
    """
    Description: Display a Marimo Markdown note summarising that the DataFrame has no
    true null values, and announcing that the next check will look at empty strings across
    string columns. Pure presentation; no computation.

    Input:
      - mo (module): Marimo (for ``mo.md``).

    Output:
      - (none): renders the summary note in the notebook UI.
    """
    mo.md(r"""
    We see that no values are null, but let's investigate empty and NA values in the string columns.

    Empty values:
    """)
    return


@app.cell
def _(cs, df):
    """
    Description: Count empty-string values across all string columns. TPG survey data
    sometimes stores missing values as empty strings rather than nulls, so this check
    complements the null_count above.

    Input:
      - cs (module): Polars column selectors (``polars.selectors``).
      - df (pl.DataFrame): the raw trips DataFrame.

    Output:
      - (none): renders a one-row DataFrame showing the empty-string count per string column.
    """
    df.select((cs.string() == "").sum())
    return


@app.cell(hide_code=True)
def _(mo):
    """
    Description: Display a Marimo Markdown label announcing that the next cell checks for
    the literal string "NA" across all string columns. Pure presentation; no computation.

    Input:
      - mo (module): Marimo (for ``mo.md``).

    Output:
      - (none): renders the label in the notebook UI.
    """
    mo.md(r"""
    `"NA"` values:
    """)
    return


@app.cell
def _(cs, df):
    """
    Description: Count the literal string ``"NA"`` across all string columns. TPG survey
    data uses ``"NA"`` to indicate "not available" or "not applicable" in location and
    route columns, which is distinct from both null and empty string.

    Input:
      - cs (module): Polars column selectors.
      - df (pl.DataFrame): the raw trips DataFrame.

    Output:
      - (none): renders a one-row DataFrame showing the ``"NA"`` string count per column.
    """
    df.select((cs.string() == "NA").sum())
    return


@app.cell(hide_code=True)
def _(mo):
    """
    Description: Display a Marimo Markdown note summarising the finding that only a small
    number of rows have null or empty values in the ``jour_depart``, ``motif_depart``,
    ``motif_arrivee``, and ``date`` columns, and introducing the next investigation cell
    that filters and inspects those rows. Pure presentation.

    Input:
      - mo (module): Marimo (for ``mo.md``).

    Output:
      - (none): renders the summary note in the notebook UI.
    """
    mo.md(r"""
    We see that there are few null or empty values in the `jour_depart`, `motif_depart`, `motif_arrivee`, and `date` columns. Let's take a look at them.
    """)
    return


@app.cell
def _(df, pl):
    """
    Description: Extract all trip rows where at least one of the critical columns
    (``jour_depart``, ``motif_depart``, ``motif_arrivee``, ``date``) contains an empty
    string. These rows have incomplete activity-purpose or date information and need to
    be investigated before deciding whether to remove them or the affected users.

    Input:
      - df (pl.DataFrame): the raw trips DataFrame.
      - pl (module): Polars for filtering.

    Output:
      - na_rows (pl.DataFrame): subset of rows with at least one empty value in the
        four critical columns listed above.
    """
    _cols = ["jour_depart", "motif_depart", "motif_arrivee", "date"]

    na_rows = df.filter(pl.any_horizontal(pl.col(*_cols) == ""))
    na_rows
    return (na_rows,)


@app.cell
def _(df, na_rows, pl):
    """
    Description: For each user affected by empty-value rows, compare the total number of
    trips against the number of problematic trips. This helps decide whether to:
      (a) drop only the affected trips (if the user still has sufficient valid data), or
      (b) remove the entire user (if most of their trips are affected).

    Input:
      - df (pl.DataFrame): the full raw trips DataFrame.
      - na_rows (pl.DataFrame): the subset of rows with empty critical-column values.
      - pl (module): Polars for grouping and joining.

    Output:
      - (none): renders a joined DataFrame with columns ``id_utilisateur``, ``num_trips``
        (total), and ``num_na_trips`` in the notebook UI.
    """
    # List of all user IDs that have at least one row with empty/NA values
    na_users = na_rows["id_utilisateur"].unique().to_list()

    # Count the total number of trips (all rows) for each affected user
    _num_trips_per_na_user = (
        df.filter(pl.col("id_utilisateur").is_in(na_users)).group_by("id_utilisateur").agg(pl.len().alias("num_trips"))
    )
    # Count how many of those trips are the problematic empty/NA ones
    _num_na_trips_per_na_user = na_rows.group_by("id_utilisateur").agg(pl.len().alias("num_na_trips"))

    # Join to compare total trips vs NA trips per user; helps decide if user should be removed
    _num_trips_per_na_user.join(_num_na_trips_per_na_user, on="id_utilisateur")
    return


@app.cell(hide_code=True)
def _(mo):
    """
    Description: Display a Marimo Markdown note concluding the per-user NA investigation:
    since only a small number of users are affected and not all their trips are problematic,
    the recommended cleaning decision is to remove these users entirely from the dataset.
    Pure presentation; no computation.

    Input:
      - mo (module): Marimo (for ``mo.md``).

    Output:
      - (none): renders the conclusion note in the notebook UI.
    """
    mo.md(r"""
    So not for a given user, not all their trips necessarily have NA / empty values in the `jour_depart`, `motif_depart`, `motif_arrivee`, and `date` columns. Therefore, given how few users have NA / empty values in these columns, we suggest removing these users and their trips from the dataset.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    """
    Description: Display a Marimo Markdown transition note introducing the next phase of
    the null/NA investigation — specifically looking at the location and mode columns
    (``lieu_depart_trajet``, ``mode``, ``ligne_trajet``, ``lieu_arrivee_trajet``).
    Pure presentation; no computation.

    Input:
      - mo (module): Marimo (for ``mo.md``).

    Output:
      - (none): renders the transition note in the notebook UI.
    """
    mo.md(r"""
    Let's now move to the NA and empty values in `lieu_depart_trajet`, `mode`, `ligne_trajet`, and `lieu_arrivee_trajet`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    """
    Description: Display a Marimo Markdown note explaining the cleaning decision for the
    location columns: ``"NA"`` locations are kept as a valid category (not treated as
    missing), and the single empty ``lieu_arrivee_trajet`` value will be imputed to
    ``"NA"``. Pure presentation; no computation.

    Input:
      - mo (module): Marimo (for ``mo.md``).

    Output:
      - (none): renders the cleaning-decision note in the notebook UI.
    """
    mo.md(r"""
    For now, we will treat `"NA"` departure and arrival locations as existing categories, and will not delete them from the dataset.

    There's a single empty value in `lieu_arrivee_trajet`, which we will impute to `"NA"`.
    """)
    return


@app.cell
def _(df):
    """
    Description: Inspect the single trip row that has an empty ``lieu_arrivee_trajet``
    (arrival location) value. This is the only empty arrival location in the dataset
    and will be imputed to ``"NA"`` during cleaning.

    Input:
      - df (pl.DataFrame): the raw trips DataFrame.

    Output:
      - (none): renders the single-row DataFrame in the notebook UI.
    """
    df.filter(lieu_arrivee_trajet="")
    return


@app.cell
def _():
    """
    Description: Define lists of columns by their missing-value representation:
      - ``na_columns``: columns where ``"NA"`` means "unknown location" — kept as a
        valid category rather than treated as missing.
      - ``empty_columns``: columns where ``""`` means unknown — need special imputation.
    These lists are used by the cleaning functions in the activitygraphs pipeline.

    Input:
      - (none)

    Output:
      - (none): variables are local to the cell (documentation only, no exports).
    """
    # Columns where the literal string "NA" means "not available / unknown location"
    # and should be kept as a valid category rather than treated as missing data.
    na_columns = ["lieu_depart_trajet", "lieu_arrivee_trajet"]

    # Columns where an empty string ("") means the value is unknown or not applicable.
    # These need special imputation logic before modelling.
    empty_columns = ["mode", "ligne_trajet"]
    return


@app.cell(hide_code=True)
def _(mo):
    """
    Description: Display a Marimo Markdown note explaining the imputation decision for
    empty ``mode`` values: since they represent ~13% of records they cannot be dropped,
    so they are encoded as a new ``mode_unknown`` category. Pure presentation.

    Input:
      - mo (module): Marimo (for ``mo.md``).

    Output:
      - (none): renders the imputation rationale in the notebook UI.
    """
    mo.md(r"""
    For the `mode` column, it is not immediately obvious what an empty mode means. It is not possible to ignore it, because this mode represents 13% of records. We assume it to be unknown, perhaps because the user has not entered it, and therefore encode it as its own `mode_unknown` category.
    """)
    return


@app.cell
def _(df):
    """
    Description: Inspect the first 3 rows with an empty ``mode`` string to understand
    the context in which mode is missing (e.g. which departure/arrival locations, what
    purpose). Helps decide the appropriate imputation strategy.

    Input:
      - df (pl.DataFrame): the raw trips DataFrame.

    Output:
      - (none): renders the first 3 empty-mode rows in the notebook UI.
    """
    df.filter(mode="").head(3)
    return


@app.cell
def _(df, pl):
    """
    Description: Compute the frequency distribution of transport modes in the raw dataset,
    normalised to proportions. This quantifies how common each mode (bus, tram, walk, etc.)
    is and, importantly, how large the empty-mode category is (approximately 13% of records).

    Input:
      - df (pl.DataFrame): the raw trips DataFrame.
      - pl (module): Polars for value_counts and formatting.

    Output:
      - (none): renders the mode frequency table in the notebook UI.
    """
    df["mode"].value_counts(sort=True, normalize=True).with_columns(pl.format("{}", pl.col("proportion")))
    return


@app.cell(hide_code=True)
def _(mo):
    """
    Description: Display a Marimo Markdown note explaining the imputation rules for
    ``ligne_trajet`` (the route line column):
      - For non-bus/tram/boat modes, an empty ``ligne_trajet`` is imputed to ``"NA"``
        (route line is not applicable).
      - For bus, tram, or boat modes with an empty ``ligne_trajet``, it is imputed to
        ``mode_unknown`` (line should be known but was not recorded).
    Pure presentation; no computation.

    Input:
      - mo (module): Marimo (for ``mo.md``).

    Output:
      - (none): renders the imputation rules in the notebook UI.
    """
    mo.md(r"""
    Finally, the `ligne_trajet` columns is only present if it makes sense for the mode and it is available data to the TPG, i.e.: `mode_bus`, `mode_tramway` and `mode_bateau_navette`. If another mode is taken, we impute an empty `ligne_trajet` to `"NA"`.

    In the mode is `mode_bus`, `mode_tramway` or `mode_bateau_navette` and `ligne_trajet` is empty, we impute it to `mode_unknown`.
    """)
    return


@app.cell
def _(df):
    """
    Description: Display all boat-ferry (``mode_bateau_navette``) trip rows for inspection.
    This is a rare mode in the dataset; checking it helps verify that the ``ligne_trajet``
    (route line) column is populated correctly for boat trips.

    Input:
      - df (pl.DataFrame): the raw trips DataFrame.

    Output:
      - (none): renders the filtered boat-trip rows in the notebook UI.
    """
    df.filter(mode="mode_bateau_navette")
    return


@app.cell
def _(cfg, project_root):
    """
    Description: Run the private data-loading and null-handling steps from the
    ``activitygraphs.data.geneva`` module to verify that boat-trip rows have their
    empty ``ligne_trajet`` values correctly imputed after cleaning.

    Input:
      - cfg: project configuration.
      - project_root (Path): root directory.

    Output:
      - n (pl.DataFrame): the cleaned trips DataFrame after null imputation; exported for
        the Altair histogram cell below.
    """
    from activitygraphs.data.geneva import _handle_null_values, _read_raw_data

    r = _read_raw_data(cfg.data, project_root=project_root)
    n = _handle_null_values(r)
    n.filter(mode="mode_bateau_navette")
    return (n,)


@app.cell
def _(n, pl):
    """
    Description: Plot a histogram of trips-per-user to understand how active users are
    in the cleaned dataset. Enables the VegaFusion data transformer for efficient
    Altair rendering of large DataFrames.

    Input:
      - n (pl.DataFrame): the cleaned trips DataFrame (after null imputation).
      - pl (module): Polars for the group-by aggregation.

    Output:
      - (none): renders the histogram in the notebook UI.
    """
    import altair as alt

    alt.data_transformers.enable("vegafusion")  # needed for large DataFrames in Altair

    n.group_by("id_utilisateur").agg(pl.len())["len"].plot.hist()
    return


@app.cell(hide_code=True)
def _(mo):
    """
    Description: Display a Marimo Markdown sub-heading for the "Investigating users and
    their behaviour" section. Subsequent cells count unique users and plot the distribution
    of travel days per user. Pure presentation; no computation.

    Input:
      - mo (module): Marimo (for ``mo.md``).

    Output:
      - (none): renders the sub-heading in the notebook UI.
    """
    mo.md(r"""
    ### Investigating users and their behaviour
    """)
    return


@app.cell
def _(df, mo):
    """
    Description: Compute and display the total number of unique users in the raw dataset
    as a simple summary statistic.

    Input:
      - df (pl.DataFrame): the raw trips DataFrame.
      - mo (module): Marimo (for ``mo.md``).

    Output:
      - (none): renders a Markdown string with the user count in the notebook UI.
    """
    _n_users = df.n_unique("id_utilisateur")  # count distinct user IDs

    mo.md(f"This dataset has {_n_users} unique users")
    return


@app.cell
def _(df, pl):
    """
    Description: Compute and display the distribution of the number of travel days per user.
    For each user, counts how many unique departure days (``jour_depart``) they have, then
    shows how many users recorded 1 day, 2 days, etc. Helps understand data collection
    duration across participants.

    Input:
      - df (pl.DataFrame): the raw trips DataFrame with a ``jour_depart`` column.
      - pl (module): Polars for grouping and aggregation.

    Output:
      - (none): renders a bar chart of travel-day-count distribution in the notebook UI.
    """
    _num_days_per_user = (
        df.group_by("id_utilisateur")
        .agg(pl.col("jour_depart").n_unique())["jour_depart"]  # count unique days per user
        .value_counts()  # count how many users have each day count
    )
    _num_days_per_user.plot.bar(x="jour_depart:N", y="count")
    return


@app.cell(hide_code=True)
def _(mo):
    """
    Description: Display a Marimo Markdown top-level heading for the "GTFS to NetworkX
    construction" section. Subsequent cells load geographic boundary files, filter GTFS
    stops, and build the unified locations GeoDataFrame used for graph construction.
    Pure presentation; no computation.

    Input:
      - mo (module): Marimo (for ``mo.md``).

    Output:
      - (none): renders the section heading in the notebook UI.
    """
    mo.md(r"""
    # GTFS to NetworkX construction
    """)
    return


@app.cell
def _():
    """
    Description: Import GeoPandas for reading spatial boundary shapefiles and performing
    geographic operations (point creation, spatial joins, CRS reprojection).

    Input:
      - (none)

    Output:
      - gpd (module): the ``geopandas`` module.
    """
    import geopandas as gpd

    return (gpd,)


@app.cell(hide_code=True)
def _(mo):
    """
    Description: Display a Marimo Markdown reference list with download links for all
    geographic datasets used in this section: Geneva subsectors, Swiss GTFS timetable,
    Swiss postcode/locality boundaries, and French postal-code boundaries. Pure presentation.

    Input:
      - mo (module): Marimo (for ``mo.md``).

    Output:
      - (none): renders the dataset reference list in the notebook UI.
    """
    mo.md(r"""
    Links to various datasets
    - Geneva subsectors: [home](https://sitg.ge.ch/donnees/geo-girec), [file](https://ge.ch/sitg/geodata/SITG/OPENDATA/GEO_GIREC-SHP.zip)
    - Swiss 2022 timetable: [home](https://archive.opentransportdata.swiss/timetable_gtfs_archive.htm), [file](https://archive.opentransportdata.swiss/timetable_gtfs/timetable-2022-gtfs2020/GTFS_FP2022_2022-12-07_04-15.zip)
    - Swiss postcode and locality boundaries: [home](https://www.swisstopo.admin.ch/en/official-directory-of-towns-and-cities), [file](https://data.geo.admin.ch/ch.swisstopo-vd.ortschaftenverzeichnis_plz/ortschaftenverzeichnis_plz/ortschaftenverzeichnis_plz_2056.shp.zip)
    - French postcode boundaries: [home](https://www.data.gouv.fr/datasets/fond-de-carte-des-codes-postaux/), [file](https://www.data.gouv.fr/api/1/datasets/r/029656d6-0fcd-48ec-917d-d511e1f36ff6)
    """)
    return


@app.cell
def _(project_root):
    """
    Description: Build absolute Path objects for all required input geographic data files:
    GTFS timetable stops, Geneva subsectors, Swiss admin boundaries, Swiss postal codes,
    and French postal codes. These paths are used by subsequent cells that load the data.

    Input:
      - project_root (Path): the root directory of the project.

    Output:
      - french_path (Path): path to the French postal-code boundary shapefile.
      - postcodes_path (Path): path to the Swiss postal-code/locality shapefile directory.
      - stops_path (Path): path to the GTFS ``stops.txt`` file.
      - subsectors_path (Path): path to the Geneva subsector (GEO_GIREC) shapefile.
      - swiss_boundaries_path (Path): path to the Swiss national boundary shapefile.
    """
    # Path to the 2022 Swiss national GTFS timetable directory
    gtfs_path = project_root / "data/raw/gtfs/gtfs_2022_switzerland"
    # GTFS stops file: contains stop_id, stop_name, stop_lat, stop_lon, etc.
    stops_path = gtfs_path / "stops.txt"

    # Root directory for all geographic boundary shapefiles
    boundaries_path = project_root / "data/raw/boundaries"

    # Geneva canton subsectors (GEO_GIREC) — fine-grained internal zones
    subsectors_path = boundaries_path / "GEO_GIREC-SHP.shp"
    # Swiss national administrative boundaries (cantons, municipalities)
    swiss_boundaries_path = boundaries_path / "swissboundaries3d_2025-04_2056_5728.shp"
    # Swiss postal code and locality boundaries (LV95 coordinate system)
    postcodes_path = boundaries_path / "ortschaftenverzeichnis_plz_2056.shp/AMTOVZ_SHP_LV95"
    # French postal code boundaries (used for cross-border trips)
    french_path = boundaries_path / "code_postaux_V5.shp"
    return (
        french_path,
        postcodes_path,
        stops_path,
        subsectors_path,
        swiss_boundaries_path,
    )


@app.cell
def _(french_path, gpd, pl, postcodes_path, stops_path, subsectors_path):
    """
    Description: Load all geographic boundary data and GTFS stops into memory.
    All GeoDataFrames are reprojected to WGS-84 (EPSG:4326) for compatibility.

    Input:
      - french_path (Path): French postal-code shapefile path.
      - gpd (module): GeoPandas.
      - pl (module): Polars (for reading the GTFS CSV).
      - postcodes_path (Path): Swiss postal-code shapefile directory path.
      - stops_path (Path): GTFS stops.txt path.
      - subsectors_path (Path): Geneva subsector shapefile path.

    Output:
      - french_gdf (gpd.GeoDataFrame): French municipal postal-code boundaries.
      - localities_gdf (gpd.GeoDataFrame): Swiss locality polygon boundaries.
      - postcodes_gdf (gpd.GeoDataFrame): Swiss postal-code polygon boundaries.
      - stops (pl.DataFrame): GTFS stop records (stop_id, stop_name, lat, lon, etc.).
      - subsectors_gdf (gpd.GeoDataFrame): Geneva canton subsector polygons.
    """
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
def _(gpd, stops, subsectors_gdf, swiss_boundaries_path):
    """
    Description: Extract the polygon of the canton of Geneva from the Swiss national
    boundary file, filter GTFS stops to those within Geneva, and display both on an
    interactive map overlaid with the subsector polygons.

    Input:
      - gpd (module): GeoPandas.
      - stops (pl.DataFrame): GTFS stop records with lat/lon columns.
      - subsectors_gdf (gpd.GeoDataFrame): Geneva subsector polygons.
      - swiss_boundaries_path (Path): path to the Swiss canton boundaries shapefile.

    Output:
      - (none): renders the interactive Folium map in the notebook UI. All local variables
        (geneva_gdf, geneva_shape, stops_gdf, geneva_stops_gdf) are cell-local.
    """
    geneva_gdf = (
        gpd
        .read_file(swiss_boundaries_path, layer="swissBOUNDARIES3D_1_5_TLM_KANTONSGEBIET")
        .to_crs("EPSG:4326")
        .query("NAME == 'Genève'")
    )
    geneva_shape = geneva_gdf.iloc[0]["geometry"]

    stops_gdf = gpd.GeoDataFrame(
        stops.to_pandas(), geometry=gpd.points_from_xy(stops["stop_lon"], stops["stop_lat"], crs="EPSG:4326")
    )
    geneva_stops_gdf = stops_gdf[stops_gdf.intersects(geneva_shape)]

    _m = subsectors_gdf.explore()
    geneva_stops_gdf.explore(m=_m, color="orange")
    return


@app.cell
def _(french_gdf, localities_gdf, pl, postcodes_gdf, stops, subsectors_gdf):
    """
    Description: Build the unified locations GeoDataFrame by combining all geographic
    data sources: GTFS PT stops, Geneva canton subsectors, Swiss postal codes/localities,
    and French postal boundaries. Also creates a non-spatial Polars DataFrame version
    for tabular analysis.

    Input:
      - french_gdf (gpd.GeoDataFrame): French postal-code boundaries.
      - localities_gdf (gpd.GeoDataFrame): Swiss locality boundaries.
      - pl (module): Polars for the non-spatial copy.
      - postcodes_gdf (gpd.GeoDataFrame): Swiss postal-code boundaries.
      - stops (pl.DataFrame): GTFS stop records.
      - subsectors_gdf (gpd.GeoDataFrame): Geneva subsector polygons.

    Output:
      - locations_df (pl.DataFrame): tabular version of all locations (no geometry).
      - locations_gdf (gpd.GeoDataFrame): full locations GeoDataFrame with a ``loc_id``
        column and geometry (Points for PT stops, Polygons for zones).
    """
    from activitygraphs.data.geneva import build_geneva_locations

    locations_gdf = build_geneva_locations(stops, subsectors_gdf, postcodes_gdf, localities_gdf, french_gdf)
    locations_df = pl.DataFrame(locations_gdf.drop(columns=["geometry"]))  # strip geometry for tabular use

    locations_gdf
    return locations_df, locations_gdf


@app.cell
def _(df, locations_df, stops):
    """
    Description: Match the raw departure and arrival location strings in the trip records
    (which use GTFS stop IDs or postal-code strings) to the internal ``loc_id`` keys in
    the locations DataFrame. This is a key preprocessing step before graph construction.

    Input:
      - df (pl.DataFrame): the raw trips DataFrame with ``lieu_depart_trajet`` and
        ``lieu_arrivee_trajet`` location string columns.
      - locations_df (pl.DataFrame): the tabular locations table with ``loc_id`` keys.
      - stops (pl.DataFrame): the GTFS stop records (for matching stop IDs).

    Output:
      - trips_df (pl.DataFrame): the trips DataFrame with ``dep_loc_id`` and ``arr_loc_id``
        columns added, containing the internal location identifiers.
    """
    from activitygraphs.data.geneva import match_loc_ids

    trips_df = match_loc_ids(df, locations_df, stops)
    trips_df
    return (trips_df,)


@app.cell
def _(locations_gdf, trips_df):
    """
    Description: Display an interactive map showing how frequently each location is used
    as a trip departure point (location affluence). Darker/larger markers indicate
    locations used more often. Helps identify major origin zones.

    Input:
      - locations_gdf (gpd.GeoDataFrame): the locations GeoDataFrame with geometry.
      - trips_df (pl.DataFrame): the matched trips DataFrame with ``dep_loc_id`` column.

    Output:
      - explore_location_affluence (callable): the mapping helper; exported for the
        arrival-location map in the next cell.
    """
    from activitygraphs.network import explore_location_affluence

    explore_location_affluence(trips_df, locations_gdf, "dep_loc_id")
    return (explore_location_affluence,)


@app.cell
def _(explore_location_affluence, locations_gdf, trips_df):
    """
    Description: Display an interactive map showing how frequently each location is used
    as a trip arrival point. Mirrors the departure-location map for comparison.

    Input:
      - explore_location_affluence (callable): the mapping helper from the previous cell.
      - locations_gdf (gpd.GeoDataFrame): the locations GeoDataFrame with geometry.
      - trips_df (pl.DataFrame): the matched trips DataFrame with ``arr_loc_id`` column.

    Output:
      - (none): renders the arrival-location affluence map in the notebook UI.
    """
    explore_location_affluence(trips_df, locations_gdf, "arr_loc_id")
    return


@app.cell
def _():
    """
    Description: Empty placeholder cell at the end of the notebook. No computation is
    performed here; Marimo requires at least one cell after the last content cell.

    Input:
      - (none)

    Output:
      - (none)
    """
    return


if __name__ == "__main__":
    app.run()
