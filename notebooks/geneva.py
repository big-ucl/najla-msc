import marimo

__generated_with = "0.17.7"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import polars.selectors as cs
    from pathlib import Path
    return Path, cs, mo, pl


@app.cell
def _(Path, mo):
    from activitygraphs.config import load_config

    project_root = Path(mo.notebook_dir().parent)
    cfg = load_config(project_root)
    return cfg, project_root


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exploratory Data Analysis
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We read the raw data into memory.
    """)
    return


@app.cell
def _(cfg, pl, project_root):
    raw_path = project_root / cfg.data.paths.raw / cfg.data.files.raw_trips

    df = pl.read_parquet(raw_path)
    df
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Null and NA values
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's first investigate the dataframe for null values
    """)
    return


@app.cell
def _(df):
    df.null_count()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We see that no values are null, but let's investigate empty and NA values in the string columns.

    Empty values:
    """)
    return


@app.cell
def _(cs, df):
    df.select((cs.string() == "").sum())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `"NA"` values:
    """)
    return


@app.cell
def _(cs, df):
    df.select((cs.string() == "NA").sum())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We see that there are few null or empty values in the `jour_depart`, `motif_depart`, `motif_arrivee`, and `date` columns. Let's take a look at them.
    """)
    return


@app.cell
def _(df, pl):
    _cols = ["jour_depart", "motif_depart", "motif_arrivee", "date"]

    na_rows = df.filter(pl.any_horizontal(pl.col(*_cols) == ""))
    na_rows
    return (na_rows,)


@app.cell
def _(df, na_rows, pl):
    na_users = na_rows["id_utilisateur"].unique().to_list()

    _num_trips_per_na_user = (
        df.filter(pl.col("id_utilisateur").is_in(na_users)).group_by("id_utilisateur").agg(pl.len().alias("num_trips"))
    )
    _num_na_trips_per_na_user = na_rows.group_by("id_utilisateur").agg(pl.len().alias("num_na_trips"))

    _num_trips_per_na_user.join(_num_na_trips_per_na_user, on="id_utilisateur")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    So not for a given user, not all their trips necessarily have NA / empty values in the `jour_depart`, `motif_depart`, `motif_arrivee`, and `date` columns. Therefore, given how few users have NA / empty values in these columns, we suggest removing these users and their trips from the dataset.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's now move to the NA and empty values in `lieu_depart_trajet`, `mode`, `ligne_trajet`, and `lieu_arrivee_trajet`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For now, we will treat `"NA"` departure and arrival locations as existing categories, and will not delete them from the dataset.

    There's a single empty value in `lieu_arrivee_trajet`, which we will impute to `"NA"`.
    """)
    return


@app.cell
def _(df):
    df.filter(lieu_arrivee_trajet="")
    return


@app.cell
def _():
    na_columns = ["lieu_depart_trajet", "lieu_arrivee_trajet"]
    empty_columns = ["mode", "ligne_trajet"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For the `mode` column, it is not immediately obvious what an empty mode means. It is not possible to ignore it, because this mode represents 13% of records. We assume it to be unknown, perhaps because the user has not entered it, and therefore encode it as its own `mode_unknown` category.
    """)
    return


@app.cell
def _(df):
    df.filter(mode="").head(3)
    return


@app.cell
def _(df, pl):
    df["mode"].value_counts(sort=True, normalize=True).with_columns(pl.format("{}", pl.col("proportion")))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally, the `ligne_trajet` columns is only present if it makes sense for the mode and it is available data to the TPG, i.e.: `mode_bus`, `mode_tramway` and `mode_bateau_navette`. If another mode is taken, we impute an empty `ligne_trajet` to `"NA"`.

    In the mode is `mode_bus`, `mode_tramway` or `mode_bateau_navette` and `ligne_trajet` is empty, we impute it to `mode_unknown`.
    """)
    return


@app.cell
def _(df):
    df.filter(mode="mode_bateau_navette")
    return


@app.cell
def _(cfg, project_root):
    from activitygraphs.data.geneva import _read_raw_data, _handle_null_values

    r = _read_raw_data(cfg.data, project_root=project_root)
    n = _handle_null_values(r)
    n.filter(mode="mode_bateau_navette")
    return (n,)


@app.cell
def _(n, pl):
    import altair as alt

    alt.data_transformers.enable("vegafusion")


    n.group_by("id_utilisateur").agg(pl.len())["len"].plot.hist()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Investigating users and their behaviour
    """)
    return


@app.cell
def _(df, mo):
    _n_users = df.n_unique("id_utilisateur")

    mo.md(f"This dataset has {_n_users} unique users")
    return


@app.cell
def _(df, pl):
    _num_days_per_user = df.group_by("id_utilisateur").agg(pl.col("jour_depart").n_unique())["jour_depart"].value_counts()
    _num_days_per_user.plot.bar(x="jour_depart:N", y="count")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # GTFS to NetworkX construction
    """)
    return


@app.cell
def _():
    import geopandas as gpd
    return (gpd,)


@app.cell(hide_code=True)
def _(mo):
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
    gtfs_path = project_root / "data/raw/gtfs/gtfs_2022_switzerland"
    stops_path = gtfs_path / "stops.txt"
    boundaries_path = project_root / "data/raw/boundaries"
    subsectors_path = boundaries_path / "GEO_GIREC-SHP.shp"
    swiss_boundaries_path = boundaries_path / "swissboundaries3d_2025-04_2056_5728.shp"
    postcodes_path = boundaries_path / "ortschaftenverzeichnis_plz_2056.shp/AMTOVZ_SHP_LV95"
    french_path = boundaries_path / "codes_postaux_V5"
    return (
        french_path,
        postcodes_path,
        stops_path,
        subsectors_path,
        swiss_boundaries_path,
    )


@app.cell
def _(french_path, gpd, pl, postcodes_path, stops_path, subsectors_path):
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
    geneva_gdf = (
        gpd.read_file(swiss_boundaries_path, layer="swissBOUNDARIES3D_1_5_TLM_KANTONSGEBIET")
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
    from activitygraphs.network import build_locations

    locations = build_locations(stops, subsectors_gdf, postcodes_gdf, localities_gdf, french_gdf)
    locations_df = pl.DataFrame(locations.drop(columns=["geometry"]))

    locations
    return (locations_df,)


@app.cell
def _(stops):
    from activitygraphs.network import build_stop_names_to_loc_id_mapping

    stop_names_to_id_df = build_stop_names_to_loc_id_mapping(stops)
    stop_names_to_id_df
    return (stop_names_to_id_df,)


@app.cell
def _(df, locations_df, stops):
    from activitygraphs.network import match_loc_ids

    matched_df = match_loc_ids(df, locations_df, stops)
    matched_df
    return (matched_df,)


@app.cell
def _(mo):
    mo.md(r"""
    Types of unmatched location names:
    - [Subsector (Geneva)] - sous_secteur
    - [Municipality (CH)] - [Postcode (CH)]
    - [MUNICIPALITY (FR)] - [Postcode (FR)]
    - NA
    - Typos
    """)
    return


@app.cell
def _(matched_df, pl, stop_names_to_id_df):
    from activitygraphs.network import match_loc_id_on_fuzzy_stop_names

    test = match_loc_id_on_fuzzy_stop_names(
        matched_df, stop_names_to_id_df, "dep_loc_id", "lieu_depart_trajet", "dep_match_type"
    )
    match_loc_id_on_fuzzy_stop_names(
        test, stop_names_to_id_df, "arr_loc_id", "lieu_arrivee_trajet", "arr_match_type"
    ).filter(pl.col("dep_loc_id").is_null())
    return


@app.cell
def _(matched_df, pl, stop_names_to_id_df):
    from rapidfuzz import process, fuzz

    _loc_id_col = "dep_loc_id"
    _match_type_col = "dep_match_type"
    _stop_name_col = "lieu_depart_trajet"

    _unmatched_stop_names = matched_df.filter(pl.col("dep_loc_id").is_null(), pl.col("arr_loc_id").is_null())
    _unmatched_stop_names = pl.concat([
        _unmatched_stop_names["lieu_depart_trajet"],
        _unmatched_stop_names["lieu_arrivee_trajet"],
    ]).unique()
    _unmatched_stop_names = pl.DataFrame(_unmatched_stop_names.alias("stop_name"))

    _stop_names_list = stop_names_to_id_df["loc_name"].str.to_lowercase().to_list()
    _stop_ids_list = stop_names_to_id_df["loc_id"].to_list()


    def fuzzy_match(stop_name: str):
        best_name, best_score, best_idx = process.extractOne(stop_name, _stop_names_list, scorer=fuzz.ratio)

        return {
            "closest_stop_name": best_name,
            "closest_stop_id": _stop_ids_list[best_idx],
            "score": best_score,
        }


    FUZZY_MATCH_THRESHOLD = 65

    stripped_stop_names = pl.col("stop_name").str.to_lowercase().str.strip_chars("0123456789- ")
    matched_stop_names = (
        _unmatched_stop_names.with_columns(stripped_stop_names.map_elements(fuzzy_match).alias("result"))
        .unnest("result")
        .filter(pl.col("score") > FUZZY_MATCH_THRESHOLD)
        .select("stop_name", pl.col("closest_stop_id").alias("loc_id"))
    )

    is_fuzzy_match_success = pl.col(_loc_id_col).is_null() & pl.col("loc_id").is_not_null()

    matched_df.join(matched_stop_names, left_on=_stop_name_col, right_on="stop_name", how="inner").with_columns(
        pl.when(is_fuzzy_match_success).then("loc_id").otherwise(_loc_id_col).alias(_loc_id_col),
        pl.when(is_fuzzy_match_success).then(pl.lit("fuzzy_stop_name")).otherwise(_match_type_col).alias(_match_type_col),
    ).drop("loc_id")
    return (matched_stop_names,)


@app.cell
def _(matched_stop_names, pl, stop_names_df, trips_df):
    stop_name_col = "lieu_depart_trajet"
    stop_id_col = "dep_stop_id"
    closest_stop_col_prefix = "closest_"

    _matches = matched_stop_names.select("stop_name", "closest_stop_id")

    trips_df.join(_matches, left_on=stop_name_col, right_on="stop_name", how="inner").with_columns(
        pl.when(pl.col(stop_id_col).is_null()).then("closest_stop_id").otherwise(stop_id_col).alias(stop_id_col)
    ).join(
        stop_names_df.select(pl.all().name.prefix("dep_")),
        on="dep_stop_id",
        how="left",
    ).join(
        stop_names_df.select(pl.all().name.prefix("arr_")),
        on="arr_stop_id",
        how="left",
    )
    return


@app.cell
def _(locations_df):
    locations_df
    return


@app.cell
def _(location_regexes, pl, trips_df):
    _stop_name_col = "lieu_depart_trajet"
    _regex_name = "subsector"


    trips_df.filter(pl.col(_stop_name_col).str.contains(location_regexes[_regex_name]))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
