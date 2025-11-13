import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import polars.selectors as cs
    return cs, mo, pl


@app.cell
def _(mo):
    from activitygraphs.config import load_config

    cfg = load_config(mo.notebook_dir().parent)
    return (cfg,)


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
def _(cfg, pl):
    raw_path = cfg.data.paths.raw + "/" + cfg.data.files.raw_trips

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
def _(cfg):
    from activitygraphs.data.geneva import _read_raw_data, _handle_null_values

    r = _read_raw_data(cfg.data)
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


@app.cell
def _(mo):
    mo.md(r"""
    # GTFS to NetworkX construction
    """)
    return


@app.cell
def _():
    import partridge as ptg
    from pathlib import Path

    # Pick a random date in 2022

    path = Path("data/raw/gtfs/gtfs_2022_switzerland")
    boundaries = Path("data/raw/boundaries/swissboundaries3d_2025-04_2056_5728.shp/swissBOUNDARIES3D_1_5_TLM_KANTONSGEBIET.shp")
    return boundaries, path


@app.cell
def _(path, pl):
    stops = pl.read_csv(path / "stops.txt", schema={
        "stop_id": pl.String,
        "stop_name": pl.String,
        "stop_lat": pl.Float32,
        "stop_lon": pl.Float32,
        "location_type": pl.Categorical,
        "parent_station": pl.String,
    })
    return (stops,)


@app.cell
def _(boundaries):
    import geopandas as gpd

    geneva = gpd.read_file(boundaries).to_crs("EPSG:4326").query("NAME == 'Genève'").iloc[0]["geometry"]
    geneva
    return geneva, gpd


@app.cell
def _(geneva, gpd, stops):
    stops_gdf = gpd.GeoDataFrame(stops.to_pandas(), geometry=gpd.points_from_xy(stops["stop_lon"], stops["stop_lat"], crs="EPSG:4326"))

    stops_gdf[stops_gdf.intersects(geneva)].explore()
    return


@app.cell
def _(stops):
    stops
    return


if __name__ == "__main__":
    app.run()
