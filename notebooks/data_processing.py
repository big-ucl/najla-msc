import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import pprint

    import marimo as mo
    import polars as pl

    from pathlib import Path
    return Path, mo, pl


@app.cell
def _(mo):
    from config import LTDSConfig, load_config

    project_root = mo.notebook_dir().parent

    cfg = load_config(project_root)
    data_path = project_root / cfg.paths.data_raw_ltds

    print(f"Configuration loaded: {cfg}")
    return cfg, data_path


@app.cell
def _(mo):
    run_button = mo.ui.run_button(
        kind="warn", label="Run excel to parquet conversion"
    )
    run_button
    return (run_button,)


@app.cell
def _(Path, cfg, data_path, mo, run_button):
    from dataprocessing import convert_excel_to_parquet

    mo.stop(not run_button.value, mo.md("Click button above to run conversion"))

    _files = (Path(s) for s in cfg.files.values())
    convert_excel_to_parquet(data_path, *_files)
    return


@app.cell
def _(Path, cfg, data_path, pl):
    def read_from_parquet(path: Path, schema: dict = None) -> pl.DataFrame:
        schema = {} if schema is None else schema

        df = pl.read_parquet(path)
        return pl.DataFrame(df, schema_overrides=schema)


    raw_household_df = read_from_parquet(
        data_path / cfg.files.raw_household,
        schema={
            "hhid": pl.String,
        },
    )

    raw_person_df = read_from_parquet(
        data_path / cfg.files.raw_person,
        schema={
            "phid": pl.String,
            "ppid": pl.String,
        },
    )

    raw_trip_df = read_from_parquet(
        data_path / cfg.files.raw_trip,
        schema={
            "thid": pl.String,
            "tpid": pl.String,
            "ttid": pl.String,
        },
    )
    return raw_household_df, raw_person_df, raw_trip_df


@app.cell
def _(pl, raw_household_df, raw_person_df):
    SURVEY_START_YEAR = 2000


    def combine_postcode_col(pc_out: str, pc_in: str) -> pl.Expr:
        postcode_non_null = (pl.col(pc_out) != "-1") & (pl.col(pc_in) != "-1")
        return (
            pl.when(postcode_non_null)
            .then(pl.col(pc_out) + " " + pl.col(pc_in))
            .otherwise(-1)
        )


    def create_hh_person_df(
        raw_person_df: pl.DataFrame, raw_household_df: pl.DataFrame
    ) -> pl.DataFrame:
        person_df = raw_person_df.select(
            hh_id="phid",
            person_id="ppid",
            year=pl.col("pyearid") + SURVEY_START_YEAR,
            loc_work_postcode=combine_postcode_col("pwspcout", "pwspcin"),
        )

        household_df = raw_household_df.select(
            hh_id="hhid",
            loc_home_postcode=combine_postcode_col("hhpcout", "hhpcin"),
            year=pl.col("hyearid") + SURVEY_START_YEAR,
        )

        hh_person_df = person_df.join(household_df, on=["hh_id", "year"])
        return hh_person_df


    hh_person_df = create_hh_person_df(raw_person_df, raw_household_df)
    hh_person_df.head()
    return SURVEY_START_YEAR, combine_postcode_col, hh_person_df


@app.cell
def _(SURVEY_START_YEAR, combine_postcode_col, pl, raw_trip_df):
    trip_df = raw_trip_df.select(
        hh_id="thid",
        person_id="tpid",
        trip_id="ttid",
        trip_number="tseqno",
        year=pl.col("tyearid") + SURVEY_START_YEAR,
        loc_origin_postcode=combine_postcode_col("topcout", "topcin"),
        loc_dest_postcode=combine_postcode_col("tdpcout", "tdpcin"),
        mode="tdbmmode",
        duration="tdurn",
        distance="tlenn",
        purpose="topurpi",
        land_use="toland",
        start_time="tstime",
        end_time="tetime",
    )

    trip_df.head()
    return (trip_df,)


@app.cell
def _(pl, trip_df):
    single_hh_id = "10227111"

    single_trips = trip_df.filter(pl.col("hh_id") == single_hh_id)
    single_trips
    return (single_hh_id,)


@app.cell
def _(hh_person_df, pl, single_hh_id, trip_df):
    import networkx as nx

    def generate_hh_graph(
        hh_id: str, hh_person_df: pl.DataFrame, trip_df: pl.DataFrame
    ):
        single_hh_person_df = hh_person_df.filter(pl.col("hh_id") == hh_id)
        single_trip_df = trip_df.filter(pl.col("hh_id") == hh_id)

        G = nx.from_pandas_edgelist(
            single_trip_df,
            source="loc_origin_postcode",
            target="loc_dest_postcode",
            edge_key="trip_id",
            create_using=nx.MultiDiGraph,
            edge_attr=["mode", "duration", "distance", "start_time", "end_time"],
        )

        return G


    G = generate_hh_graph(single_hh_id, hh_person_df, trip_df)
    return G, nx


@app.cell
def _(G, nx):
    import matplotlib.pyplot as plt

    def draw_hh_graph(G: nx.MultiDiGraph):
        layout = nx.layout.kamada_kawai_layout(G, weight="distance")
    
        fig, ax = plt.subplots()
        nx.draw(G, ax=ax, with_labels=True, connectionstyle="arc3,rad=0.1")
        return fig

    draw_hh_graph(G)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
