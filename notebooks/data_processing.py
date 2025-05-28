import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells

    # Import modules
    import networkx as nx
    import marimo as mo
    import polars as pl
    from pathlib import Path
    from config import load_config
    import numpy as np
    import random

    # Load configuration from files
    project_root = mo.notebook_dir().parent

    cfg = load_config(project_root)
    data_path = project_root / cfg.paths.data_raw_ltds

    print(f"Configuration loaded: {cfg}")

    # Set random seeds
    np.random.seed(42)
    random.seed(42)


@app.cell
def _():
    run_button = mo.ui.run_button(
        kind="warn", label="Run excel to parquet conversion"
    )
    run_button
    return (run_button,)


@app.cell
def _(run_button):
    from dataprocessing import convert_excel_to_parquet

    mo.stop(not run_button.value, mo.md("Click button above to run conversion"))

    _files = (Path(s) for s in cfg.files.values())
    convert_excel_to_parquet(data_path, *_files)
    return


@app.cell
def _():
    from dataprocessing import Purposes, LandUses

    purposes_keys = [
        "-2",
        "-1",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        "18",
        "19",
        "20",
        "21",
    ]


    land_uses_keys = [
        "-2",
        "-1",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
    ]

    purposes_mapping = {k: v for k, v in zip(purposes_keys, Purposes.names())}
    land_uses_mapping = {k: v for k, v in zip(land_uses_keys, LandUses.names())}
    return LandUses, Purposes, land_uses_mapping, purposes_mapping


@app.cell
def _():
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
            "topurpi": pl.String,
            "tdpurp": pl.String,
            "toland": pl.String,
            "tdland": pl.String,
        },
    )
    return raw_household_df, raw_person_df, raw_trip_df


@app.cell
def _(raw_household_df, raw_person_df):
    SURVEY_START_YEAR = 2000


    def is_ltds_entry_valid(colname: str) -> bool:
        return (pl.col(colname) != "-1") & (pl.col(colname) != "-2")


    def combine_postcode_col(pc_out: str, pc_in: str) -> pl.Expr:
        postcode_non_null = is_ltds_entry_valid(pc_out) & is_ltds_entry_valid(
            pc_in
        )
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
def _(
    LandUses,
    Purposes,
    SURVEY_START_YEAR,
    combine_postcode_col,
    land_uses_mapping,
    purposes_mapping,
    raw_trip_df,
):
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
        purpose=pl.col("topurpi")
        .replace(purposes_mapping)
        .cast(Purposes.polars_enum()),
        purpose_dest=pl.col("tdpurp")
        .replace(purposes_mapping)
        .cast(Purposes.polars_enum()),
        land_use=pl.col("toland")
        .replace(land_uses_mapping)
        .cast(LandUses.polars_enum()),
        start_time="tstime",
        end_time="tetime",
    )

    trip_df.head()
    return (trip_df,)


@app.cell
def _(LandUses, Purposes):
    def generate_node_attribute_df(
        single_hh_person_df: pl.DataFrame, single_trip_df: pl.DataFrame
    ):
        schema_df = pl.DataFrame(
            schema={
                "postcode": pl.String,
                "is_home": pl.Boolean,
                "is_work": pl.Boolean,
                "land_use": LandUses.polars_enum(),
                "purpose": Purposes.polars_enum(),
            }
        )

        origin_trip_nodes = single_trip_df.select(
            postcode="loc_origin_postcode",
            is_home=False,
            is_work=False,
            land_use="land_use",
            purpose="purpose",
        ).unique()

        dest_trip_nodes = single_trip_df.select(
            postcode="loc_dest_postcode",
            is_home=False,
            is_work=False,
            land_use=None,
            purpose="purpose_dest",
        ).unique()

        home_nodes = single_hh_person_df.select(
            postcode="loc_home_postcode",
            is_home=True,
            is_work=False,
            land_use=None,
            purpose=pl.lit(Purposes.HOME).cast(Purposes.polars_enum()),
        ).unique()

        work_nodes = (
            single_hh_person_df.filter(pl.col("loc_work_postcode") != "-1")
            .select(
                postcode="loc_work_postcode",
                is_home=False,
                is_work=True,
                land_use=None,
                purpose=pl.lit(Purposes.WORK).cast(Purposes.polars_enum()),
            )
            .unique()
        )

        nodes = pl.concat(
            [schema_df, origin_trip_nodes, dest_trip_nodes, home_nodes, work_nodes]
        )

        return nodes.group_by("postcode").agg(
            pl.col("is_home").any(),
            pl.col("is_work").any(),
            pl.col("land_use").unique().drop_nulls().alias("land_uses"),
            pl.col("purpose").unique().drop_nulls().alias("purposes"),
        )
    return (generate_node_attribute_df,)


@app.cell
def _(generate_node_attribute_df):
    def generate_hh_node_and_edgelist(
        hh_id: str, hh_person_df: pl.DataFrame, trip_df: pl.DataFrame
    ):
        single_hh_person_df = hh_person_df.filter(pl.col("hh_id") == hh_id)
        single_trip_df = trip_df.filter(pl.col("hh_id") == hh_id)

        node_attribute_df = generate_node_attribute_df(
            single_hh_person_df, single_trip_df
        )

        return node_attribute_df, single_trip_df


    def generate_hh_graph(nodelist_df: pl.DataFrame, edgelist_df: pl.DataFrame):
        G = nx.from_pandas_edgelist(
            edgelist_df,
            source="loc_origin_postcode",
            target="loc_dest_postcode",
            edge_key="trip_id",
            create_using=nx.MultiDiGraph,
            edge_attr=[
                "person_id",
                "mode",
                "duration",
                "distance",
                "start_time",
                "end_time",
            ],
        )

        node_attribute_dict = nodelist_df.rows_by_key(
            key="postcode", named=True, unique=True
        )
        nx.set_node_attributes(G, node_attribute_dict)

        return G
    return generate_hh_graph, generate_hh_node_and_edgelist


@app.cell
def _():
    interesting_hh_id = "12109151"
    new_graph_switch = mo.ui.switch(label="Use random graph")
    return interesting_hh_id, new_graph_switch


@app.cell
def _(hh_person_df, interesting_hh_id, new_graph_switch):
    def _draw_new_random_hh_id(value: str) -> str:
        return hh_person_df.select("hh_id").unique().sample(1)[0, "hh_id"]


    new_graph_button = mo.ui.button(
        label="Click to draw new random graph",
        disabled=not new_graph_switch.value,
        value=interesting_hh_id,
        on_click=_draw_new_random_hh_id,
    )

    mo.hstack([new_graph_switch, new_graph_button])
    return (new_graph_button,)


@app.cell
def _(
    generate_hh_graph,
    generate_hh_node_and_edgelist,
    hh_person_df,
    new_graph_button,
    trip_df,
):
    from plotting import (
        draw_hh_graph,
        line_styles_by_key,
        node_colours_by_purpose,
        node_short_labels_by_purpose,
    )

    _hh_id = new_graph_button.value

    nodelist_df, edgelist_df = generate_hh_node_and_edgelist(
        _hh_id, hh_person_df, trip_df
    )

    G = generate_hh_graph(nodelist_df, edgelist_df)
    line_styles = line_styles_by_key(G, key="person_id")
    node_colours = node_colours_by_purpose(G)
    node_labels = node_short_labels_by_purpose(G)

    draw_hh_graph(G, _hh_id, line_styles=line_styles, node_colours=node_colours, node_labels=node_labels)
    return G, edgelist_df


@app.cell
def _(edgelist_df):
    edgelist_df
    return


@app.cell
def _(G):
    for _n in G.nodes(data=True):
        print(_n)
    return


@app.cell
def _(G):
    for _e in G.edges(data=True):
        print(_e)
    return


@app.cell
def _():
    mo.md(
        r"""
    TODO Notes:

     - Draw graph in a more interesting manner
     - Handle case with unknown location (negative weights)
     - Add physical location information to graph
    """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
