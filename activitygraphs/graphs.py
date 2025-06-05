import dataprocessing as dp
import networkx as nx
import polars as pl

NODE_ATTRIBUTE_SCHEMA = pl.Schema({
    "loc_id": pl.String,
    "is_home": pl.Boolean,
    "is_work": pl.Boolean,
    "land_uses": dp.LandUse.polars_enum(),
    "purposes": dp.Purpose.polars_enum(),
    "lat": pl.Float64(),
    "lon": pl.Float64(),
})

EDGELIST_SCHEMA = dp.TRIP_SCHEMA


def generate_node_attribute_df(
    hh_person_df: pl.DataFrame, trip_df: pl.DataFrame
) -> pl.DataFrame:
    origin_trip_nodes = trip_df.select(
        loc_id="loc_origin_loc_id",
        is_home=False,
        is_work=False,
        land_use="land_use",
        purpose="purpose",
        lat="loc_origin_lat",
        lon="loc_origin_lon",
    ).unique()

    dest_trip_nodes = trip_df.select(
        loc_id="loc_dest_loc_id",
        is_home=False,
        is_work=False,
        land_use=None,
        purpose="purpose_dest",
        lat="loc_destination_lat",
        lon="loc_destination_lon",
    ).unique()

    home_nodes = hh_person_df.select(
        loc_id="loc_home_loc_id",
        is_home=True,
        is_work=False,
        land_use=None,
        purpose=pl.lit(dp.Purpose.HOME, dtype=dp.Purpose.polars_enum()),
        lat="loc_home_lat",
        lon="loc_home_lon",
    ).unique()

    work_nodes = (
        hh_person_df.filter(pl.col("loc_work_loc_id") != "-1")
        .select(
            loc_id="loc_work_loc_id",
            is_home=False,
            is_work=True,
            land_use=None,
            purpose=pl.lit(dp.Purpose.WORK, dtype=dp.Purpose.polars_enum()),
            lat="loc_work_lat",
            lon="loc_work_lon",
        )
        .unique()
    )

    nodes = pl.concat([
        origin_trip_nodes,
        dest_trip_nodes,
        home_nodes,
        work_nodes,
    ])

    node_attributes_df = nodes.group_by("loc_id").agg(
        pl.col("is_home").any(),
        pl.col("is_work").any(),
        pl.col("land_use").drop_nulls().bitwise_or().alias("land_uses"),
        pl.col("purpose").drop_nulls().bitwise_or().alias("purposes"),
        pl.col("lat").drop_nulls().first().alias("lat"),
        pl.col("lon").drop_nulls().first().alias("lon"),
    )

    return dp.check_schema(node_attributes_df, NODE_ATTRIBUTE_SCHEMA)


def generate_hh_node_and_edgelist(
    hh_id: str, hh_person_df: pl.DataFrame, trip_df: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    single_hh_person_df = hh_person_df.filter(pl.col("hh_id") == hh_id)
    single_trip_df = trip_df.filter(pl.col("hh_id") == hh_id).with_columns(
        pl.col("mode").map_elements(dp.Mode, return_dtype=pl.Object),
    )

    node_attribute_df = generate_node_attribute_df(single_hh_person_df, single_trip_df)

    return node_attribute_df, single_trip_df


def generate_hh_graph(
    nodelist_df: pl.DataFrame, edgelist_df: pl.DataFrame
) -> nx.MultiDiGraph:
    G = nx.from_pandas_edgelist(
        edgelist_df,
        source="loc_origin_loc_id",
        target="loc_dest_loc_id",
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

    node_attribute_dict = nodelist_df.with_columns(
        pl.col("purposes").map_elements(dp.Purpose, return_dtype=pl.Object),
        pl.col("land_uses").map_elements(dp.LandUse, return_dtype=pl.Object),
    ).rows_by_key(key="loc_id", named=True, unique=True)
    nx.set_node_attributes(G, node_attribute_dict)
    return G
