import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo

    import torch
    import torch_geometric as pyg

    import polars as pl

    from activitygraphs.config import load_config
    from pathlib import Path

    project_root = Path(mo.notebook_dir().parent)
    cfg = load_config(project_root)


@app.cell
def _():
    from activitygraphs.data.geneva import GenevaData
    from activitygraphs.network import Network

    network_name = "routes"

    gva_data = GenevaData.load(cfg.data, project_root)
    gva_network = Network.load(cfg.data, project_root, network_name)
    return (gva_data,)


@app.cell
def _(gva_data):
    _group_keys = ["user_id", "journey_id"]

    _locations = gva_data.locations_df.select("loc_id", "type")
    _modes = gva_data.user_journeys_df.group_by(_group_keys, maintain_order=True).agg(modes="leg_mode")
    _trips = (
        gva_data.user_journeys_df
        .group_by(_group_keys, maintain_order=True)
        .agg(pl.all().gather([0, -1]))
        .select(
            "user_id",
            "journey_id",
            (pl.col("leg_id").list.last() + 1).alias("num_legs"),
            pl.col("dep_day").list.first(),
            pl.col("dep_time").list.first(),
            pl.col("dep_purpose").list.first(),
            pl.col("dep_loc_id").list.first(),
            pl.col("arr_loc_id").list.last(),
            pl.col("arr_purpose").list.last(),
        )
    )


    trips = (
        _trips
        .join(_modes, on=_group_keys)
        .join(_locations.select(dep_loc_id="loc_id", dep_loc_type="type"), on="dep_loc_id")
        .join(_locations.select(arr_loc_id="loc_id", arr_loc_type="type"), on="arr_loc_id")
        .filter(dep_loc_type="subsector", arr_loc_type="subsector")
    )

    trips
    return (trips,)


@app.cell
def _():
    return


@app.cell
def _(trips):
    visits = (
        pl
        .concat([
            trips.select("user_id", purpose="dep_purpose", loc_id="dep_loc_id"),
            trips.select("user_id", purpose="arr_purpose", loc_id="arr_loc_id"),
        ])
        .unique()
        .sort("user_id")
    )

    visits_by_purpose = visits.group_by("user_id", "purpose").agg(pl.col("loc_id").unique()).sort("user_id")

    visits
    return visits, visits_by_purpose


@app.cell
def _(visits):
    num_visits_per_person = visits.group_by("user_id").agg(num_locs=pl.len())
    num_visits_per_person["num_locs"].value_counts().sort("num_locs")
    return (num_visits_per_person,)


@app.cell
def _(num_visits_per_person, visits):
    avg_visits_per_person = num_visits_per_person.select("num_locs").mean().item()
    avg_non_hwe_visits_per_person = (
        visits
        .filter(~pl.col("purpose").is_in(["od_lieu_domicile", "od_lieu_travail", "od_lieu_etude"]))
        .group_by("user_id")
        .agg(num_locs=pl.len())
        .select("num_locs")
        .mean()
        .item()
    )

    f"Average: {avg_visits_per_person:.3f}, average without home/work/edu: {avg_non_hwe_visits_per_person}"
    return


@app.cell
def _(gpd, visits, visits_by_purpose):
    home_locations = visits_by_purpose.filter(purpose="od_lieu_domicile").with_columns(pl.col("loc_id").list.first())
    work_locations = visits_by_purpose.filter(purpose="od_lieu_travail").with_columns(pl.col("loc_id").list.len())
    edu_locations = visits_by_purpose.filter(purpose="od_lieu_etude").with_columns(pl.col("loc_id").list.len())


    def add_user_cols(locations_gdf: gpd.GeoDataFrame, user_id: str) -> pl.DataFrame:
        locations_gdf = locations_gdf.copy()

        home_location = home_locations.filter(user_id=user_id)["loc_id"].to_list()
        work_location = work_locations.filter(user_id=user_id)["loc_id"].to_list()
        edu_location = edu_locations.filter(user_id=user_id)["loc_id"].to_list()
        user_visits = visits.filter(user_id=user_id)["loc_id"].to_list()

        locations_gdf["is_home"] = locations_gdf["loc_id"].isin(home_location)
        locations_gdf["is_work"] = locations_gdf["loc_id"].isin(work_location)
        locations_gdf["is_edu"] = locations_gdf["loc_id"].isin(edu_location)
        locations_gdf["is_visited"] = locations_gdf["loc_id"].isin(user_visits)

        return locations_gdf

    return (add_user_cols,)


@app.cell
def _(add_user_cols, locations):
    user_id = "20703"

    add_user_cols(locations, user_id)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Adding POI and land use information
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Graph creation
    """)
    return


@app.cell
def _():
    import geopandas as gpd
    import pandas as pd

    CRS = "EPSG:4326"
    return CRS, gpd, pd


@app.cell
def _(gva_data):
    locations = gva_data.locations_gdf.query("type == 'subsector'")
    return (locations,)


@app.cell
def _(locations):
    utm_crs = locations.estimate_utm_crs()
    return (utm_crs,)


@app.cell
def _():
    import city2graph as c2g

    return (c2g,)


@app.cell
def _(visits):
    _user_ids = visits["user_id"].unique().sort()
    select_user_id = mo.ui.dropdown(_user_ids, value=_user_ids.first(), label="User ID:", searchable=True)
    return (select_user_id,)


@app.cell
def _(CRS, add_user_cols, c2g, locations, pd, select_user_id):
    _locations = add_user_cols(locations, select_user_id.value)

    _locations = _locations.to_crs(locations.estimate_utm_crs()).set_index("loc_id")
    nodes, edges = c2g.contiguity_graph(_locations, set_point_nodes=True)

    _island_loc_ids = ["subsector-174", "subsector-141", "subsector-10", "subsector-325"]
    _cross_lake_loc_ids = ["subsector-243", "subsector-261", "subsector-54"]
    _island_nodes = nodes[nodes.index.isin(_island_loc_ids)]
    _mainland_nodes = nodes[~nodes.index.isin(_island_loc_ids) & ~nodes.index.isin(_cross_lake_loc_ids)]

    _, _island_edges = c2g.knn_graph(_island_nodes, k=3, target_gdf=_mainland_nodes)
    _island_edges = _island_edges.reset_index()
    _island_edges["source"] = _island_edges["source"].str[1]
    _island_edges["target"] = _island_edges["target"].str[1]
    _island_edges = _island_edges.set_index(["source", "target"])

    nodes = nodes.to_crs(CRS)
    edges = pd.concat([edges, _island_edges]).to_crs(CRS)
    return edges, nodes


@app.cell
def _(nodes):
    nodes
    return


@app.cell
def _(edges):
    edges.reset_index()
    return


@app.cell
def _(edges, nodes, select_user_id):
    _m = edges.explore(color="gray", tiles="Cartodb Positron")
    _m = nodes.explore(m=_m, column="visit", marker_kwds={"radius": 4})

    mo.vstack([select_user_id, _m])
    return


@app.cell
def _():
    output_dir = project_root / cfg.data.paths.external / "ouverture"
    return (output_dir,)


@app.cell
def _(CRS, c2g, nodes, output_dir):
    mo.stop(True)
    c2g.load_overture_data(area=nodes["original_geometry"].to_crs(CRS).union_all(), types=["land_use", "place"], output_dir=output_dir, save_to_file=True)
    return


@app.cell
def _(gpd, output_dir):
    land_use = gpd.read_file(output_dir / "land_use.geojson")[["id", "subtype", "class", "geometry"]].set_index("id")
    land_use
    return (land_use,)


@app.cell
def _(gpd, output_dir):
    places = gpd.read_file(output_dir / "place.geojson")[["id", "names", "basic_category", "taxonomy", "geometry"]].set_index("id")
    places
    return


@app.cell
def _(land_use, nodes, utm_crs):
    nodes.to_crs(utm_crs).set_geometry("original_geometry").sjoin(land_use.to_crs(utm_crs), how="left")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
