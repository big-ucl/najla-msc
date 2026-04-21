import marimo

__generated_with = "0.21.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo

    import torch
    import torch_geometric as pyg

    import polars as pl

    from activitygraphs.config import load_config
    from pathlib import Path

    project_root = Path(mo.notebook_dir().parent.parent)
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
    _modes = gva_data.user_journeys_df.group_by(
        _group_keys, maintain_order=True
    ).agg(modes="leg_mode")
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
        .join(
            _locations.select(dep_loc_id="loc_id", dep_loc_type="type"),
            on="dep_loc_id",
        )
        .join(
            _locations.select(arr_loc_id="loc_id", arr_loc_type="type"),
            on="arr_loc_id",
        )
        .filter(dep_loc_type="subsector", arr_loc_type="subsector")
    )

    trips
    return (trips,)


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

    visits_by_purpose = (
        visits
        .group_by("user_id", "purpose")
        .agg(pl.col("loc_id").unique())
        .sort("user_id")
    )
    visits_by_purpose
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
        .filter(
            ~pl.col("purpose").is_in([
                "od_lieu_domicile",
                "od_lieu_travail",
                "od_lieu_etude",
            ])
        )
        .group_by("user_id")
        .agg(num_locs=pl.len())
        .select("num_locs")
        .mean()
        .item()
    )

    f"Average: {avg_visits_per_person:.3f}, average without home/work/edu: {avg_non_hwe_visits_per_person}"
    return


@app.cell
def _(gva_data):
    locations = gva_data.locations_gdf.query("type == 'subsector'")
    return (locations,)


@app.cell
def _(gpd, locations, visits, visits_by_purpose):
    home_locations = visits_by_purpose.filter(purpose="od_lieu_domicile").with_columns(pl.col("loc_id").list.first())
    work_locations = visits_by_purpose.filter(purpose="od_lieu_travail").with_columns(pl.col("loc_id").list.len())
    edu_locations = visits_by_purpose.filter(purpose="od_lieu_etude").with_columns(pl.col("loc_id").list.len())


    def add_user_cols(nodes: gpd.GeoDataFrame, user_id: str) -> pl.DataFrame:
        nodes = nodes.copy().reset_index()

        home_location = home_locations.filter(user_id=user_id)["loc_id"].to_list()
        work_location = work_locations.filter(user_id=user_id)["loc_id"].to_list()
        edu_location = edu_locations.filter(user_id=user_id)["loc_id"].to_list()
        user_visits = visits.filter(user_id=user_id)["loc_id"].to_list()

        nodes["is_home"] = nodes["loc_id"].isin(home_location).astype(int)
        nodes["is_work"] = nodes["loc_id"].isin(work_location).astype(int)
        nodes["is_edu"] = nodes["loc_id"].isin(edu_location).astype(int)
        nodes["is_visited"] = nodes["loc_id"].isin(user_visits).astype(int)

        visit_purposes = visits.filter(user_id=user_id).select("loc_id", pl.col("purpose")).group_by("loc_id").agg(pl.col("purpose").str.join(", ")).to_pandas()
        nodes = nodes.merge(visit_purposes, on="loc_id", how="left")
        nodes = nodes.set_index("loc_id").sort_index()

        return nodes

    _user_id = "20704"
    add_user_cols(locations, _user_id)
    return (add_user_cols,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Adding POI and land use information
    """)
    return


@app.cell
def _(locations):
    import geopandas as gpd
    import pandas as pd

    CRS = "EPSG:4326"
    utm_crs = locations.estimate_utm_crs()
    return CRS, gpd, pd, utm_crs


@app.cell
def _():
    output_dir = project_root / cfg.data.paths.external / "ouverture"
    return (output_dir,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Load external data
    """)
    return


@app.cell
def _(CRS, c2g, locations, output_dir):
    mo.stop(True)
    c2g.load_overture_data(
        area=locations["geometry"].to_crs(CRS).union_all(),
        types=["land_use", "place"],
        output_dir=output_dir,
        save_to_file=True,
    )
    return


@app.cell
def _(gpd, output_dir):
    land_use = gpd.read_file(output_dir / "land_use.geojson")[
        ["id", "subtype", "class", "geometry"]
    ].set_index("id")
    land_use = land_use[
        land_use.geom_type.isin(["Polygon", "MultiPolygon"])
    ].copy()
    land_use = land_use.explode(index_parts=False)
    land_use
    return (land_use,)


@app.cell
def _(gpd, output_dir):
    places = gpd.read_file(output_dir / "place.geojson")[
        ["id", "names", "basic_category", "taxonomy", "geometry"]
    ].set_index("id")
    places
    return (places,)


@app.cell
def _(gpd, utm_crs):
    statistics = gpd.read_file(
        project_root
        / cfg.data.paths.external
        / "statistics/geneva"
        / "AGGLO_CARREAU_200-SHP.zip"
    )

    statistics["population"] = (
        statistics["D_POP_HA"] * statistics["GEOM_AREA"] / 10000
    )
    statistics["jobs"] = statistics["D_EMP_HA"] * statistics["GEOM_AREA"] / 10000

    statistics = statistics.rename(columns={"GRID_ID": "id"}).to_crs(utm_crs)
    statistics = statistics[["id", "population", "jobs", "geometry"]]

    statistics
    return (statistics,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Enhance node features
    """)
    return


@app.cell
def _(gpd, locations, pd, places, utm_crs):
    def add_poi_counts(
        locations: gpd.GeoDataFrame,
        places: gpd.GeoDataFrame,
        utm_crs: str,
        category="top_category",
        normalise=True,
    ) -> pd.DataFrame:
        original_locations = locations
        locations = (
            locations.reset_index()[["loc_id", "geometry"]].to_crs(utm_crs).copy()
        )
        locations["area"] = locations.geometry.area

        places = places.copy()
        places["top_category"] = places["taxonomy"].str.extract(
            r'hierarchy[\'"]:\s?\[[\'"]([^\'"]+)[\'"]'
        )  # Extract the first element in the "hiearchy"

        intersection = locations.sjoin(
            places.to_crs(utm_crs), predicate="intersects", how="left"
        )
        poi_by_sector = pd.DataFrame(
            intersection.groupby(["loc_id", category]).size(), columns=["count"]
        ).reset_index()

        if normalise:
            poi_by_sector = locations[["loc_id", "area"]].merge(
                poi_by_sector, on="loc_id", how="left"
            )
            poi_by_sector["count"] = poi_by_sector["count"] / poi_by_sector["area"]

        noi_poi_counts = pd.pivot_table(
            poi_by_sector,
            columns=category,
            index="loc_id",
            values="count",
            fill_value=0,
        ).add_prefix("poi_")
        noi_poi_counts = original_locations.merge(
            noi_poi_counts, on="loc_id", how="left"
        ).fillna(0)

        return noi_poi_counts


    add_poi_counts(
        locations, places, utm_crs, category="top_category", normalise=True
    )
    return (add_poi_counts,)


@app.cell
def _(gpd, land_use, locations, pd, utm_crs):
    def add_land_uses(
        locations: gpd.GeoDataFrame,
        land_uses: gpd.GeoDataFrame,
        utm_crs,
        normalise=True,
    ) -> pd.DataFrame:
        original_locations = locations

        locations = (
            locations.reset_index()[["loc_id", "geometry"]].to_crs(utm_crs).copy()
        )
        locations["total_area"] = locations.geometry.area

        intersection = locations.overlay(
            land_uses.to_crs(utm_crs), how="intersection"
        )
        intersection["land_use_area"] = intersection.geometry.area
        land_use_by_sector = (
            intersection
            .drop(columns=["geometry"])
            .groupby(["loc_id", "subtype"])
            .sum(numeric_only=True)
            .reset_index()
            .rename(columns={"subtype": "land_use"}, errors="raise")
        )

        if normalise:
            land_use_by_sector["pct_area"] = (
                land_use_by_sector["land_use_area"]
                / land_use_by_sector["total_area"]
            )

        value_col = "pct_area" if normalise else "land_use_area"
        loc_land_uses = pd.pivot_table(
            land_use_by_sector,
            columns="land_use",
            index="loc_id",
            values=value_col,
            fill_value=0,
        ).add_prefix("land_use_")
        loc_land_uses = original_locations.merge(
            loc_land_uses, on="loc_id", how="left"
        ).fillna(0)

        return loc_land_uses


    add_land_uses(locations, land_use, utm_crs)
    return (add_land_uses,)


@app.cell
def _(gpd, locations, statistics, utm_crs):
    def add_pop_empl_stats(
        locations: gpd.GeoDataFrame,
        statistics: gpd.GeoDataFrame,
        utm_crs,
        normalise=True,
    ):
        original_locations = locations
        locations = (
            locations.reset_index()[["loc_id", "geometry"]].to_crs(utm_crs).copy()
        )
        locations["area"] = locations.geometry.area

        intersection = locations.sjoin(
            statistics.to_crs(utm_crs), how="left", predicate="intersects"
        )
        stats_by_sector = intersection.groupby("loc_id")[
            ["population", "jobs"]
        ].sum()

        if normalise:
            stats_by_sector = stats_by_sector.merge(
                locations[["loc_id", "area"]], on="loc_id"
            )
            stats_by_sector["population"] = (
                stats_by_sector["population"] / stats_by_sector["area"]
            )
            stats_by_sector["jobs"] = (
                stats_by_sector["jobs"] / stats_by_sector["area"]
            )
            stats_by_sector = stats_by_sector.drop(columns=["area"])

        stats_by_sector = original_locations.merge(
            stats_by_sector, on="loc_id", how="right"
        )
        stats_by_sector["area"] = stats_by_sector.to_crs(utm_crs).geometry.area

        return stats_by_sector


    add_pop_empl_stats(locations, statistics, utm_crs)
    return (add_pop_empl_stats,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Individual graph creation
    """)
    return


@app.cell
def _():
    import city2graph as c2g

    processed_path = project_root / cfg.data.paths.processed / "GenevaTPG2"
    dataset_path = processed_path / "dataset.pickle"
    network_path = processed_path / "networkgraph"

    processed_path.mkdir(parents=True, exist_ok=True)
    network_path.mkdir(parents=True, exist_ok=True)
    return c2g, dataset_path, network_path


@app.cell
def _(visits):
    user_ids = visits["user_id"].unique().sort()
    return (user_ids,)


@app.cell
def _(
    CRS,
    add_land_uses,
    add_poi_counts,
    add_pop_empl_stats,
    c2g,
    gpd,
    land_use,
    locations,
    network_path,
    pd,
    places,
    statistics,
    utm_crs,
):
    def build_network_graph(
        locations: gpd.GeoDataFrame,
        places: gpd.GeoDataFrame,
        land_use: gpd.GeoDataFrame,
        statistics: gpd.GeoDataFrame,
        utm_crs,
    ):
        locations = locations.to_crs(utm_crs)

        network_locations = add_pop_empl_stats(locations, statistics, utm_crs)
        network_locations = add_poi_counts(network_locations, places, utm_crs)
        network_locations = add_land_uses(network_locations, land_use, utm_crs)

        network_locations = network_locations.set_index("loc_id")

        nodes, edges = c2g.contiguity_graph(
            network_locations, set_point_nodes=True
        )

        # Connect disconnected subsectors to main graph
        island_loc_ids = [
            "subsector-174",
            "subsector-141",
            "subsector-10",
            "subsector-325",
        ]
        cross_lake_loc_ids = ["subsector-243", "subsector-261", "subsector-54"]
        island_nodes = nodes[nodes.index.isin(island_loc_ids)]
        mainland_nodes = nodes[
            ~nodes.index.isin(island_loc_ids)
            & ~nodes.index.isin(cross_lake_loc_ids)
        ]

        _, island_edges = c2g.knn_graph(
            island_nodes, k=3, target_gdf=mainland_nodes
        )
        island_edges = island_edges.reset_index()
        island_edges["source"] = island_edges["source"].str[1]
        island_edges["target"] = island_edges["target"].str[1]
        island_edges = island_edges.set_index(["source", "target"])

        nodes = nodes.to_crs(CRS)
        edges = pd.concat([edges, island_edges]).to_crs(CRS)

        return nodes, edges


    mo.stop(False)

    _network_nodes, _network_edges = build_network_graph(
        locations, places, land_use, statistics, utm_crs
    )

    _network_nodes.to_parquet(network_path / "nodes.parquet")
    _network_edges.to_parquet(network_path / "edges.parquet")
    return


@app.cell
def _(gpd, network_path):
    network_nodes = gpd.read_parquet(network_path / "nodes.parquet")
    network_edges = gpd.read_parquet(network_path / "edges.parquet")
    return network_edges, network_nodes


@app.cell
def _():
    run_gen = mo.ui.run_button()
    run_gen
    return (run_gen,)


@app.cell
def _(
    add_user_cols,
    c2g,
    dataset_path,
    network_edges,
    network_nodes,
    run_gen,
    user_ids,
):
    from tqdm import tqdm
    import pickle

    import torch_geometric.transforms as T

    from joblib import Parallel, delayed

    _excluded_feature_cols = [
        "loc_name",
        "type",
        "lon",
        "lat",
        "is_work",
        "is_edu",
        "is_visited",
        "purpose",
        "geometry",
        "original_geometry",
    ]

    node_counts = []


    def build_graph(user_id):
        indiv_nodes = add_user_cols(network_nodes, user_id)
        node_feature_cols = [
            col for col in indiv_nodes.columns if col not in _excluded_feature_cols
        ]

        indiv_graph = c2g.gdf_to_pyg(
            indiv_nodes,
            network_edges,
            node_feature_cols=node_feature_cols,
            node_label_cols=["is_visited"],
            edge_feature_cols=["weight"],
            keep_geom=False,
            device="cpu",
        )

        indiv_graph.user_id = user_id

        if indiv_graph.num_nodes != 469:
            raise ValueError(user_id, len(indiv_nodes))

        transforms = T.Compose([
            T.AddRandomWalkPE(walk_length=20, attr_name=None),
            T.AddLaplacianEigenvectorPE(k=8, attr_name=None),
        ])

        return transforms(indiv_graph)


    mo.stop(not run_gen.value)

    _graphs = Parallel(n_jobs=-1)(
        delayed(build_graph)(user_id) for user_id in user_ids
    )


    with open(dataset_path, "wb") as _f:
        pickle.dump(_graphs, _f)
    return


@app.cell
def _(dataset_path):
    with open(dataset_path, "rb") as _f:
        pass #graphs = pickle.load(_f)

    #graphs
    return


@app.cell
def _(user_id, user_ids):
    select_user_id = mo.ui.dropdown(
        user_ids, value=user_id, label="User ID:", searchable=True
    )
    return (select_user_id,)


@app.cell
def _(add_user_cols, network_nodes, select_user_id):
    _user_id = select_user_id.value
    user_nodes = add_user_cols(network_nodes, _user_id)
    return (user_nodes,)


@app.cell
def _(user_nodes):
    select_node_col = mo.ui.dropdown(
        list(user_nodes.columns), searchable=True, label="Column:", value="purpose"
    )
    return (select_node_col,)


@app.cell
def _():
    toggle_polygons = mo.ui.switch(value=False, label="Show subsectors")
    return (toggle_polygons,)


@app.cell
def _(
    network_edges,
    select_node_col,
    select_user_id,
    toggle_polygons,
    user_nodes,
):
    _nodes = (
        user_nodes.set_geometry("original_geometry")
        if toggle_polygons.value
        else user_nodes
    )
    _m = network_edges.explore(color="gray", tiles="Cartodb Positron")
    _m = _nodes.explore(
        m=_m, column=select_node_col.value, marker_kwds={"radius": 5}
    )

    mo.vstack([
        mo.hstack(
            [select_user_id, select_node_col, toggle_polygons], justify="start"
        ),
        _m,
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Sampling and visualising predictions
    """)
    return


@app.cell
def _():
    import numpy as np

    return


@app.cell
def _():
    from activitygraphs.run import load_dataset

    test_size = 0.2
    seed = 42

    train_dataset, test_dataset = load_dataset(cfg, test_size, seed)
    return test_dataset, train_dataset


@app.cell
def _(train_dataset):
    from activitygraphs.run import build_gat, build_mlp

    gat = build_gat(train_dataset, 8, 128, 0.2)
    gat.load_state_dict(torch.load(project_root / "models" / "GATSkip-8-res.pth", weights_only=True))

    mlp = build_mlp(train_dataset, 3, 128, 0.2)
    mlp.load_state_dict(torch.load(project_root / "models" / "MLP.pth", weights_only=True))


    gat, mlp
    return (gat,)


@app.cell
def _(test_dataset):
    index = 1
    data = test_dataset[index]
    user_id = data.user_id
    data
    return data, user_id


@app.cell
def _(data, gat):
    _batch = next(iter(pyg.loader.DataLoader([data])))
    preds = torch.sigmoid(gat(_batch.x, _batch.edge_index, _batch.edge_attr, _batch.batch)).detach().cpu().numpy()
    preds.T
    return (preds,)


@app.cell
def _(network_nodes):
    network_nodes.sort_index()
    return


@app.cell
def _(
    network_edges,
    preds,
    select_node_col,
    select_user_id,
    toggle_polygons,
    user_nodes,
):
    _pred_nodes = user_nodes.copy().sort_index()
    _pred_nodes["predictions"] = preds

    nodes = (
        _pred_nodes.set_geometry("original_geometry")
        if toggle_polygons.value
        else _pred_nodes
    )
    _m = network_edges.explore(color="gray", tiles="Cartodb Positron")
    _m = nodes.explore(
        m=_m, column="predictions", marker_kwds={"radius": 5}
    )

    mo.vstack([
        mo.hstack(
            [select_user_id, select_node_col, toggle_polygons], justify="start"
        ),
        _m,
    ])
    return (nodes,)


@app.cell
def _():
    import contextily as cx

    return (cx,)


@app.cell
def _(CRS, cx, nodes):
    def plot_preds(nodes, col="predictions"):
        nodes = nodes.to_crs(CRS)

        ax = nodes.plot(column=col, figsize=(15, 15), legend=True, cmap="OrRd")
        nodes.boundary.plot(ax=ax, color="lightgrey")
    
        cx.add_basemap(ax, crs=CRS, source=cx.providers.CartoDB.PositronNoLabels)

        return ax

    plot_preds(nodes)
    return (plot_preds,)


@app.cell
def _(plot_preds):
    plot_preds()
    return


if __name__ == "__main__":
    app.run()
