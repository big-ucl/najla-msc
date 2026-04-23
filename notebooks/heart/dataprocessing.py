import marimo

__generated_with = "0.21.1"
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

    gva_data = GenevaData.load(cfg.data, project_root)
    return (gva_data,)


@app.cell
def _(gva_data):
    from activitygraphs.dataprocessing import load_gva_network_graph

    network_nodes, network_edges = load_gva_network_graph(gva_data, cfg.data, project_root)
    return network_edges, network_nodes


@app.cell
def _(gva_data):
    visits = gva_data.with_filter("subsector").location_visits
    visits_by_purpose = visits.group_by("user_id", "purpose").agg(pl.col("loc_id").unique()).sort("user_id")
    visits_by_purpose
    return visits, visits_by_purpose


@app.cell
def _(gva_data, home_locations):
    home_locations2 = gva_data.location_visits_by_purpose.filter(purpose="od_lieu_domicile").with_columns(pl.col("loc_id").list.first())

    home_locations.equals(gva_data.filter_by_loc_type(home_locations2, "subsector"))
    return


@app.cell
def _(gva_data):
    gva_data.locations_df
    return


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

        visit_purposes = (
            visits
            .filter(user_id=user_id)
            .select("loc_id", pl.col("purpose"))
            .group_by("loc_id")
            .agg(pl.col("purpose").str.join(", "))
            .to_pandas()
        )
        nodes = nodes.merge(visit_purposes, on="loc_id", how="left")
        nodes = nodes.set_index("loc_id").sort_index()

        return nodes

    _user_id = "20704"
    add_user_cols(locations, _user_id)
    return add_user_cols, home_locations


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Adding POI and land use information
    """)
    return


@app.cell
def _():
    import geopandas as gpd
    import pandas as pd

    return (gpd,)


@app.cell
def _(locations):
    CRS = "EPSG:4326"
    utm_crs = locations.estimate_utm_crs()
    return (CRS,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Load external data
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Enhance node features
    """)
    return


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
    return c2g, dataset_path


@app.cell
def _(visits):
    user_ids = visits["user_id"].unique().sort()
    return (user_ids,)


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
        node_feature_cols = [col for col in indiv_nodes.columns if col not in _excluded_feature_cols]

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

    _graphs = Parallel(n_jobs=-1)(delayed(build_graph)(user_id) for user_id in user_ids)

    with open(dataset_path, "wb") as _f:
        pickle.dump(_graphs, _f)
    return


@app.cell
def _(dataset_path):
    with open(dataset_path, "rb") as _f:
        pass  # graphs = pickle.load(_f)

    # graphs
    return


@app.cell
def _(user_id, user_ids):
    select_user_id = mo.ui.dropdown(user_ids, value=user_id, label="User ID:", searchable=True)
    return (select_user_id,)


@app.cell
def _(add_user_cols, network_nodes, select_user_id):
    _user_id = select_user_id.value
    user_nodes = add_user_cols(network_nodes, _user_id)
    return (user_nodes,)


@app.cell
def _(user_nodes):
    select_node_col = mo.ui.dropdown(list(user_nodes.columns), searchable=True, label="Column:", value="purpose")
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
    _nodes = user_nodes.set_geometry("original_geometry") if toggle_polygons.value else user_nodes
    _m = network_edges.explore(color="gray", tiles="Cartodb Positron")
    _m = _nodes.explore(m=_m, column=select_node_col.value, marker_kwds={"radius": 5})

    mo.vstack([
        mo.hstack([select_user_id, select_node_col, toggle_polygons], justify="start"),
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
    return


@app.cell
def _():
    from activitygraphs.run import load_dataset

    mo.stop(True)

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

    nodes = _pred_nodes.set_geometry("original_geometry") if toggle_polygons.value else _pred_nodes
    _m = network_edges.explore(color="gray", tiles="Cartodb Positron")
    _m = nodes.explore(m=_m, column="predictions", marker_kwds={"radius": 5})

    mo.vstack([
        mo.hstack([select_user_id, select_node_col, toggle_polygons], justify="start"),
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
