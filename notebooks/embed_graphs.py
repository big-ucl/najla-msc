import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells
    import numpy as np
    import random

    # Set random seeds
    np.random.seed(42)
    random.seed(42)


@app.cell
def _():
    import marimo as mo
    import polars as pl

    from pathlib import Path
    return mo, pl


@app.cell
def _(mo):
    from config import load_config

    cfg = load_config(mo.notebook_dir().parent)
    return (cfg,)


@app.cell
def _(mo):
    mo.md("""## Data loading""")
    return


@app.cell
def _(cfg):
    from exploration.dataprocessing import ActivityDataset
    from exploration.graphs import ActivityGraph

    dataset = ActivityDataset.load(cfg.data.paths.act_dataset, cfg.data.name)
    graph = ActivityGraph.from_dataset(dataset)
    return ActivityGraph, graph


@app.cell
def _(mo):
    mo.md("""## Graph kernel embeddding""")
    return


@app.cell
def _():
    import networkx as nx
    from plotting import draw_hh_graph
    return (draw_hh_graph,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    return (plt,)


@app.cell
def _():
    import grakel as gk
    from grakel.kernels import GraphletSampling, WeisfeilerLehman, VertexHistogram
    import itertools
    return GraphletSampling, VertexHistogram, WeisfeilerLehman, gk, itertools


@app.cell
def _(graph):
    _g = graph.to_nxs()


    hh_id_1, G1 = next(_g)
    hh_id_2, G2 = next(_g)

    graph1 = graph.hh_graph(hh_id_1)
    graph2 = graph.hh_graph(hh_id_2)
    return


@app.cell
def _(graph, itertools, n_graphs):
    _all_hh_graphs = (graph.hh_graph(hh_id) for hh_id in graph.hh_ids())
    _non_empty_hh_graphs = (graph for graph in _all_hh_graphs if not graph.edge_df.is_empty())
    hh_graphs = list(itertools.islice(_non_empty_hh_graphs, n_graphs))
    hh_ids = [g.hh_ids()[0] for g in hh_graphs]
    return hh_graphs, hh_ids


@app.cell
def _(draw_hh_graph, hh_graphs, mo, n_graphs, plt):
    mo.stop(n_graphs > 5)

    for h in hh_graphs:
        id, g = next(h.to_nxs())
        draw_hh_graph(
            g,
            id,
        )
        plt.show()
    return


@app.cell
def _(
    ActivityGraph,
    GraphletSampling,
    VertexHistogram,
    WeisfeilerLehman,
    gk,
    hh_graphs,
    n_kernel_iter,
    pl,
):
    def generate_graph(subgraph: ActivityGraph):
        edgelist = subgraph.edge_df.select("loc_origin_loc_id", "loc_dest_loc_id", "duration").rows()

        node_labels = subgraph.node_df.select("loc_id", pl.col("purposes")).rows_by_key("loc_id", unique=True)

        edge_labels = subgraph.edge_df.select("loc_origin_loc_id", "loc_dest_loc_id", "person_id").rows_by_key(
            ["loc_origin_loc_id", "loc_dest_loc_id"], unique=True
        ) # TODO remove

        return gk.Graph(edgelist, node_labels=node_labels, edge_labels=edge_labels)


    def compute_graphlet_kernel(*subgraphs: ActivityGraph):
        gk = GraphletSampling(random_state=1, k=4, sampling={"a": -1}, normalize=True)
        return gk.fit_transform(map(generate_graph, subgraphs))


    def compute_wl_kernel(*subgraphs: ActivityGraph, n_iter=2):
        gk = WeisfeilerLehman(n_iter=3, normalize=True, base_graph_kernel=VertexHistogram)
        return gk.fit_transform(map(generate_graph, subgraphs))


    kernel_matrix = compute_wl_kernel(*hh_graphs, n_iter=n_kernel_iter)
    return (kernel_matrix,)


@app.cell
def _(mo, n_graphs, plot_heatmap, similarities):
    mo.stop(n_graphs >= 30)

    plot_heatmap(similarities)
    return


@app.cell
def _():
    n_graphs = 10_00
    n_kernel_iter = 3
    n_clusters = 5
    return n_clusters, n_graphs, n_kernel_iter


@app.cell
def _(extra_data, hh_ids, kernel_matrix, n_clusters, pl):
    from sklearn.decomposition import KernelPCA
    from sklearn.cluster import SpectralClustering

    pca = KernelPCA(n_components=3, kernel="precomputed")
    cluster = SpectralClustering(n_clusters=n_clusters)

    results_np = pca.fit_transform(kernel_matrix)
    cluster_np = cluster.fit_predict(results_np)

    results = pl.DataFrame(
        {
            "hh_id": hh_ids,
            "x": results_np[:, 0],
            "y": results_np[:, 1],
            "z": results_np[:, 2],
            "c": cluster_np,
        }
    ).join(extra_data, on="hh_id")
    return (results,)


@app.cell
def _(results):
    results.write_parquet("data/processed/TEST_graphs.parquet")
    return


@app.cell
def _(alt, mo, n_graphs, results):
    mo.stop(n_graphs >= 5000)

    alt.Chart(results).mark_point().encode(x="x", y="y", color="z", shape="c:N", tooltip=["hh_id", "c"])
    return


@app.cell
def _(results):
    results
    return


@app.cell
def _(mo, n_graphs, plt, results):
    mo.stop(n_graphs >= 5000)

    _fig = plt.figure(figsize=(6, 6))
    ax = _fig.add_subplot(projection="3d")

    ax.scatter(results["x"], results["y"], results["z"], c=results["n_edges"])
    plt.show()
    return


@app.cell
def _(hh_ids, kernel_matrix, pl):
    source = pl.from_numpy(kernel_matrix, schema=hh_ids)
    similarities = pl.concat([pl.DataFrame({"hh_id_1": hh_ids}), source], how="horizontal").unpivot(
        index="hh_id_1", variable_name="hh_id_2", value_name="similarity"
    )
    return (similarities,)


@app.cell
def _(results):
    results
    return


@app.cell
def _(hh_graphs, hh_ids, pl):
    extra_data = pl.DataFrame(
        {
            "hh_id": hh_ids,
            "n_nodes": [g.node_df.height for g in hh_graphs],
            "n_edges": [g.edge_df.height for g in hh_graphs],
        }
    )
    return (extra_data,)


@app.cell
def _():
    import altair as alt
    return (alt,)


@app.cell
def _(alt, extra_data):
    def plot_extra_data(extra_data, axis="x", data="n_nodes", selection=None):
        data_cls = alt.X if axis == "x" else alt.Y
        hh_id_name = "hh_id_2" if axis == "x" else "hh_id_1"

        extra_data = extra_data.rename({"hh_id": hh_id_name})

        if selection is not None:
            color = alt.when(selection).then(alt.value("pink")).otherwise(alt.Color(f"{data}:O", legend=None))
        else:
            color = alt.Color(f"{data}:O", legend=None)

        main_kwargs = {
            axis: data_cls(
                hh_id_name,
                axis=alt.Axis(title=data, labels=False, ticks=False),
            ),
            "color": color,
        }

        main = (
            alt.Chart(extra_data)
            .mark_rect()
            .encode(
                **main_kwargs,
            )
        )

        text_kwargs = {axis: hh_id_name, "text": f"{data}:O"}
        text = alt.Chart(extra_data).mark_text().encode(**text_kwargs)

        return main + text


    def plot_heatmap(similarities):
        select_hh_1 = alt.selection_point(on="pointerover", fields=["hh_id_1"], empty=False)

        select_hh_2 = alt.selection_point(on="pointerover", fields=["hh_id_2"], empty=False)

        heatmap = (
            alt.Chart(similarities)
            .mark_rect()
            .encode(
                y="hh_id_1:N",
                x="hh_id_2:N",
                color="similarity:Q",
                tooltip=["similarity"],
            )
        ).add_params(select_hh_1, select_hh_2)

        chart = (
            plot_extra_data(extra_data, "y", data="n_nodes", selection=select_hh_1)
            | plot_extra_data(extra_data, "y", data="n_edges", selection=select_hh_1)
            | heatmap
            & plot_extra_data(extra_data, "x", data="n_edges", selection=select_hh_2)
            & plot_extra_data(extra_data, "x", data="n_nodes", selection=select_hh_2)
        ).resolve_scale(x="shared", color="independent")

        return chart
    return (plot_heatmap,)


@app.cell
def _(mo):
    mo.md(
        r"""
    Plan:
     - Existing Metrics -> Vector -> tSNE
     - Graph kernel -> Vector -> tSNE
     -
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Embedding of existing results""")
    return


@app.cell
def _(cfg, pl):
    metrics = pl.read_parquet(cfg.data.paths.metrics)
    metrics = metrics.with_columns(
        pl.col("radius_(duration)").fill_null(0),
        pl.col("diameter_(duration)").fill_null(0),
        pl.col("assortativity-duration").fill_null(0),
        pl.col("degree_central").fill_null(-1),
        pl.col("betweeness_central_(duration)").fill_null(-1),
        pl.col("closeness_central_(duration)").fill_null(-1),
        pl.col("katz_central_(duration)").fill_null(-1),
    )

    metrics
    return (metrics,)


@app.cell
def _(metrics):
    test = metrics.sample(500)
    return (test,)


@app.cell
def _(test):
    from sklearn.manifold import TSNE

    X = test.drop("hh_id").to_numpy()
    X_embedded = TSNE(n_components=2, learning_rate="auto", init="random", n_jobs=-1, verbose=1).fit_transform(X)
    return (X_embedded,)


@app.cell
def _(X_embedded, plt):
    plt.scatter(X_embedded[:, 0], X_embedded[:, 0])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
