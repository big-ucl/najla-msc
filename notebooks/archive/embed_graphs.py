"""
Module: embed_graphs.py

Description:
    A Marimo interactive notebook for embedding household activity graphs into a low-dimensional
    vector space using graph kernels and dimensionality reduction.

    Two parallel approaches are explored:
      1. Graph kernel embedding — compute a kernel (similarity) matrix over all household
         graphs using Weisfeiler-Lehman (WL) or Graphlet-Sampling graph kernels (via GraKeL),
         then apply Kernel PCA and Spectral Clustering to obtain a 3-D embedding.
      2. Metric-vector embedding — read precomputed graph-structural metrics (radius, diameter,
         centrality, etc.) and apply t-SNE to produce a 2-D embedding.

    Results are visualised with Altair heatmaps, scatter plots, and interactive 3-D scatter
    plots.  The kernel-PCA coordinates are saved to a Parquet file for downstream analysis.

Dependencies:
    - activitygraphs library (ActivityDataset, ActivityGraph)
    - GraKeL (graph kernel library)
    - scikit-learn (KernelPCA, SpectralClustering, TSNE)
    - Marimo, Polars, Altair, Matplotlib
"""

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
    """
    Description: Import the core data-manipulation libraries used throughout the notebook.

    Input:
      - (none)

    Output:
      - mo (module): Marimo, for UI widgets and reactive cell control.
      - pl (module): Polars, for fast DataFrame operations.
    """
    import marimo as mo
    import polars as pl

    return mo, pl


@app.cell
def _(mo):
    """
    Description: Load the project configuration from the YAML file located one directory
    above the notebook.

    Input:
      - mo (module): Marimo (provides ``notebook_dir()``).

    Output:
      - cfg: the loaded configuration object with data paths, metric file locations, etc.
    """
    from config import load_config

    cfg = load_config(mo.notebook_dir().parent)
    return (cfg,)


@app.cell
def _(mo):
    """
    Description: Display a Marimo Markdown section heading for the "Data loading" part
    of the notebook. Subsequent cells load the ActivityDataset and build the ActivityGraph
    container. Pure presentation; no computation.

    Input:
      - mo (module): Marimo (for ``mo.md``).

    Output:
      - (none): renders the section heading in the notebook UI.
    """
    mo.md("""## Data loading""")
    return


@app.cell
def _(cfg):
    """
    Description: Load the processed LTDS ``ActivityDataset`` from disk and build an
    ``ActivityGraph`` container from it. The ActivityGraph provides fast access to each
    household's directed multigraph via ``hh_graph(hh_id)`` and ``to_nxs()``.

    Input:
      - cfg: project configuration (provides dataset name and paths).

    Output:
      - ActivityGraph (class): the ActivityGraph class itself; exported so later cells
        can use it as a type annotation.
      - graph (ActivityGraph): the activity graph container for the full LTDS dataset.
    """
    from archive.exploration.dataprocessing import ActivityDataset
    from archive.exploration import ActivityGraph

    dataset = ActivityDataset.load(cfg.data.paths.act_dataset, cfg.data.name)
    graph = ActivityGraph.from_dataset(dataset)
    return ActivityGraph, graph


@app.cell
def _(mo):
    """
    Description: Display a Marimo Markdown section heading for the "Graph kernel embedding"
    part of the notebook. Subsequent cells compute WL / graphlet kernel matrices, apply
    Kernel PCA, run Spectral Clustering, and visualise the results. Pure presentation.

    Input:
      - mo (module): Marimo (for ``mo.md``).

    Output:
      - (none): renders the section heading in the notebook UI.
    """
    mo.md("""## Graph kernel embeddding""")
    return


@app.cell
def _():
    """
    Description: Import the ``draw_hh_graph`` plotting helper for rendering household
    activity graphs as Matplotlib figures.

    Input:
      - (none)

    Output:
      - draw_hh_graph (callable): function that draws a NetworkX household graph as a
        Matplotlib figure with colour-coded nodes and styled edges.
    """
    from plotting import draw_hh_graph

    return (draw_hh_graph,)


@app.cell
def _():
    """
    Description: Import Matplotlib's pyplot interface for creating and displaying figures.

    Input:
      - (none)

    Output:
      - plt (module): ``matplotlib.pyplot`` for figure creation and rendering.
    """
    import matplotlib.pyplot as plt

    return (plt,)


@app.cell
def _():
    """
    Description: Import graph-kernel computation libraries (GraKeL) and the Python
    ``itertools`` module for slicing the household graph generator.

    Input:
      - (none)

    Output:
      - GraphletSampling (class): GraKeL graphlet-sampling kernel implementation.
      - VertexHistogram (class): GraKeL vertex-histogram base kernel (used inside WL kernel).
      - WeisfeilerLehman (class): GraKeL Weisfeiler-Lehman subtree kernel.
      - gk (module): the full ``grakel`` module (used for ``gk.Graph``).
      - itertools (module): Python standard library module (used for ``islice``).
    """
    import grakel as gk
    from grakel.kernels import GraphletSampling, WeisfeilerLehman, VertexHistogram
    import itertools

    return GraphletSampling, VertexHistogram, WeisfeilerLehman, gk, itertools


@app.cell
def _(graph):
    """
    Description: Sanity-check cell that fetches the first two households from the activity
    graph and prints their NetworkX and ActivityGraph representations. Used to verify that
    the graph conversion pipeline works correctly before running expensive kernel computations.

    Input:
      - graph (ActivityGraph): the full activity graph container.

    Output:
      - (none): all variables are cell-local (prefixed with underscore convention).
    """
    # Fetch the first two household graphs as a quick sanity check.
    # _g is a generator that yields (hh_id, nx.MultiDiGraph) tuples.
    _g = graph.to_nxs()

    # Extract the first two households and their NetworkX graph representations
    hh_id_1, G1 = next(_g)  # first household ID and its NetworkX graph
    hh_id_2, G2 = next(_g)  # second household ID and its NetworkX graph

    # Also retrieve ActivityGraph sub-objects (contain the raw Polars DataFrames)
    graph1 = graph.hh_graph(hh_id_1)
    graph2 = graph.hh_graph(hh_id_2)
    return


@app.cell
def _(graph, itertools, n_graphs):
    """
    Description: Build the list of household ActivityGraph objects that will be used
    for kernel computation. Filters out households with no trips (empty edge_df), then
    takes at most ``n_graphs`` households to keep computation tractable.

    Input:
      - graph (ActivityGraph): the full activity graph container.
      - itertools (module): for ``islice`` to cap the number of households.
      - n_graphs (int): maximum number of households to include (from the config cell).

    Output:
      - hh_graphs (list[ActivityGraph subobject]): list of at most ``n_graphs`` non-empty
        household ActivityGraph objects.
      - hh_ids (list[str]): corresponding list of household ID strings, one per graph.
    """
    # Lazily generate ActivityGraph sub-objects for every household
    _all_hh_graphs = (graph.hh_graph(hh_id) for hh_id in graph.hh_ids())

    # Filter out households with no trip edges (empty edge_df) — these would break
    # graph-kernel computation which requires at least one edge.
    _non_empty_hh_graphs = (graph for graph in _all_hh_graphs if not graph.edge_df.is_empty())

    # Take at most n_graphs households for computational tractability
    hh_graphs = list(itertools.islice(_non_empty_hh_graphs, n_graphs))

    # Collect the household ID string for each ActivityGraph (used as row/column labels)
    hh_ids = [g.hh_ids()[0] for g in hh_graphs]
    return hh_graphs, hh_ids


@app.cell
def _(draw_hh_graph, hh_graphs, mo, n_graphs, plt):
    """
    Description: Draw all selected household graphs as individual Matplotlib figures.
    Skipped automatically if ``n_graphs > 5`` to avoid flooding the notebook with plots
    when running large-scale kernel computations.

    Input:
      - draw_hh_graph (callable): the plotting helper function.
      - hh_graphs (list): the list of household ActivityGraph objects.
      - mo (module): Marimo (for ``mo.stop``).
      - n_graphs (int): the configured number of graphs; controls the early-exit guard.
      - plt (module): Matplotlib pyplot for displaying each figure.

    Output:
      - (none): displays one Matplotlib figure per household in the notebook UI.
    """
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
    """
    Description: Define three helper functions for graph-kernel computation, then run the
    Weisfeiler-Lehman (WL) kernel on all selected household graphs to produce a pairwise
    similarity matrix. The three helpers are:
      - ``generate_graph``: converts an ActivityGraph into a GraKeL Graph object.
      - ``compute_graphlet_kernel``: computes a graphlet-sampling kernel matrix.
      - ``compute_wl_kernel``: computes a WL subtree kernel matrix (used here).

    Input:
      - ActivityGraph (class): used as type annotation inside ``generate_graph``.
      - GraphletSampling (class): GraKeL graphlet-sampling kernel (available but not used
        in the main computation; used via ``compute_graphlet_kernel`` if needed).
      - VertexHistogram (class): GraKeL vertex-histogram base kernel for the WL kernel.
      - WeisfeilerLehman (class): GraKeL WL subtree kernel.
      - gk (module): full GraKeL module (provides ``gk.Graph``).
      - hh_graphs (list): list of household ActivityGraph objects to compare.
      - n_kernel_iter (int): number of WL refinement iterations.
      - pl (module): Polars (used inside ``generate_graph`` to build edge/node tables).

    Output:
      - kernel_matrix (np.ndarray): shape (n_graphs, n_graphs) normalised WL kernel
        similarity matrix, where entry [i, j] is the kernel similarity between household i
        and household j.
    """
    def generate_graph(subgraph: ActivityGraph):
        """
        Description:
            Converts an ActivityGraph sub-object into a GraKeL Graph object that can be
            consumed by graph-kernel functions.

        Input:
          - subgraph (ActivityGraph): a single household's activity graph containing
                node_df (locations with purpose flags) and edge_df (trips with durations).

        Output:
          - (gk.Graph): a GraKeL graph with edge-weight attributes (trip duration),
                node label attributes (activity purposes), and edge label attributes
                (person IDs).
        """
        # Build the edge list as (origin, destination, duration) triples
        edgelist = subgraph.edge_df.select("loc_origin_loc_id", "loc_dest_loc_id", "duration").rows()

        # Node labels: dict mapping loc_id -> purposes bitmask (used by WL kernel)
        node_labels = subgraph.node_df.select("loc_id", pl.col("purposes")).rows_by_key("loc_id", unique=True)

        # Edge labels: dict mapping (origin, dest) -> person_id (identifies who made the trip)
        edge_labels = subgraph.edge_df.select("loc_origin_loc_id", "loc_dest_loc_id", "person_id").rows_by_key(
            ["loc_origin_loc_id", "loc_dest_loc_id"], unique=True
        )  # TODO remove

        return gk.Graph(edgelist, node_labels=node_labels, edge_labels=edge_labels)

    def compute_graphlet_kernel(*subgraphs: ActivityGraph):
        """
        Description:
            Computes a Graphlet Sampling kernel matrix over a collection of activity graphs.
            The graphlet kernel measures graph similarity by counting occurrences of small
            subgraph patterns (graphlets) of size k.

        Input:
          - *subgraphs (ActivityGraph): one or more ActivityGraph objects to compare.

        Output:
          - (np.ndarray): an N x N symmetric matrix of pairwise kernel values, normalised
                to the range [0, 1].
        """
        gk = GraphletSampling(random_state=1, k=4, sampling={"a": -1}, normalize=True)
        return gk.fit_transform(map(generate_graph, subgraphs))

    def compute_wl_kernel(*subgraphs: ActivityGraph, n_iter=2):
        """
        Description:
            Computes a Weisfeiler-Lehman (WL) subtree kernel matrix over activity graphs.
            The WL kernel iteratively aggregates neighbourhood labels and computes histogram
            overlap — providing a powerful, scalable graph similarity measure.

        Input:
          - *subgraphs (ActivityGraph): one or more ActivityGraph objects to compare.
          - n_iter (int): number of WL refinement iterations. More iterations capture larger
                neighbourhood structure. Defaults to 2.

        Output:
          - (np.ndarray): an N x N symmetric normalised kernel matrix.
        """
        gk = WeisfeilerLehman(n_iter=3, normalize=True, base_graph_kernel=VertexHistogram)
        return gk.fit_transform(map(generate_graph, subgraphs))

    # Compute the WL kernel matrix for the selected household graphs.
    # Shape: (n_graphs, n_graphs), values close to 1 mean structurally similar graphs.
    kernel_matrix = compute_wl_kernel(*hh_graphs, n_iter=n_kernel_iter)
    return (kernel_matrix,)


@app.cell
def _(mo, n_graphs, plot_heatmap, similarities):
    """
    Description: Render the interactive kernel-similarity heatmap. Skipped automatically
    when ``n_graphs >= 30`` because the heatmap becomes too dense to be readable at
    large scale.

    Input:
      - mo (module): Marimo (for ``mo.stop``).
      - n_graphs (int): the configured number of graphs; controls the early-exit guard.
      - plot_heatmap (callable): the heatmap plotting function from the functions cell.
      - similarities (pl.DataFrame): long-form similarity DataFrame (hh_id_1, hh_id_2, similarity).

    Output:
      - (none): renders the composite Altair heatmap chart in the notebook UI.
    """
    mo.stop(n_graphs >= 30)

    plot_heatmap(similarities)
    return


@app.cell
def _():
    """
    Description: Central configuration cell for the kernel embedding experiment.
    Adjust these three integers to trade off computation time vs. embedding quality.

    Input:
      - (none)

    Output:
      - n_graphs (int): maximum number of household graphs to include in the kernel matrix.
        1000 is a reasonable default that completes in a few minutes on a modern CPU.
      - n_kernel_iter (int): number of WL refinement iterations. More iterations capture
        larger neighbourhood structures but increase computation time.
      - n_clusters (int): number of spectral clusters to assign to the kernel-PCA embeddings.
    """
    n_graphs = 10_00         # number of household graphs to include in kernel computation (1000)
    n_kernel_iter = 3        # WL kernel refinement iterations; larger = more structure captured
    n_clusters = 5           # number of clusters for SpectralClustering
    return n_clusters, n_graphs, n_kernel_iter


@app.cell
def _(extra_data, hh_ids, kernel_matrix, n_clusters, pl):
    """
    Description: Reduce the kernel similarity matrix to a 3-D embedding using Kernel PCA,
    cluster the embeddings with Spectral Clustering, and collect everything (including
    graph-size statistics) into a single Polars DataFrame for downstream visualisation and
    export.

    Input:
      - extra_data (pl.DataFrame): graph-size stats (hh_id, n_nodes, n_edges).
      - hh_ids (list[str]): household IDs in the same order as kernel_matrix rows/columns.
      - kernel_matrix (np.ndarray): (n_graphs, n_graphs) WL kernel similarity matrix.
      - n_clusters (int): number of spectral clusters.
      - pl (module): Polars for building the results DataFrame.

    Output:
      - results (pl.DataFrame): DataFrame with columns: hh_id, x, y, z (PCA coords),
        c (cluster label), n_nodes, n_edges.
    """
    from sklearn.decomposition import KernelPCA
    from sklearn.cluster import SpectralClustering

    # KernelPCA with a precomputed kernel reduces the N x N kernel matrix to a 3-D
    # coordinate for each graph, preserving as much variance as possible.
    pca = KernelPCA(n_components=3, kernel="precomputed")

    # SpectralClustering groups the 3-D embeddings into n_clusters groups.
    cluster = SpectralClustering(n_clusters=n_clusters)

    # results_np: shape (n_graphs, 3) — each row is the 3-D PCA embedding of one household
    results_np = pca.fit_transform(kernel_matrix)
    # cluster_np: shape (n_graphs,) — integer cluster label 0..n_clusters-1 per household
    cluster_np = cluster.fit_predict(results_np)

    # Collect results into a Polars DataFrame and join with extra node/edge count data
    results = pl.DataFrame({
        "hh_id": hh_ids,     # household identifier
        "x": results_np[:, 0],  # first principal component (embedding dimension 1)
        "y": results_np[:, 1],  # second principal component (embedding dimension 2)
        "z": results_np[:, 2],  # third principal component (embedding dimension 3)
        "c": cluster_np,         # cluster assignment
    }).join(extra_data, on="hh_id")  # append n_nodes and n_edges columns
    return (results,)


@app.cell
def _(results):
    """
    Description: Persist the kernel-PCA embedding results (household IDs, 3-D coordinates,
    cluster assignments, graph sizes) to a Parquet file for use in downstream analysis
    or other notebooks.

    Input:
      - results (pl.DataFrame): the embedding DataFrame from the PCA cell.

    Output:
      - (none): writes ``data/processed/TEST_graphs.parquet`` to disk as a side effect.
    """
    results.write_parquet("data/processed/TEST_graphs.parquet")
    return


@app.cell
def _(alt, mo, n_graphs, results):
    """
    Description: Create an Altair 2-D scatter plot of the first two kernel-PCA dimensions,
    with points coloured by the third PCA dimension and shaped by cluster assignment.
    Skipped for very large graphs (>= 5000) to avoid browser rendering slowdowns.

    Input:
      - alt (module): Altair for declarative chart specification.
      - mo (module): Marimo (for ``mo.stop``).
      - n_graphs (int): the configured graph count; used for the early-exit guard.
      - results (pl.DataFrame): the embedding DataFrame.

    Output:
      - (none): renders the Altair scatter chart in the notebook UI.
    """
    mo.stop(n_graphs >= 5000)

    alt.Chart(results).mark_point().encode(x="x", y="y", color="z", shape="c:N", tooltip=["hh_id", "c"])
    return


@app.cell
def _(results):
    """
    Description: Display the full embedding results DataFrame for tabular inspection.

    Input:
      - results (pl.DataFrame): embedding DataFrame with hh_id, PCA coords, cluster, sizes.

    Output:
      - (none): renders the DataFrame in the notebook UI.
    """
    results
    return


@app.cell
def _(mo, n_graphs, plt, results):
    """
    Description: Create a 3-D Matplotlib scatter plot of all three kernel-PCA dimensions,
    with point colour representing the number of edges (trips) per household. Skipped for
    very large graphs (>= 5000) to avoid long rendering times.

    Input:
      - mo (module): Marimo (for ``mo.stop``).
      - n_graphs (int): the configured graph count; used for the early-exit guard.
      - plt (module): Matplotlib pyplot.
      - results (pl.DataFrame): the embedding DataFrame with x, y, z, n_edges columns.

    Output:
      - (none): displays the 3-D scatter plot in the notebook UI.
    """
    mo.stop(n_graphs >= 5000)

    _fig = plt.figure(figsize=(6, 6))
    ax = _fig.add_subplot(projection="3d")  # create a 3-D axes subplot

    ax.scatter(results["x"], results["y"], results["z"], c=results["n_edges"])
    plt.show()
    return


@app.cell
def _(hh_ids, kernel_matrix, pl):
    """
    Description: Reshape the (n_graphs, n_graphs) kernel matrix from wide NumPy format into
    a long-form Polars DataFrame suitable for Altair heatmap visualisation. Each row in the
    output represents one (household_1, household_2, similarity) pair.

    Input:
      - hh_ids (list[str]): household IDs used as column names in the wide format.
      - kernel_matrix (np.ndarray): (n_graphs, n_graphs) kernel similarity matrix.
      - pl (module): Polars for DataFrame manipulation.

    Output:
      - similarities (pl.DataFrame): long-form DataFrame with columns:
          - hh_id_1 (str): row household ID
          - hh_id_2 (str): column household ID
          - similarity (float): pairwise kernel similarity value in [0, 1]
    """
    # Convert the (n_graphs, n_graphs) NumPy kernel matrix to a wide Polars DataFrame
    # where columns are labelled by household ID.
    source = pl.from_numpy(kernel_matrix, schema=hh_ids)

    # Reshape to long form (hh_id_1, hh_id_2, similarity) for Altair visualisation
    similarities = pl.concat([pl.DataFrame({"hh_id_1": hh_ids}), source], how="horizontal").unpivot(
        index="hh_id_1",          # keep hh_id_1 as identifier column
        variable_name="hh_id_2",  # column names become a new category column
        value_name="similarity",  # kernel values become the value column
    )
    return (similarities,)


@app.cell
def _(results):
    """
    Description: Second display cell for the embedding results (for quick tabular reference
    after the 3-D scatter plot cell).

    Input:
      - results (pl.DataFrame): the embedding DataFrame.

    Output:
      - (none): renders the DataFrame in the notebook UI.
    """
    results
    return


@app.cell
def _(hh_graphs, hh_ids, pl):
    """
    Description: Build a supplementary DataFrame containing simple graph-size statistics
    (node count and edge count) for each household graph. These statistics are joined onto
    the embedding results and used to colour the annotation bars in the heatmap.

    Input:
      - hh_graphs (list): list of household ActivityGraph objects.
      - hh_ids (list[str]): household IDs in the same order as hh_graphs.
      - pl (module): Polars for DataFrame construction.

    Output:
      - extra_data (pl.DataFrame): DataFrame with columns:
          - hh_id (str): household identifier
          - n_nodes (int): number of unique visited locations in this household's graph
          - n_edges (int): number of trips (graph edges) in this household's graph
    """
    # Build a supplementary DataFrame with simple graph-size statistics.
    # n_nodes: number of unique visited locations for this household
    # n_edges: number of trips made by this household (including duplicates)
    # These are used to annotate the heatmap and scatter plots.
    extra_data = pl.DataFrame({
        "hh_id": hh_ids,                                       # household identifier
        "n_nodes": [g.node_df.height for g in hh_graphs],     # number of graph nodes
        "n_edges": [g.edge_df.height for g in hh_graphs],     # number of graph edges (trips)
    })
    return (extra_data,)


@app.cell
def _():
    """
    Description: Import Altair for declarative, interactive chart creation.

    Input:
      - (none)

    Output:
      - alt (module): the Altair charting library.
    """
    import altair as alt

    return (alt,)


@app.cell
def _(alt, extra_data):
    """
    Description: Define two helper functions for visualising the kernel similarity matrix:
      - ``plot_extra_data``: builds a colour-coded annotation bar chart (n_nodes or n_edges)
        to place alongside the heatmap, with interactive hover highlighting.
      - ``plot_heatmap``: assembles the full composite Altair chart — a central similarity
        heatmap flanked by annotation bars for row and column households.
    Neither function is called here; they are exported for the display cell that follows.

    Input:
      - alt (module): Altair for declarative chart specification.
      - extra_data (pl.DataFrame): graph-size stats (hh_id, n_nodes, n_edges) used inside
        ``plot_heatmap`` to build annotation bars.

    Output:
      - plot_heatmap (callable): function that takes a long-form ``similarities`` DataFrame
        and returns a composite Altair chart with similarity heatmap and annotation bars.
    """
    def plot_extra_data(extra_data, axis="x", data="n_nodes", selection=None):
        """
        Description:
            Creates a colour-coded bar annotation chart to be placed alongside the similarity
            heatmap.  Each bar corresponds to one household and is coloured by a chosen graph
            statistic (n_nodes or n_edges).  Hovering over a cell in the heatmap highlights
            the corresponding bar in pink via the interactive selection.

        Input:
          - extra_data (pl.DataFrame): DataFrame with columns "hh_id", "n_nodes", "n_edges".
          - axis (str): "x" to orient bars horizontally (column annotations),
                "y" for vertical (row annotations).  Defaults to "x".
          - data (str): which column to use for colour encoding.  Defaults to "n_nodes".
          - selection (alt.Selection | None): an Altair interactive selection; bars that
                match the selection are coloured pink.  Defaults to None (no highlighting).

        Output:
          - (alt.Chart): a combined Altair chart of colour bars + text labels.
        """
        # Choose the Altair encoding class (X or Y) based on the desired axis
        data_cls = alt.X if axis == "x" else alt.Y
        # Choose the ID column name depending on which axis this annotation is for
        hh_id_name = "hh_id_2" if axis == "x" else "hh_id_1"

        # Rename so the column matches the heatmap's expected field names
        extra_data = extra_data.rename({"hh_id": hh_id_name})

        if selection is not None:
            # Highlight matching household in pink; otherwise colour by the data column
            color = alt.when(selection).then(alt.value("pink")).otherwise(alt.Color(f"{data}:O", legend=None))
        else:
            color = alt.Color(f"{data}:O", legend=None)  # plain colour scale

        main_kwargs = {
            axis: data_cls(
                hh_id_name,
                axis=alt.Axis(title=data, labels=False, ticks=False),  # hide axis labels for compactness
            ),
            "color": color,
        }

        # Coloured rectangle bars
        main = (
            alt
            .Chart(extra_data)
            .mark_rect()
            .encode(
                **main_kwargs,
            )
        )

        # Text overlay showing the numeric value
        text_kwargs = {axis: hh_id_name, "text": f"{data}:O"}
        text = alt.Chart(extra_data).mark_text().encode(**text_kwargs)

        return main + text

    def plot_heatmap(similarities):
        """
        Description:
            Creates a composite Altair chart consisting of a similarity heatmap in the
            centre with n_nodes and n_edges annotation bars along the sides.  Interactive
            hover selections cross-highlight the corresponding annotation bars.

        Input:
          - similarities (pl.DataFrame): long-form similarity DataFrame with columns
                "hh_id_1", "hh_id_2", "similarity".

        Output:
          - (alt.Chart): the composite Altair chart ready for display.
        """
        # Pointer-hover selections: hovering row/column highlights its annotation bar
        select_hh_1 = alt.selection_point(on="pointerover", fields=["hh_id_1"], empty=False)
        select_hh_2 = alt.selection_point(on="pointerover", fields=["hh_id_2"], empty=False)

        # Central heatmap: y=row household, x=column household, colour=kernel similarity
        heatmap = (
            alt
            .Chart(similarities)
            .mark_rect()
            .encode(
                y="hh_id_1:N",
                x="hh_id_2:N",
                color="similarity:Q",
                tooltip=["similarity"],
            )
        ).add_params(select_hh_1, select_hh_2)

        # Composite layout: annotation bars | heatmap, stacked with bottom bars
        chart = (
            plot_extra_data(extra_data, "y", data="n_nodes", selection=select_hh_1)
            | plot_extra_data(extra_data, "y", data="n_edges", selection=select_hh_1)
            | heatmap
            & plot_extra_data(extra_data, "x", data="n_edges", selection=select_hh_2)
            & plot_extra_data(extra_data, "x", data="n_nodes", selection=select_hh_2)
        ).resolve_scale(x="shared", color="independent")  # independent colour scales per sub-chart

        return chart

    return (plot_heatmap,)


@app.cell
def _(mo):
    """
    Description: Display a Marimo Markdown planning note outlining the two embedding
    strategies explored in this notebook:
      1. Existing graph-structural metrics -> vector -> t-SNE.
      2. Graph kernel similarity matrix -> vector -> t-SNE.
    Pure presentation; no computation.

    Input:
      - mo (module): Marimo (for ``mo.md``).

    Output:
      - (none): renders the planning note in the notebook UI.
    """
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
    """
    Description: Display a Marimo Markdown section heading for the "Embedding of existing
    results" part of the notebook. Subsequent cells load precomputed graph-structural
    metrics, sample a subset, and run t-SNE to produce a 2-D embedding. Pure presentation.

    Input:
      - mo (module): Marimo (for ``mo.md``).

    Output:
      - (none): renders the section heading in the notebook UI.
    """
    mo.md(r"""## Embedding of existing results""")
    return


@app.cell
def _(cfg, pl):
    """
    Description: Load the precomputed graph-structural metrics from Parquet and impute
    known null patterns before use in t-SNE embedding. Some metrics (e.g. radius and
    diameter of disconnected graphs, or centrality for single-node graphs) can be null
    or NaN; these are filled with sensible defaults so the t-SNE algorithm receives a
    complete feature matrix.

    Input:
      - cfg: project configuration (provides ``cfg.data.paths.metrics``).
      - pl (module): Polars for reading and filling the DataFrame.

    Output:
      - metrics (pl.DataFrame): the metrics DataFrame with nulls filled:
          - radius/diameter null -> 0 (isolated or trivial graph)
          - assortativity null -> 0 (undefined for small graphs)
          - centrality nulls -> -1 (sentinel for "not computable")
    """
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
    """
    Description: Draw a random sample of 500 households from the metrics DataFrame for
    fast t-SNE prototyping. Running t-SNE on the full dataset can take a long time;
    500 samples complete in seconds while still showing the overall structure.

    Input:
      - metrics (pl.DataFrame): the full imputed metrics DataFrame.

    Output:
      - test (pl.DataFrame): a random 500-row sample of the metrics DataFrame.
    """
    test = metrics.sample(500)
    return (test,)


@app.cell
def _(test):
    """
    Description: Run t-SNE dimensionality reduction on the sampled metric feature matrix
    to produce a 2-D embedding. Uses automatic learning rate and random initialisation;
    runs in parallel on all available CPU cores.

    Input:
      - test (pl.DataFrame): the 500-row sample of graph metrics (all columns except hh_id).

    Output:
      - X_embedded (np.ndarray): shape (500, 2) — the 2-D t-SNE coordinates for each
        sampled household, preserving local neighbourhood structure from the metric space.
    """
    from sklearn.manifold import TSNE

    X = test.drop("hh_id").to_numpy()  # drop the ID column; only keep numeric metric values
    X_embedded = TSNE(n_components=2, learning_rate="auto", init="random", n_jobs=-1, verbose=1).fit_transform(X)
    return (X_embedded,)


@app.cell
def _(X_embedded, plt):
    """
    Description: Display a 2-D scatter plot of the t-SNE embedding coordinates.
    Note: the current code plots X_embedded[:, 0] on both axes (likely a bug —
    dimension 1 should be X_embedded[:, 1]).

    Input:
      - X_embedded (np.ndarray): shape (500, 2) t-SNE coordinates.
      - plt (module): Matplotlib pyplot.

    Output:
      - (none): renders the scatter plot in the notebook UI.
    """
    plt.scatter(X_embedded[:, 0], X_embedded[:, 0])
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
