"""
Module: build_pyg_dataset.py

Description:
    A Marimo interactive notebook that converts the multi-layer Geneva transport network
    (built by build_network_graph.py) and the GenevaData user-journey records into a
    PyTorch Geometric (PyG) HeteroData dataset — one graph object per user.

    Workflow:
      Step 1 – Load GenevaData (user journeys, locations) and the Network object.
      Step 2 – Convert the Network into a base pyg.HeteroData object containing network
                topology and node/edge features (no user information yet).
      Step 3 – Use ActivityGraphBuilder to annotate the base graph with per-user home
                indicators and visited-location labels, then persist the resulting dataset.

    The final dataset is used for model training and evaluation in subsequent notebooks.

Dependencies:
    - activitygraphs library (GenevaData, ActivityGraphBuilder, ActivityDataset, network_to_pyg)
    - PyTorch Geometric
    - Marimo, Polars, NetworkX
"""

import marimo

__generated_with = "0.19.8"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    import polars as pl
    from pathlib import Path

    from activitygraphs.config import load_config

    project_root = Path(mo.notebook_dir().parent)
    cfg = load_config(project_root)


@app.cell(hide_code=True)
def _():
    """
    Description: Display a Marimo Markdown top-level heading for this notebook, explaining
    that the notebook converts GenevaData user information and a Network object into a
    PyTorch Geometric HeteroData graph. Pure presentation; no computation.

    Input:
      - (none)

    Output:
      - (none): renders the notebook title heading in the notebook UI.
    """
    mo.md(r"""
    # Building the PyG graph

    Create the `pyg.HeteroData` graph from `GenevaData` user information and `Network` network information
    """)
    return


@app.cell(hide_code=True)
def _():
    """
    Description: Display a Marimo Markdown section heading for Step 1 of the notebook
    pipeline: loading the GenevaData user information and the serialised Network from disk.
    Pure presentation; no computation.

    Input:
      - (none)

    Output:
      - (none): renders the "Step 1" heading in the notebook UI.
    """
    mo.md(r"""
    ## Step 1

    Load the user and network information generated in previous notebooks.
    """)
    return


@app.cell
def _():
    """
    Description: Load the GenevaData object (user journeys + locations) from disk.
    The object bundles cleaned trip records, user journey chains, and the locations
    GeoDataFrame used for building the activity graph.

    Input:
      - (none): depends on ``cfg`` and ``project_root`` from the setup block.

    Output:
      - gva_data (GenevaData): the fully loaded Geneva dataset, exported for all later cells.
    """
    from activitygraphs.data.geneva import GenevaData

    gva_data = GenevaData.load(cfg.data, project_root)
    gva_data
    return (gva_data,)


@app.cell
def _():
    """
    Description: Load the serialised multi-layer Geneva transport Network from disk.
    The network was previously built and saved by ``build_network_graph.py``.
    ``network_name = "routes"`` selects the expanded variant (one PyG node per
    stop-route pair); change to ``"stops"`` for the simpler single-node-per-stop variant.

    Input:
      - (none): depends on ``cfg`` and ``project_root`` from the setup block.

    Output:
      - gva_network (Network): the loaded Network object with all layers and links.
      - network_name (str): the name string used to locate the saved dataset files
        in subsequent cells (``"routes"``).
    """
    from archive.network import Network

    # "routes" selects the expanded-route variant of the network where each
    # (stop, route) pair is a separate node.  Change to "stops" for the simpler
    # single-node-per-stop variant.
    network_name = "routes"

    # Load the serialised Network from the processed-data directory configured in cfg.
    gva_network = Network.load(cfg.data, project_root, network_name)
    gva_network
    return gva_network, network_name


@app.cell(hide_code=True)
def _():
    """
    Description: Display a Marimo Markdown section heading for Step 2 of the pipeline:
    converting the Network object into a base ``pyg.HeteroData`` graph that contains
    topology and edge features but no per-user annotations yet. Pure presentation.

    Input:
      - (none)

    Output:
      - (none): renders the "Step 2" heading in the notebook UI.
    """
    mo.md(r"""
    ## Step 2

    Create the base `pyg.HeteroData` object from the network, without user information
    """)
    return


@app.cell
def _(gva_network):
    """
    Description: Convert the multi-layer Geneva Network object into a base
    ``pyg.HeteroData`` object that contains the graph topology and node/edge features,
    but does not yet contain any per-user information (home indicators, visited labels).
    This base graph is shared across all users and annotated individually in Step 3.

    Input:
      - gva_network (Network): the loaded Geneva network from the previous cell.

    Output:
      - base_data (pyg.HeteroData): the base heterogeneous PyG graph without user data.
    """
    from archive.locations import network_to_pyg

    base_data = network_to_pyg(gva_network)
    base_data
    return (base_data,)


@app.cell(hide_code=True)
def _():
    """
    Description: Display a Marimo Markdown section heading for Step 3 of the pipeline:
    using ActivityGraphBuilder to annotate the base graph with per-user home indicators
    and visited-location labels, then persisting the result as an ActivityDataset.
    Pure presentation; no computation.

    Input:
      - (none)

    Output:
      - (none): renders the "Step 3" heading in the notebook UI.
    """
    mo.md(r"""
    ## Step 3

    Using `ActivityGraphBuilder`, label the base network with user information to create one `pyg.HeteroData` per user, then save it to a `pyg.Dataset`.
    """)
    return


@app.cell
def _(base_data, gva_data, network_name):
    """
    Description: Build (or load from disk) the per-user annotated ``ActivityDataset``.
    If the dataset has already been saved to disk, it is loaded directly to avoid the
    expensive per-user graph annotation loop. Otherwise, ``ActivityGraphBuilder`` is used
    to annotate the base graph for each user (adding home indicators and is_visited labels),
    and the resulting dataset is persisted to disk.

    Input:
      - base_data (pyg.HeteroData): the base PyG graph from Step 2.
      - gva_data (GenevaData): provides ``user_journeys_df`` (trip chains per user).
      - network_name (str): used to name the on-disk files for this dataset variant.

    Output:
      - (none): ``dataset`` is a local variable displayed in the notebook cell output.
    """
    from archive.locations import ActivityGraphBuilder, ActivityDataset

    try:
        # If the dataset has already been built and saved to disk, load it directly
        # to avoid expensive recomputation.
        dataset = ActivityDataset.from_files(cfg.data, project_root, name=network_name)
    except FileNotFoundError:
        # Dataset does not exist yet — build it from scratch.
        # ActivityGraphBuilder annotates base_data with per-user home indicators and
        # visited-location binary labels.  separate_na_source_sink=True means the
        # NA (not-assigned) location is split into a source node (trip origins) and
        # a sink node (trip destinations).
        builder = ActivityGraphBuilder(base_data, gva_data.user_journeys_df, separate_na_source_sink=True)
        # Persist each annotated graph to a separate .pt file on disk.
        dataset = ActivityDataset.from_builder(builder, cfg.data, project_root, name=network_name)

    dataset
    return


@app.cell(hide_code=True)
def _():
    """
    Description: Display a Marimo Markdown top-level heading for the "Analysing the
    generated pyg.HeteroData graph" section. The cells that follow this heading inspect
    structural properties (diameter, journey distributions, purpose statistics) of the
    constructed graph. Pure presentation; no computation.

    Input:
      - (none)

    Output:
      - (none): renders the section heading in the notebook UI.
    """
    mo.md(r"""
    # Analysing the generated `pyg.HeteroData` graph
    """)
    return


@app.cell
def _(base_data):
    """
    Description: Convert the base PyG HeteroData object to a NetworkX MultiDiGraph for
    structural analysis (e.g. connectivity, diameter). The NetworkX representation
    allows using the full NetworkX algorithm library.

    Input:
      - base_data (pyg.HeteroData): the base PyG graph.

    Output:
      - G (nx.MultiDiGraph): the equivalent NetworkX directed multigraph.
    """
    from torch_geometric.utils import to_networkx

    G = to_networkx(base_data, to_multi=True)
    G
    return (G,)


@app.cell
def _(G):
    """
    Description: Compute the diameter of the largest strongly connected component of the
    transport network graph. The diameter is the longest shortest path (in hops) between
    any two nodes in the main component — a measure of how "wide" the network is.
    Isolated nodes and the NA source/sink are excluded because they are not part of the
    main reachable subgraph.

    Input:
      - G (nx.MultiDiGraph): the full NetworkX graph from the previous cell.

    Output:
      - (none): prints the diameter value as a Markdown message in the notebook UI.
    """
    import networkx as nx

    # Exclude isolated nodes and NA source and sink, all other nodes are strongly connected.
    # nx.strongly_connected_components returns a set of sets; we pick the largest one as the
    # "main" component — this is the part of the network reachable from any node to any other.
    main_subgraph = max(nx.strongly_connected_components(G), key=lambda x: len(x))

    # Graph diameter: the longest shortest path between any two nodes in the main component.
    # Gives an idea of how "wide" the network is (in hops).
    diameter = nx.diameter(G.subgraph(main_subgraph))

    mo.md(f"Main graph diameter: {diameter}")
    return


@app.cell
def _(gva_data):
    """
    Description: Analyse the distribution of unique journeys per user by stripping return-
    trip suffixes and counting how many distinct journey IDs each user made. Displays the
    result as a bar chart. This is useful for understanding how many activity patterns
    are available per user.

    Input:
      - gva_data (GenevaData): provides ``user_journeys_df`` with columns ``user_id`` and
        ``journey_id``.

    Output:
      - (none): renders the bar chart in the notebook UI.
    """
    # Strip the "_r" suffix that marks return trips so outward and return legs of the
    # same journey are treated as a single journey.  Then count how many unique journeys
    # each user made, and plot the frequency distribution over the dataset.
    num_unique_journeys_per_user = (
        gva_data.user_journeys_df
        .with_columns(pl.col("journey_id").str.replace("_r", ""))  # normalise journey IDs
        .group_by("user_id")
        .agg(num_journeys=pl.col("journey_id").unique().len())  # count unique journeys per user
        ["num_journeys"]
        .value_counts()  # count how many users have each journey count
        .sort(by="num_journeys")  # sort by journey count for readable chart
    )

    num_unique_journeys_per_user.plot.bar(x="num_journeys:N", y="count").properties(
        title="Freq. of number of unique journeys per user, not incuding return trips"
    )
    return


@app.cell
def _(gva_data):
    """
    Description: Display a bar chart of the frequency of departure activity purposes
    across all user journeys. Helps understand the distribution of trip purposes
    (e.g. home, work, shopping, leisure) in the Geneva dataset.

    Input:
      - gva_data (GenevaData): provides ``user_journeys_df`` with a ``dep_purpose`` column.

    Output:
      - (none): renders the bar chart in the notebook UI.
    """
    gva_data.user_journeys_df["dep_purpose"].value_counts(sort=True).plot.bar(x="dep_purpose:N", y="count")
    return


if __name__ == "__main__":
    app.run()
