"""
Module: notebooks/heart/sampling.py

Description:
    A Marimo interactive notebook for loading trained models, making predictions and
    visualising two probabilistic sampling strategies on the Geneva network:

      1. Poisson sampling: each node is independently included with probability p_i,
         yielding a variable-size location choice set.
      2. PPS (Probability-Proportional-to-Size) sampling: a fixed number n of nodes
         are sampled without replacement where inclusion probability is proportional to
         the predicted visit probability.

    Four models are compared: GATSkipRes, NodeMLP, NodeBaseline, ConditionalNodeBaseline.
    Predictions and samples are attached to the geographic network GeoDataFrame and
    visualised on interactive maps.

Dependencies:
    - activitygraphs library (GenevaData, load_gva_network_graph, add_user_cols,
      build_gat, build_mlp, NodeBaseline, ConditionalNodeBaseline,
      poisson_sampling, pps_sampling)
    - PyTorch, PyTorch Geometric
    - Marimo, GeoPandas, Contextily, Matplotlib, Seaborn
"""

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")

with app.setup:
    from pathlib import Path

    import geopandas as gpd
    import marimo as mo
    import pyarrow  # noqa: F401
    import pandas  # noqa: F401
    import polars as pl
    import torch
    import torch_geometric as pyg

    from activitygraphs.config import load_config

    project_root = Path(mo.notebook_dir().parent.parent)
    cfg = load_config(project_root)


@app.cell
def _():
    """
    Description:
        Defines shared configuration constants used throughout the sampling notebook:
        the path to saved model checkpoints, the DataLoader batch size, the test
        fraction, and the random seed.

    Output:
      - batch_size (int): number of graphs processed together during inference.
      - models_path (Path): absolute path to the directory of .pth checkpoint files.
      - seed (int): random seed for the train/test split.
      - test_size (float): proportion of individuals reserved for evaluation.
    """
    # Directory where saved model .pth checkpoint files live
    models_path = project_root / cfg.paths.models

    batch_size = 64   # number of graphs per DataLoader batch during inference
    test_size = 0.2   # fraction of the dataset held out for evaluation
    seed = 42         # random seed for reproducible train/test split
    return batch_size, models_path, seed, test_size


@app.cell(hide_code=True)
def _():
    """
    Description: Render the section heading for the model loading section.

    Output:
      - (none): renders a Markdown heading in the notebook UI.
    """
    mo.md(r"""
    # Loading the models
    """)
    return


@app.cell
def _():
    """
    Description:
        Imports the dataset loader and model builder functions that are shared across
        several downstream cells.  Separating imports into their own cell makes
        dependencies explicit and keeps larger computation cells focused.

    Output:
      - build_gat (function): factory that constructs a GATSkipRes model.
      - build_mlp (function): factory that constructs a NodeMLP model.
      - load_gva_dataset (function): loads and splits the Geneva PyG dataset.
    """
    from activitygraphs.ml.dataset import load_gva_dataset
    from activitygraphs.run import build_gat, build_mlp

    return build_gat, build_mlp, load_gva_dataset


@app.cell
def _(batch_size, load_dataset, seed, test_size):
    """
    Description:
        Loads the Geneva dataset, splits it into training and test sets, and wraps
        each split in a PyG DataLoader for batch-based inference.

    Input:
      - batch_size (int): number of graphs per DataLoader mini-batch.
      - load_dataset (function): the dataset-loading function imported above.
      - seed (int): random seed for the reproducible train/test split.
      - test_size (float): fraction of individuals reserved for testing.

    Output:
      - test_set (pyg.data.Dataset): the held-out test individuals.
      - train_loader (pyg.loader.DataLoader): batched DataLoader over the training set.
      - train_set (pyg.data.Dataset): the training individuals (needed to build models).
    """
    train_set, test_set = load_dataset(cfg, test_size, seed)
    # DataLoader wraps each dataset so graphs are collated into batches automatically
    train_loader = pyg.loader.DataLoader(train_set, batch_size=batch_size)
    test_loader = pyg.loader.DataLoader(test_set, batch_size=batch_size)
    return test_set, train_loader, train_set


@app.cell
def _(build_gat, build_mlp, models_path, train_set):
    """
    Description: Define the ``load_model`` helper function and use it to instantiate and
    load pretrained weights for the GATSkipRes model (best GNN) and the NodeMLP baseline.

    Input:
      - build_gat (function): factory that builds a GATSkipRes model.
      - build_mlp (function): factory that builds a NodeMLP model.
      - models_path (Path): directory containing the .pth checkpoint files.
      - train_set (pyg.data.Dataset): used to infer input/output dimensions for each model.

    Output:
      - gat (nn.Module): the pretrained GATSkipRes model in eval mode on CUDA.
      - mlp (nn.Module): the pretrained NodeMLP model in eval mode on CUDA.
    """
    hidden_channels = 128  # size of each hidden layer in the GNN
    dropout = 0.2          # dropout rate used during training (kept same for eval)
    gat_layers = 8         # number of GATSkip conv layers in the best-performing model
    gps_layers = 4         # GPS (Graph Positional-Structural) layers (unused here)
    mlp_layers = 3         # number of linear layers in the MLP baseline

    def load_model(file, builder, layers):
        """
        Description:
            Instantiates a model using the given builder function, loads pretrained
            weights from a .pth file, moves the model to GPU and sets it to eval mode.

        Input:
          - file (str): filename of the saved checkpoint (relative to models_path).
          - builder (callable): a function like build_gat or build_mlp that constructs
                the model architecture from (train_set, layers, hidden_channels, dropout).
          - layers (int): number of GNN layers to pass to the builder.

        Output:
          - (nn.Module): the pretrained model in eval mode on CUDA.
        """
        # Resolve the full path to the checkpoint file
        path = models_path / file

        # Build the model architecture (same hyperparameters used during training)
        model = builder(train_set, layers, hidden_channels, dropout)
        # Load the saved parameter values (weights_only=True for security)
        model.load_state_dict(torch.load(path, weights_only=True))
        # Move to GPU and switch to evaluation mode (disables dropout)
        return model.cuda().eval()

    gat = load_model("GATSkip-8-res.pth", build_gat, gat_layers)  # best GNN model
    mlp = load_model("MLP.pth", build_mlp, mlp_layers)             # MLP baseline
    gat, mlp
    return gat, mlp


@app.cell
def _(train_loader, train_set):
    """
    Description:
        Builds and fits the two frequency-based baseline models from the training set.
        These baselines assign logit scores based on per-node visit frequencies
        observed in the training data, with no use of the graph structure.

        NodeBaseline:            predicts visit frequency per node position (ignoring home).
        ConditionalNodeBaseline: predicts visit frequency per node position conditional on
                                 which node is the home.

    Input:
      - train_loader (pyg.loader.DataLoader): batched training data for fitting baselines.
      - train_set (pyg.data.Dataset): used to determine the number of nodes per graph.

    Output:
      - cond_baseline (ConditionalNodeBaseline): fitted conditional frequency baseline.
      - node_baseline (NodeBaseline): fitted unconditional frequency baseline.
    """
    from activitygraphs.ml.baselines import ConditionalNodeBaseline, NodeBaseline

    # num_nodes is fixed across all graphs in the Geneva dataset
    node_baseline = NodeBaseline(train_set[0].num_nodes).fit(train_loader)
    cond_baseline = ConditionalNodeBaseline(train_set[0].num_nodes).fit(train_loader)

    node_baseline, cond_baseline
    return cond_baseline, node_baseline


@app.cell(hide_code=True)
def _():
    """
    Description: Render the section heading for the geographic network data loading section.

    Output:
      - (none): renders a Markdown heading in the notebook UI.
    """
    mo.md(r"""
    # Loading network data
    """)
    return


@app.cell
def _():
    """
    Description:
        Loads the Geneva survey dataset and builds the geographic network GeoDataFrames
        for the map visualisation steps in this notebook.

    Output:
      - gva_data (GenevaData): the full Geneva survey dataset.
      - network_edges (gpd.GeoDataFrame): edge geometries for map overlay.
      - network_nodes (gpd.GeoDataFrame): node geometries for map overlay.
    """
    from activitygraphs.data.geneva import GenevaData
    from activitygraphs.dataprocessing import load_gva_network_graph

    gva_data = GenevaData.load(cfg.data, project_root)
    network_nodes, network_edges = load_gva_network_graph(gva_data, cfg.data, project_root)
    return gva_data, network_edges, network_nodes


@app.cell(hide_code=True)
def _():
    """
    Description: Render the section heading for the model inference / prediction section.

    Output:
      - (none): renders a Markdown heading in the notebook UI.
    """
    mo.md(r"""
    # Making predictions
    """)
    return


@app.cell
def _():
    """
    Description:
        Placeholder / note cell — previously hard-coded a specific user ID.
        No variables are returned; kept for reference.
    """
    "user_id = 25895"
    return


@app.cell
def _(test_set):
    """
    Description:
        Picks a specific individual from the test set by integer index.
        Change `index` to visualise predictions for a different user.

    Input:
      - test_set (pyg.data.Dataset): the held-out test individuals.

    Output:
      - data (pyg.data.Data): PyG graph for the chosen individual.
      - user_id (str): string user ID from the Data object, used to filter network nodes.
    """
    index = 1            # position in the test set; change to inspect another individual
    data = test_set[index]
    user_id = data.user_id  # string identifier used to look up geographic node columns
    return data, user_id


@app.cell
def _():
    """
    Description:
        Imports the two probabilistic sampling functions used to convert per-node
        logit scores into binary location choice sets.

        poisson_sampling: independently samples each node with probability sigmoid(logit).
        pps_sampling:     samples a fixed number of nodes without replacement, with
                          inclusion probability proportional to sigmoid(logit).

    Output:
      - poisson_sampling (function): Poisson (variable-size) sampler.
      - pps_sampling (function): PPS (probability-proportional-to-size, fixed-size) sampler.
    """
    from activitygraphs.ml.sampling import poisson_sampling, pps_sampling

    return poisson_sampling, pps_sampling


@app.cell
def _(
    cond_baseline,
    data,
    gat,
    mlp,
    node_baseline,
    poisson_sampling,
    pps_sampling,
):
    """
    Description: Define the ``predict_and_sample`` helper function and apply it to all
    four models (GATSkipRes, NodeMLP, NodeBaseline, ConditionalNodeBaseline) for the
    selected test individual.  Each call returns logits, probabilities, and two binary
    sample vectors (Poisson and PPS).

    Input:
      - cond_baseline (ConditionalNodeBaseline): conditional frequency baseline.
      - data (pyg.data.Data): the selected test individual's PyG graph.
      - gat (nn.Module): the pretrained GATSkipRes model.
      - mlp (nn.Module): the pretrained NodeMLP model.
      - node_baseline (NodeBaseline): unconditional frequency baseline.
      - poisson_sampling (function): variable-size Poisson sampler.
      - pps_sampling (function): fixed-size PPS sampler.

    Output:
      - cond_poisson, cond_pps, cond_probs: baseline predictions from the conditional model.
      - mlp_poisson, mlp_pps, mlp_probs: predictions from the NodeMLP model.
      - node_poisson, node_pps, node_probs: predictions from the unconditional baseline.
      - poisson, pps, probs: predictions from the GATSkipRes model.
    """
    # Move the single test sample to GPU for inference
    batch = next(iter(pyg.loader.DataLoader([data]))).cuda()

    def predict_and_sample(model, batch, n=15):
        """
        Description:
            Runs a model forward pass and applies two sampling strategies to the
            resulting logits to generate binary location choice sets.

        Input:
          - model (nn.Module): a trained model that accepts (x, edge_index, edge_attr, batch)
                and returns per-node logits.
          - batch (pyg.data.Batch): a batched PyG data object (single graph here).
          - n (int): the desired size of the PPS sample.  Defaults to 15.

        Output:
          - logits (Tensor): raw per-node logit scores, shape (num_nodes, 1).
          - probs (Tensor): sigmoid-transformed visit probabilities, shape (num_nodes, 1).
          - poisson (Tensor): binary Poisson sample (variable size), shape (num_nodes,).
          - pps (Tensor): binary fixed-size PPS sample of n nodes, shape (num_nodes,).
        """
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        probs = torch.sigmoid(logits)  # convert logits to probabilities

        generator = None  # torch.Generator(device="cuda").manual_seed(seed)

        # Poisson sampling: include node i with probability sigmoid(logit_i)
        poisson = poisson_sampling(logits, generator)
        # PPS sampling: sample exactly n nodes with probability proportional to sigmoid(logit_i)
        pps = pps_sampling(n, logits, batch.batch, generator)

        return logits, probs, poisson, pps

    # Run predictions and sampling for all four models
    logits, probs, poisson, pps = predict_and_sample(gat, batch)
    mlp_logits, mlp_probs, mlp_poisson, mlp_pps = predict_and_sample(mlp, batch)
    node_logits, node_probs, node_poisson, node_pps = predict_and_sample(node_baseline, batch)
    cond_logits, cond_probs, cond_poisson, cond_pps = predict_and_sample(cond_baseline, batch)
    return (
        cond_poisson,
        cond_pps,
        cond_probs,
        mlp_poisson,
        mlp_pps,
        mlp_probs,
        node_poisson,
        node_pps,
        node_probs,
        poisson,
        pps,
        probs,
    )


@app.cell(hide_code=True)
def _():
    """
    Description: Render the section heading for the predictions and samples visualisation section.

    Output:
      - (none): renders a Markdown heading in the notebook UI.
    """
    mo.md(r"""
    # Visualising predictions and samples
    """)
    return


@app.cell
def _():
    """
    Description:
        Imports the add_user_cols helper function for use in the visualisation cell below.
        Separating this import keeps the long prediction+visualisation cell cleaner.

    Output:
      - add_user_cols (function): enriches network_nodes with per-user indicator columns.
    """
    from activitygraphs.dataprocessing import add_user_cols

    return (add_user_cols,)


@app.cell
def _(
    add_user_cols,
    cond_poisson,
    cond_pps,
    cond_probs,
    gva_data,
    mlp_poisson,
    mlp_pps,
    mlp_probs,
    network_nodes,
    node_poisson,
    node_pps,
    node_probs,
    poisson,
    pps,
    probs,
    user_id,
):
    """
    Description:
        Marimo cell that assembles a combined GeoDataFrame of network nodes annotated
        with prediction probabilities and sampled locations from every model (GNN,
        MLP, node-baseline, conditional-node-baseline). Each model's outputs are
        attached as separate columns using a short prefix so they can be compared
        side-by-side on a map.

    Input:
      - add_user_cols: function that attaches user-specific columns to a GeoDataFrame.
      - cond_poisson, cond_pps, cond_probs: conditional-node-baseline Poisson count,
            PPS-sampled nodes, and raw probability array for the current user.
      - gva_data: the loaded Geneva dataset providing trip and location tables.
      - mlp_poisson, mlp_pps, mlp_probs: MLP model outputs (same structure as above).
      - network_nodes (GeoDataFrame): the spatial nodes of the transport network.
      - node_poisson, node_pps, node_probs: node-baseline model outputs.
      - poisson, pps, probs: GNN model outputs for the current user.
      - user_id (int): the index of the individual being visualised.

    Output:
      - nodes_with_preds (GeoDataFrame): network nodes with columns for each model's
            probability scores and sampled binary visit indicators.
    """
    def add_preds_and_sample(nodes, preds, poisson, pps, prefix=""):
        """
        Description:
            Attaches per-node prediction probabilities and sampling results from one model
            to the network nodes GeoDataFrame, using an optional column name prefix to
            distinguish results from different models.

        Input:
          - nodes (gpd.GeoDataFrame): the network nodes GeoDataFrame to annotate.
          - preds (torch.Tensor): predicted visit probabilities, shape (num_nodes, 1).
          - poisson (torch.Tensor): Poisson sample binary indicators, shape (num_nodes,).
          - pps (torch.Tensor): PPS sample binary indicators, shape (num_nodes,).
          - prefix (str): column name prefix, e.g. "mlp_" to name columns "mlp_preds",
                "mlp_poisson", "mlp_pps".  Defaults to "" (no prefix, for the GAT model).

        Output:
          - (gpd.GeoDataFrame): a copy of nodes with three new columns appended.
        """
        # Ensure rows are aligned with the tensor ordering (PyG indexes by sorted index)
        nodes = nodes.copy().sort_index()

        # Convert tensors to NumPy and assign as new columns
        nodes[prefix + "preds"] = preds.detach().cpu().numpy()      # visit probabilities
        nodes[prefix + "poisson"] = poisson.detach().cpu().numpy()  # Poisson sample
        nodes[prefix + "pps"] = pps.detach().cpu().numpy()          # PPS sample

        return nodes

    user_nodes = add_user_cols(
        user_id,
        network_nodes,
        gva_data.location_visits,
        gva_data.home_locations,
        gva_data.work_locations,
        gva_data.edu_locations,
    )
    user_nodes = add_preds_and_sample(user_nodes, probs, poisson, pps)
    user_nodes = add_preds_and_sample(user_nodes, mlp_probs, mlp_poisson, mlp_pps, prefix="mlp_")
    user_nodes = add_preds_and_sample(user_nodes, node_probs, node_poisson, node_pps, prefix="node_")
    user_nodes = add_preds_and_sample(user_nodes, cond_probs, cond_poisson, cond_pps, prefix="cond_")

    user_nodes
    return (user_nodes,)


@app.cell
def _(user_nodes):
    """
    Description:
        Creates the interactive widgets for the sampling visualisation map:
        a column selector (pre-set to "preds") and a polygon/point view toggle
        (defaulting to showing polygon subsector geometry).

    Input:
      - user_nodes (gpd.GeoDataFrame): node table with all four model's prediction columns.

    Output:
      - select_node_col (mo.ui.dropdown): column selector for the map colouring.
      - toggle_polygons (mo.ui.switch): True = show subsector polygons, False = show points.
    """
    select_node_col = mo.ui.dropdown(list(user_nodes.columns), searchable=True, label="Column:", value="preds")
    toggle_polygons = mo.ui.switch(value=True, label="Show subsectors")
    return select_node_col, toggle_polygons


@app.cell
def _(network_edges, select_node_col, toggle_polygons, user_nodes):
    """
    Description:
        Renders the interactive Folium map of user_nodes, coloured by the column
        chosen in select_node_col.  An orange-red (OrRd) colour map is used to make
        high-probability / sampled nodes visually prominent.

    Input:
      - network_edges (gpd.GeoDataFrame): edge geometries drawn in grey.
      - select_node_col (mo.ui.dropdown): the column to visualise (e.g. "preds", "pps").
      - toggle_polygons (mo.ui.switch): whether to render polygon or point geometry.
      - user_nodes (gpd.GeoDataFrame): enriched node table with prediction/sample columns.

    Output:
      - (displayed in notebook, nothing returned to other cells)
    """
    # Switch between polygon (subsector) and point (centroid) geometry
    _nodes = user_nodes.set_geometry("original_geometry") if toggle_polygons.value else user_nodes
    _m = network_edges.explore(color="gray", tiles="Cartodb Positron")
    _m = _nodes.explore(m=_m, column=select_node_col.value, marker_kwds={"radius": 5}, cmap="OrRd")

    mo.vstack([
        mo.hstack([select_node_col, toggle_polygons], justify="start"),
        _m,
    ])
    return


@app.cell
def _(network_nodes):
    """
    Description:
        Defines the coordinate reference system (CRS) constants used throughout the
        plotting cells.  CRS is the standard WGS-84 geographic CRS that contextily
        expects.  utm_crs is estimated from the network extent for metric-distance
        operations (area calculations, etc.) but is not exported to other cells.

    Input:
      - network_nodes (gpd.GeoDataFrame): used to auto-detect the best UTM zone.

    Output:
      - CRS (str): "EPSG:4326" — the WGS-84 lat/lon CRS string.
    """
    CRS = "EPSG:4326"   # WGS-84 lat/lon coordinate reference system for all map plots
    # utm_crs: the UTM zone best suited for the network's extent, used for area calculations
    utm_crs = network_nodes.estimate_utm_crs()
    return (CRS,)


@app.cell
def _():
    """
    Description:
        Imports contextily (for basemap tiles) and matplotlib.pyplot (for figure and
        axes management).  These are needed for the static Matplotlib map cells below.

    Output:
      - cx (module): contextily, used to add CartoDB Positron basemaps.
      - plt (module): matplotlib.pyplot, used to create subplots and save figures.
    """
    import contextily as cx
    import matplotlib.pyplot as plt

    return cx, plt


@app.cell
def _(user_id):
    """
    Description:
        Display cell — prints the currently selected user ID string so the notebook
        user can see which individual's predictions are being visualised below.

    Input:
      - user_id (str): the user ID string from the selected test-set Data object.

    Output:
      - (displayed in notebook, nothing returned)
    """
    user_id
    return


@app.cell
def _(CRS, cx):
    """
    Description:
        Imports PowerNorm and defines the plot_preds helper function used to render
        geographic choropleth and dot-plot maps of per-node predictions.

    Input:
      - CRS (str): WGS-84 CRS string for reprojection.
      - cx (module): contextily for basemap tile overlay.

    Output:
      - plot_preds (function): the mapping helper, exported for use in figure cells below.
    """
    from matplotlib.colors import PowerNorm

    def plot_preds(nodes, col, title, ax=None, as_points=False, edges=None):
        """
        Description:
            Renders a geographic map of a continuous per-node quantity (e.g. predicted
            visit probability) using either polygon choropleth (subsectors) or dot plots
            (point geometries) depending on the as_points flag.

        Input:
          - nodes (gpd.GeoDataFrame): GeoDataFrame of network nodes with prediction columns.
          - col (str): the column to visualise (e.g. "preds", "mlp_preds").
          - title (str): title text for the map subplot.
          - ax (matplotlib.axes.Axes | None): existing axes to draw on; creates a new
                7x7 inch figure if None.  Defaults to None.
          - as_points (bool): if True, use point geometry; if False, use the polygon
                "original_geometry".  Defaults to False.
          - edges (gpd.GeoDataFrame | None): optional edge GeoDataFrame drawn as grey
                lines underneath the nodes.  Defaults to None.

        Output:
          - ax (matplotlib.axes.Axes): the axes with the rendered map.
        """
        # Choose polygon geometry (subsectors) or centroid points
        nodes = nodes if as_points else nodes.set_geometry("original_geometry")
        nodes = nodes.to_crs(CRS)  # ensure WGS-84 for contextily compatibility

        # figsize is only provided when creating a fresh figure (ax is None)
        figsize = (7, 7) if ax is None else None

        if not as_points:
            # Draw polygon outlines first so they appear below the fill colour
            ax = nodes.boundary.plot(ax=ax, color="gray", linewidth=0.3)

        if edges is not None:
            # Draw network edges as thin grey lines
            ax = edges.plot(ax=ax, figsize=figsize, color="gray", legend=True, linewidth=0.3)

        # Choose colour map and power-norm gamma based on point vs polygon mode
        cmap = "Reds" if not as_points else "viridis_r"
        gamma = 0.7 if not as_points else 0.3  # gamma < 1 stretches low values
        label = "Predicted probability of visit $\\hat{y}_{i,n}$" if not as_points else "Per-node visit frequency"

        ax = nodes.plot(
            ax=ax,
            column=col,
            figsize=figsize,
            legend=True,
            cmap=cmap,
            # PowerNorm stretches the colour scale to make low probabilities more visible
            norm=PowerNorm(gamma=gamma, vmin=0, vmax=1),
            legend_kwds={
                "shrink": 0.7,
                "label": label,
                "location": "bottom",
                "pad": 0.01,
            },
            markersize=25,
        )

        ax.set_title(title)
        ax.set_xticks([])  # remove axis tick marks for a cleaner map look
        ax.set_yticks([])

        # Overlay a CartoDB Positron basemap (no labels)
        cx.add_basemap(ax, crs=CRS, source=cx.providers.CartoDB.PositronNoLabels)

        return ax

    return (plot_preds,)


@app.cell
def _():
    """
    Description:
        Defines the figure size (width x height in inches) for two-panel (side-by-side)
        Matplotlib figures used in the comparison plots below.

    Output:
      - double_figsize (tuple[int, int]): (13, 7) — wide enough for two map panels.
    """
    double_figsize = (13, 7)  # width x height in inches for 2-panel figures
    return (double_figsize,)


@app.cell
def _(
    double_figsize,
    network_edges,
    plot_preds,
    plot_visited,
    plt,
    user_nodes,
):
    """
    Description:
        Creates a two-panel figure saved as "gva_struct.png":
          Left panel:  per-node visit frequency (as points) overlaid on the network edges,
                       showing the graph structure with visit frequencies.
          Right panel: the ground-truth visited nodes for the example individual.

    Input:
      - double_figsize (tuple): figure size in inches.
      - network_edges (gpd.GeoDataFrame): edge geometries for the left-panel overlay.
      - plot_preds (function): renders a continuous column as a choropleth.
      - plot_visited (function): renders a binary column as a categorical map.
      - plt (module): matplotlib.pyplot.
      - user_nodes (gpd.GeoDataFrame): node table with prediction and label columns.

    Output:
      - (figure saved to disk; nothing exported to other cells)
    """
    _fig, _axs = plt.subplots(1, 2, figsize=double_figsize, constrained_layout=True)

    plot_preds(
        user_nodes,
        "node_preds",
        "Graph structure of the TPG dataset, with per-node visit frequencies",
        ax=_axs[0],
        as_points=True,
        edges=network_edges,
    )

    plot_visited(user_nodes, "is_visited", "Visited nodes for example individual", ax=_axs[1])

    _fig.savefig(project_root / cfg.paths.figures / "gva_struct.png")
    _fig
    return


@app.cell
def _(double_figsize, plot_preds, plt, user_nodes):
    """
    Description:
        Creates a two-panel figure saved as "gva_preds.png" comparing the predicted
        visit probability maps from the MLP baseline (left) and GATSkipRes model (right)
        for the selected example individual.

    Input:
      - double_figsize (tuple): figure size in inches for two side-by-side panels.
      - plot_preds (function): renders a continuous prediction column as a choropleth.
      - plt (module): matplotlib.pyplot.
      - user_nodes (gpd.GeoDataFrame): node table with "mlp_preds" and "preds" columns.

    Output:
      - (figure saved to disk; nothing exported to other cells)
    """
    _fig, _axs = plt.subplots(1, 2, figsize=double_figsize, constrained_layout=True)

    plot_preds(user_nodes, "mlp_preds", "MLP predicted visit probabilities for example individual", ax=_axs[0])

    plot_preds(user_nodes, "preds", "GATSkipRes predicted visit probabilities for example individual", ax=_axs[1])

    _fig.savefig(project_root / cfg.paths.figures / "gva_preds.png")
    _fig
    return


@app.cell
def _():
    """
    Description:
        Imports the Seaborn visualisation library, used here for its colour palettes
        (e.g. "tab10" and "vlag_r") rather than for high-level chart types.

    Output:
      - sns (module): the seaborn module, available to downstream palette cells.
    """
    import seaborn as sns

    return (sns,)


@app.cell
def _(CRS, cx, sns):
    """
    Description:
        Imports ListedColormap and defines the plot_visited helper function that
        renders categorical (binary) visit label maps on the Geneva network.

    Input:
      - CRS (str): WGS-84 CRS string for reprojection.
      - cx (module): contextily for basemap tile overlay.
      - sns (module): seaborn, used to get the tab10 colour palette.

    Output:
      - plot_visited (function): the categorical map helper, exported for figure cells.
    """
    from matplotlib.colors import ListedColormap

    def plot_visited(
        nodes, col, title, ax=None, legend_name="Visited", as_points=False, edges=None, include_unvisited=False
    ):
        """
        Description:
            Renders a categorical map of binary node labels (visited / home / optionally
            unvisited) using a discrete colour scheme overlaid on a CartoDB basemap.
            Used to visualise ground-truth visit labels or sampled choice sets.

        Input:
          - nodes (gpd.GeoDataFrame): node GeoDataFrame with the binary column to display.
          - col (str): name of the binary (0/1) column to visualise (e.g. "is_visited",
                "poisson", "pps").
          - title (str): title text shown above the map.
          - ax (matplotlib.axes.Axes | None): existing axes to draw on; if None a new
                7×7 inch figure is created.  Defaults to None.
          - legend_name (str): label for the "visited" category in the legend.
                Defaults to "Visited".
          - as_points (bool): if True use centroid point geometry; if False use polygon
                "original_geometry".  Defaults to False.
          - edges (gpd.GeoDataFrame | None): optional edge GeoDataFrame to overlay in
                grey.  Defaults to None.
          - include_unvisited (bool): if True, explicitly draw unvisited nodes in light
                grey.  Defaults to False.

        Output:
          - ax (matplotlib.axes.Axes): the axes with the rendered map.
        """
        nodes = nodes if as_points else nodes.set_geometry("original_geometry")
        nodes = nodes.to_crs(CRS).copy()

        if include_unvisited:
            nodes.loc[nodes[col] == 0, "visit"] = "Unvisited"

        nodes.loc[nodes[col] == 1, "visit"] = legend_name
        nodes.loc[nodes["is_home"] == 1, "visit"] = "Home"

        figsize = (7, 7) if ax is None else None

        if not as_points:
            ax = nodes.boundary.plot(ax=ax, color="gray", linewidth=0.3)

        if edges is not None:
            ax = edges.to_crs(CRS).plot(ax=ax, figsize=figsize, color="gray", linewidth=0.3)

        tab10 = sns.color_palette()
        categories = [legend_name, "Home"] + (["Unvisited"] if include_unvisited else [])
        colors = [tab10[0], tab10[1]] + (["lightgray"] if include_unvisited else [])
        cmap = ListedColormap(colors)

        ax = nodes.plot(
            ax=ax,
            column="visit",
            figsize=figsize,
            legend=True,
            markersize=20,
            categories=categories,
            cmap=cmap,
        )

        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

        cx.add_basemap(ax, crs=CRS, source=cx.providers.CartoDB.PositronNoLabels)

        return ax

    return (plot_visited,)


@app.cell
def _(double_figsize, plot_visited, plt, user_nodes):
    """
    Description:
        Creates a two-panel figure saved as "gva_samples.png" comparing the two
        sampling strategies applied to the GATSkipRes predictions:
          Left panel:  the Poisson sample (variable size).
          Right panel: the 15-node PPS sample (fixed size).

    Input:
      - double_figsize (tuple): figure size for two-panel layouts.
      - plot_visited (function): renders a binary sample column as a categorical map.
      - plt (module): matplotlib.pyplot.
      - user_nodes (gpd.GeoDataFrame): node table with "poisson" and "pps" columns.

    Output:
      - (figure saved to disk; nothing exported to other cells)
    """
    _fig, _axs = plt.subplots(1, 2, figsize=double_figsize, constrained_layout=True)

    plot_visited(user_nodes, "poisson", "Poisson sample of GATSkipRes predictions", legend_name="Sampled", ax=_axs[0])
    plot_visited(user_nodes, "pps", "15-node PPS sample of GATSkipRes predictions", legend_name="Sampled", ax=_axs[1])

    _fig.savefig(project_root / cfg.paths.figures / "gva_samples.png")
    _fig
    return


@app.cell
def _():
    """
    Description:
        Empty placeholder cell — reserved for future additions to the notebook.
        Returns nothing and performs no computation.
    """
    return


if __name__ == "__main__":
    app.run()
