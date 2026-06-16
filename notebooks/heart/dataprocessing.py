"""
Module: notebooks/heart/dataprocessing.py

Description:
    A Marimo interactive notebook for loading and visualising the Geneva PyG dataset
    alongside the geographic network graph.  It combines:
      - Loading GenevaData (raw survey trips and user metadata).
      - Building or loading the flat network node/edge GeoDataFrames used for visualisation.
      - Optionally generating the PyG graph dataset (one graph per user).
      - Visualising per-user node attributes on an interactive map (e.g. home/work/edu
        indicators, visited-location flags, GNN predictions).

    This notebook serves as an interactive data exploration and prediction visualisation
    tool once a model has been trained.

Dependencies:
    - activitygraphs library (GenevaData, load_gva_network_graph, load_pyg_graphs,
      add_user_cols, build_gat, build_mlp)
    - PyTorch, PyTorch Geometric
    - Marimo, GeoPandas, Contextily
"""

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo

    import pyarrow  # noqa: F401
    import pandas  # noqa: F401
    import torch
    import torch_geometric as pyg

    from activitygraphs.config import load_config
    from pathlib import Path

    project_root = Path(mo.notebook_dir().parent.parent)
    cfg = load_config(project_root)


@app.cell
def _():
    """
    Description:
        Loads the Geneva survey dataset from disk using the project configuration.
        GenevaData bundles the raw survey tables (trips, home/work/edu locations,
        user metadata) into a single object for downstream processing.

    Input:
      - (no parameters; reads cfg.data and project_root from the notebook setup block)

    Output:
      - gva_data (GenevaData): the loaded Geneva dataset object.
    """
    from activitygraphs.data.geneva import GenevaData

    gva_data = GenevaData.load(cfg.data, project_root)
    return (gva_data,)


@app.cell
def _(network_nodes):
    """
    Description:
        Display cell — renders the network_nodes GeoDataFrame in the notebook output
        so the user can inspect the node table (columns, CRS, geometry type, etc.).

    Input:
      - network_nodes (gpd.GeoDataFrame): GeoDataFrame of network node locations.

    Output:
      - (displayed in notebook, nothing returned to other cells)
    """
    network_nodes
    return


@app.cell
def _(gva_data):
    """
    Description:
        Loads (or builds) the flat geographic network graph for Geneva as two
        GeoDataFrames: one for nodes (subsectors / municipalities) and one for edges
        (roads / connections between zones).  Results are cached on disk and reused
        on subsequent runs.

    Input:
      - gva_data (GenevaData): the loaded Geneva dataset, used to determine node IDs.
      - cfg.data (DataConfig): data configuration with paths to raw/processed shapefiles.
      - project_root (Path): absolute path to the repository root directory.

    Output:
      - network_nodes (gpd.GeoDataFrame): one row per node with geometry and node attributes.
      - network_edges (gpd.GeoDataFrame): one row per edge as a LineString geometry.
    """
    from activitygraphs.dataprocessing import load_gva_network_graph

    network_nodes, network_edges = load_gva_network_graph(gva_data, cfg.data, project_root)
    return network_edges, network_nodes


@app.cell(hide_code=True)
def _():
    """
    Description:
        Renders a user-triggered "Load PyG Graphs" button in the notebook.  The button
        is styled as a warning (orange) because building PyG graphs is slow and only needs
        to be done once.  The downstream cell is gated on this button's value.

    Output:
      - run_graph_gen (mo.ui.run_button): the button widget, displayed in the notebook.
    """
    run_graph_gen = mo.ui.run_button(kind="warn", label="Load PyG Graphs")
    run_graph_gen
    return (run_graph_gen,)


@app.cell
def _(gva_data, network_edges, network_nodes, run_graph_gen):
    """
    Description:
        Triggered only when the user presses the "Load PyG Graphs" button.
        Converts each user's travel data into a separate PyTorch Geometric Data object,
        combining the network graph structure with per-user node features and labels.
        Results are cached as .pt files so subsequent runs skip this step.

    Input:
      - gva_data (GenevaData): the loaded Geneva dataset.
      - network_nodes (gpd.GeoDataFrame): geographic node GeoDataFrame.
      - network_edges (gpd.GeoDataFrame): geographic edge GeoDataFrame.
      - run_graph_gen (mo.ui.run_button): gates execution; cell stops immediately
            if the button has not been pressed.

    Output:
      - graphs (list[pyg.data.Data]): one PyG Data object per user, displayed in notebook.
    """
    from activitygraphs.dataprocessing import load_pyg_graphs

    mo.stop(not run_graph_gen.value)

    graphs = load_pyg_graphs(gva_data, network_nodes, network_edges, cfg.data, project_root)
    graphs
    return


@app.cell(hide_code=True)
def _():
    """
    Description: Render the section heading for network and individual graph visualisation.

    Output:
      - (none): renders a Markdown heading in the notebook UI.
    """
    mo.md(r"""
    # Visualising network and individual graphs
    """)
    return


@app.cell(hide_code=True)
def _(gva_data):
    """
    Description:
        Builds the list of valid user IDs (those who have a subsector-level home location)
        and renders a searchable dropdown widget so the user can pick a specific individual
        to visualise.

    Input:
      - gva_data (GenevaData): the loaded Geneva dataset.

    Output:
      - add_user_cols (function): imported helper that enriches network_nodes with per-user
            visit and home/work/edu indicator columns.
      - select_user_id (mo.ui.dropdown): the user-ID selector widget, rendered in notebook.
    """
    from activitygraphs.dataprocessing import add_user_cols

    # Only include users whose home location is mapped to a known network subsector
    user_ids = gva_data.with_filter("subsector").user_ids
    # Dropdown widget so the notebook user can select which individual to inspect
    select_user_id = mo.ui.dropdown(user_ids, value=user_ids[0], label="User ID:", searchable=True)
    return add_user_cols, select_user_id


@app.cell
def _(add_user_cols, gva_data, network_nodes, select_user_id):
    """
    Description:
        Adds individual-specific columns (is_visited, is_home, is_work, is_edu) to the
        network nodes GeoDataFrame for the currently selected user.  The resulting table
        is displayed so the user can see which nodes this individual has visited and where
        their home/work/edu locations are.

    Input:
      - add_user_cols (function): helper function that enriches network_nodes.
      - gva_data (GenevaData): survey data with per-user location tables.
      - network_nodes (gpd.GeoDataFrame): the base network node GeoDataFrame.
      - select_user_id (mo.ui.dropdown): the currently selected user ID.

    Output:
      - indiv_nodes (gpd.GeoDataFrame): network_nodes enriched with per-user columns.
    """
    indiv_nodes = add_user_cols(select_user_id.value, network_nodes, gva_data.location_visits, gva_data.home_locations,
                                gva_data.work_locations, gva_data.edu_locations)
    indiv_nodes
    return (indiv_nodes,)


@app.cell(hide_code=True)
def _(indiv_nodes):
    """
    Description:
        Renders a searchable dropdown that lets the notebook user choose which column
        of indiv_nodes to colour on the map (e.g. "purpose", "is_visited", "is_home").

    Input:
      - indiv_nodes (gpd.GeoDataFrame): network nodes enriched with per-user columns.

    Output:
      - select_node_col (mo.ui.dropdown): the column selector widget.
    """
    select_node_col = mo.ui.dropdown(list(indiv_nodes.columns), searchable=True, label="Column:", value="purpose")
    return (select_node_col,)


@app.cell(hide_code=True)
def _():
    """
    Description:
        Renders a toggle switch that controls whether nodes are displayed as polygon
        (subsector) geometries or as point centroids on the interactive map.

    Output:
      - toggle_polygons (mo.ui.switch): True = show subsector polygons; False = show points.
    """
    toggle_polygons = mo.ui.switch(value=False, label="Show subsectors")
    return (toggle_polygons,)


@app.cell(hide_code=True)
def _(
    indiv_nodes,
    network_edges,
    select_node_col,
    select_user_id,
    toggle_polygons,
):
    """
    Description:
        Renders the interactive Folium map showing the Geneva network overlaid with
        the selected user's per-node attribute column.  Network edges are drawn in grey
        and nodes are coloured by the column chosen in select_node_col.  A row of widgets
        (user ID selector, column selector, polygon toggle) sits above the map.

    Input:
      - indiv_nodes (gpd.GeoDataFrame): network nodes with per-user columns.
      - network_edges (gpd.GeoDataFrame): geographic edge GeoDataFrame (drawn grey).
      - select_node_col (mo.ui.dropdown): which column to colour nodes by.
      - select_user_id (mo.ui.dropdown): currently selected user ID (displayed as label).
      - toggle_polygons (mo.ui.switch): if True, use polygon geometry for nodes.

    Output:
      - (displayed in notebook, nothing returned to other cells)
    """
    # Switch between polygon (subsector) geometry and default point geometry
    _nodes = indiv_nodes.set_geometry("original_geometry") if toggle_polygons.value else indiv_nodes
    _m = network_edges.explore(color="gray", tiles="Cartodb Positron")
    _m = _nodes.explore(m=_m, column=select_node_col.value, marker_kwds={"radius": 5})

    mo.vstack([
        mo.hstack([select_user_id, select_node_col, toggle_polygons], justify="start"),
        _m,
    ])
    return


@app.cell(hide_code=True)
def _():
    """
    Description: Render the section heading for the sampling and prediction visualisation section.

    Output:
      - (none): renders a Markdown heading in the notebook UI.
    """
    mo.md(r"""
    # Sampling and visualising predictions
    """)
    return


@app.cell
def _():
    """
    Description:
        Loads the full Geneva PyG dataset and splits it into train and test subsets.
        The split is deterministic: the same seed always produces the same partition,
        allowing reproducible model evaluation.

    Output:
      - test_dataset (pyg.data.Dataset): test portion (20% of individuals).
      - train_dataset (pyg.data.Dataset): training portion (80% of individuals).
    """
    from activitygraphs.ml.dataset import load_gva_dataset

    test_size = 0.2   # fraction of individuals reserved for evaluation
    seed = 42         # random seed for the train/test split — ensures reproducibility

    train_dataset, test_dataset = load_gva_dataset(cfg, test_size, seed, project_root)
    return test_dataset, train_dataset


@app.cell
def _(train_dataset):
    """
    Description:
        Instantiates the GATSkipRes and NodeMLP models with the same hyperparameters
        used during training, then loads their saved weights from disk.  Both models
        are kept on CPU at this stage; the forward pass cell moves data as needed.

    Input:
      - train_dataset (pyg.data.Dataset): used to infer in_channels and num_classes
            so the model architecture matches the data exactly.

    Output:
      - gat (nn.Module): the pretrained GATSkipRes model in eval mode.
      - (mlp is also built but not exported — the cell returns only gat)
    """
    from activitygraphs.run import build_gat, build_mlp

    # Build a GATSkip model with 8 GAT layers, 128 hidden channels, 0.2 dropout
    # then load the pretrained weights from a saved .pth checkpoint.
    gat = build_gat(train_dataset, 8, 128, 0.2)
    gat.load_state_dict(torch.load(project_root / "models" / "GATSkip-8-res.pth", weights_only=True))

    # Build a NodeMLP with 3 layers, 128 hidden channels, 0.2 dropout
    # then load its pretrained weights.
    mlp = build_mlp(train_dataset, 3, 128, 0.2)
    mlp.load_state_dict(torch.load(project_root / "models" / "MLP.pth", weights_only=True))

    gat, mlp
    return (gat,)


@app.cell
def _(test_dataset):
    """
    Description:
        Selects a single individual from the test dataset to use for the prediction
        visualisation below.  Change `index` to inspect a different test-set user.

    Input:
      - test_dataset (pyg.data.Dataset): the held-out test set.

    Output:
      - data (pyg.data.Data): the PyG graph for the chosen individual, displayed in notebook.
    """
    index = 1                       # index into the test set; change to view different users
    data = test_dataset[index]      # PyG Data object for the selected test user
    user_id = data.user_id          # string user ID, used to filter the network node GDF
    data
    return (data,)


@app.cell
def _(data, gat):
    """
    Description:
        Runs the pretrained GAT model on the selected individual's graph to produce
        per-node visit probability predictions.  The Data object is wrapped in a
        DataLoader to get the batch metadata (edge_attr, batch vector) the model needs.

    Input:
      - data (pyg.data.Data): graph for the selected test individual.
      - gat (nn.Module): the pretrained GATSkipRes model.

    Output:
      - preds (np.ndarray): visit probability for each node, shape (num_nodes, 1).
    """
    # Wrap the single Data object in a DataLoader to obtain a proper batch object
    _batch = next(iter(pyg.loader.DataLoader([data])))
    # Forward pass through the GAT model; outputs raw logits, shape (num_nodes, 1)
    # Sigmoid converts logits to visit probabilities in [0, 1]
    preds = torch.sigmoid(gat(_batch.x, _batch.edge_index, _batch.edge_attr, _batch.batch)).detach().cpu().numpy()
    preds.T  # transpose for display: shape (1, num_nodes)
    return (preds,)


@app.cell
def _(network_nodes):
    """
    Description:
        Display cell — shows the network node table sorted by its index.
        Sorting is needed to verify that node index order matches the tensor ordering
        used by PyG (which also sorts by node index).

    Input:
      - network_nodes (gpd.GeoDataFrame): the base network node GeoDataFrame.

    Output:
      - (displayed in notebook, nothing returned to other cells)
    """
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
    """
    Description:
        Attaches the GAT model's per-node visit predictions to the user's node
        GeoDataFrame and renders them on an interactive Folium map.  The "predictions"
        column (GAT visit probabilities) determines node colour.  The polygon/point
        toggle and user-ID/column selector widgets are shown above the map.

    Input:
      - network_edges (gpd.GeoDataFrame): geographic edge GeoDataFrame (drawn grey).
      - preds (np.ndarray): per-node visit probabilities from the GAT model, shape (N, 1).
      - select_node_col (mo.ui.dropdown): column selector widget (displayed as label).
      - select_user_id (mo.ui.dropdown): user ID selector widget (displayed as label).
      - toggle_polygons (mo.ui.switch): whether to render polygon or point geometry.
      - user_nodes (gpd.GeoDataFrame): network nodes with per-user attribute columns.

    Output:
      - nodes (gpd.GeoDataFrame): user_nodes with the "predictions" column added.
    """
    _pred_nodes = user_nodes.copy().sort_index()
    _pred_nodes["predictions"] = preds  # attach GAT predictions as a new column

    # Choose polygon (subsector) or point geometry based on the toggle switch
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
    """
    Description:
        Imports the contextily library, which is used to add web-tile basemaps
        (e.g. CartoDB Positron) underneath GeoPandas/Matplotlib map plots.

    Output:
      - cx (module): the contextily module, made available to downstream cells.
    """
    import contextily as cx

    return (cx,)


@app.cell
def _(CRS, cx, nodes):
    """
    Description:
        Defines plot_preds, a helper function that draws a static choropleth map
        of any continuous per-node column over the Geneva subsector polygons, and
        immediately calls it to visualise the GAT predictions for the selected user.

    Input:
      - CRS (str): WGS-84 CRS string for reprojection.
      - cx (module): contextily, used to add the CartoDB Positron basemap.
      - nodes (gpd.GeoDataFrame): network nodes with a "predictions" column.

    Output:
      - plot_preds (function): the helper function, exported for later reuse.
      - (matplotlib figure displayed in notebook from the call at the bottom)
    """
    def plot_preds(nodes, col="predictions"):
        """
        Description:
            Draws a choropleth map of per-node predictions (or any continuous column)
            over the subsector polygons of the Geneva network, overlaid on a CartoDB
            basemap.

        Input:
          - nodes (gpd.GeoDataFrame): GeoDataFrame of network nodes.  Should have a
                geometry column (polygons for subsectors) and the column specified by col.
          - col (str): the name of the column to visualise.  Defaults to "predictions".

        Output:
          - ax (matplotlib.axes.Axes): the matplotlib axes with the plotted map.
        """
        # Reproject to the standard WGS-84 CRS expected by contextily
        nodes = nodes.to_crs(CRS)

        # Plot the column as a filled choropleth; OrRd colour map (orange-to-red)
        ax = nodes.plot(column=col, figsize=(15, 15), legend=True, cmap="OrRd")
        # Overlay the polygon outlines in light grey for reference
        nodes.boundary.plot(ax=ax, color="lightgrey")

        # Add a CartoDB Positron basemap (no labels to keep the map clean)
        cx.add_basemap(ax, crs=CRS, source=cx.providers.CartoDB.PositronNoLabels)

        return ax

    plot_preds(nodes)
    return (plot_preds,)


@app.cell
def _(plot_preds):
    """
    Description:
        Display cell — calls plot_preds() with default arguments to render the
        choropleth map of GAT predictions for the currently selected user.

    Input:
      - plot_preds (function): the plot_preds helper from the cell above.

    Output:
      - (matplotlib.axes.Axes, displayed in notebook)
    """
    plot_preds()
    return


if __name__ == "__main__":
    app.run()
