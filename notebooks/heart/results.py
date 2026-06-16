"""
Module: notebooks/heart/results.py

Description:
    A Marimo interactive notebook for loading and visualising training results for the
    Geneva dataset.

    The notebook:
      1. Loads the training and test datasets and the pretrained GATSkipRes / MLP models.
      2. Reads saved experiment results (Parquet) and plots loss curves.
      3. Loads the geographic network GeoDataFrame and a selected test-set user.
      4. Runs model inference and displays predicted visit probabilities on a map.

    This notebook is used to produce publication-quality figures for the thesis/report.

Dependencies:
    - activitygraphs library (GenevaData, load_gva_dataset, load_gva_network_graph,
      add_user_cols, build_gat, build_mlp)
    - PyTorch, PyTorch Geometric
    - Marimo, Polars, Altair, GeoPandas, Matplotlib, Contextily
"""

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")

with app.setup:
    from pathlib import Path

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
        Loads the processed Geneva PyG dataset and splits it into training and test
        portions using a fixed random seed for reproducibility.

    Output:
      - test_dataset (pyg.data.Dataset): 20% of individuals, used for evaluation.
      - train_dataset (pyg.data.Dataset): 80% of individuals, used for model loading.
    """
    from activitygraphs.ml.dataset import load_gva_dataset

    test_size = 0.2   # fraction of the dataset reserved for testing
    seed = 42         # fixed seed so the split is always the same across runs

    train_dataset, test_dataset = load_gva_dataset(cfg, test_size, seed)
    return test_dataset, train_dataset


@app.cell
def _(train_dataset):
    """
    Description:
        Builds the GATSkipRes and NodeMLP model architectures with the same
        hyperparameters used during training, then loads their saved weights.
        Both models are built so the cell can confirm both load correctly, but only
        gat is exported for the prediction visualisation downstream.

    Input:
      - train_dataset (pyg.data.Dataset): used to infer num_features and num_classes.

    Output:
      - gat (nn.Module): the pretrained 8-layer GATSkipRes model in eval mode.
      - (mlp is also loaded but not exported — not used further in this notebook)
    """
    from activitygraphs.run import build_gat, build_mlp

    # 8 GAT layers, 128 hidden channels, 0.2 dropout — matches the training config
    gat = build_gat(train_dataset, 8, 128, 0.2)
    gat.load_state_dict(torch.load(project_root / "models" / "GATSkip-8-res.pth", weights_only=True))

    # 3 linear layers, 128 hidden channels, 0.2 dropout — matches the training config
    mlp = build_mlp(train_dataset, 3, 128, 0.2)
    mlp.load_state_dict(torch.load(project_root / "models" / "MLP.pth", weights_only=True))

    gat, mlp
    return (gat,)


@app.cell
def _():
    """
    Description:
        Resolves and exports the paths to the experiment results directory and the
        figures output directory, both derived from the project configuration.

    Output:
      - figures_path (Path): absolute path where PNG figures will be saved.
      - reports_path (Path): absolute path to the directory with Parquet result files.
    """
    # Directory containing saved experiment result Parquet files
    reports_path = project_root / cfg.paths.reports / "data"
    # Directory where generated figures are saved for the thesis/report
    figures_path = project_root / cfg.paths.figures
    return figures_path, reports_path


@app.cell
def _(reports_path):
    """
    Description:
        Reads the saved experiment results Parquet file, which contains per-epoch
        training and test metrics for all models (e.g. GATSkipRes, MLP) from the
        Geneva experiment run.

    Input:
      - reports_path (Path): path to the directory containing the Parquet result files.

    Output:
      - results (pl.DataFrame): one row per (model, epoch) combination with metric columns.
    """
    results = pl.read_parquet(reports_path / "geneva-results.parquet")
    results
    return (results,)


@app.cell
def _():
    """
    Description:
        Imports the Altair visualisation library, which is used to build interactive
        Vega-Lite charts for displaying training/validation loss curves.

    Output:
      - alt (module): the altair module, available to downstream charting cells.
    """
    import altair as alt

    return (alt,)


@app.cell
def _(alt, figures_path, results):
    """
    Description:
        Filters out the first 5 warm-up epochs, melts the wide results table into
        long format, then builds and saves a faceted Altair line chart showing BCE
        loss curves for all models.  The chart is also saved to disk as a PNG for
        inclusion in the thesis report.

    Input:
      - alt (module): the Altair visualisation library.
      - figures_path (Path): directory where the PNG output will be written.
      - results (pl.DataFrame): per-epoch metric table loaded from Parquet.

    Output:
      - fig (alt.Chart): the rendered Altair chart, displayed in the notebook.
    """
    # Drop the first 5 epochs to skip the early instability / warm-up phase
    # then reshape from wide to long format for faceted Altair chart
    _results = results.filter(pl.col("epoch") > 5).unpivot(
        on=["bce_weight"],          # columns to melt into rows
        index=["name", "epoch"],    # identifier columns kept fixed
        value_name="BCE",           # name of the new value column
        variable_name="Dataset",    # name of the new category column (e.g. "bce_weight")
    )

    # Faceted line chart: one facet per Dataset split, coloured by model name
    fig = (
        alt
        .Chart(_results)
        .mark_line(point=True)
        .encode(
            x="epoch",
            y="BCE",
            color="name",      # one line per model
            facet="Dataset",   # one subplot per metric type
            tooltip=["name", "epoch", "BCE"],
        )
    )

    # Save the chart as a PNG for inclusion in the report
    fig.save(figures_path / "results.png")

    fig
    return


@app.cell(hide_code=True)
def _():
    """
    Description: Render the section heading for the map-based prediction visualisation section.

    Output:
      - (none): renders a Markdown heading in the notebook UI.
    """
    mo.md(r"""
    # Visualising prediction on the map
    """)
    return


@app.cell
def _():
    """
    Description:
        Reloads the Geneva dataset and builds the geographic network GeoDataFrames
        needed for the map visualisation section below.

    Output:
      - gva_data (GenevaData): the loaded Geneva survey dataset.
      - network_edges (gpd.GeoDataFrame): edge geometries for the network map.
      - network_nodes (gpd.GeoDataFrame): node geometries for the network map.
    """
    from activitygraphs.data.geneva import GenevaData
    from activitygraphs.dataprocessing import load_gva_network_graph

    gva_data = GenevaData.load(cfg.data, project_root)
    network_nodes, network_edges = load_gva_network_graph(gva_data, cfg.data, project_root)
    return gva_data, network_edges, network_nodes


@app.cell
def _(test_dataset):
    """
    Description:
        Selects a specific individual from the test dataset by integer index.
        Change `index` to explore predictions for different test-set users.

    Input:
      - test_dataset (pyg.data.Dataset): the held-out test split.

    Output:
      - data (pyg.data.Data): PyG graph for the chosen individual, displayed in notebook.
    """
    index = 1             # position in the test set; change to inspect another user
    data = test_dataset[index]
    user_id = data.user_id  # string user ID attached to the Data object
    data
    return (data,)


@app.cell
def _(data, gat):
    """
    Description:
        Runs the pretrained GAT model on the selected test individual to produce
        per-node visit probability predictions.

    Input:
      - data (pyg.data.Data): PyG graph for the selected test individual.
      - gat (nn.Module): the pretrained GATSkipRes model.

    Output:
      - preds (np.ndarray): per-node visit probabilities, shape (num_nodes, 1).
            Displayed transposed (shape 1 x num_nodes) for compact viewing.
    """
    # Wrap data in a DataLoader so PyG adds the required batch-index vector
    _batch = next(iter(pyg.loader.DataLoader([data])))
    # Forward pass + sigmoid to convert logits to probabilities in [0, 1]
    preds = torch.sigmoid(gat(_batch.x, _batch.edge_index, _batch.edge_attr, _batch.batch)).detach().cpu().numpy()
    preds.T  # transposed for display: shape (1, num_nodes)
    return (preds,)


@app.cell
def _(gva_data):
    """
    Description:
        Imports the add_user_cols helper and creates two interactive widgets:
        a user-ID dropdown (limited to users with valid subsector home locations)
        and a toggle to switch between polygon and point node rendering.

    Input:
      - gva_data (GenevaData): the loaded Geneva survey dataset.

    Output:
      - add_user_cols (function): helper that enriches network_nodes with per-user columns.
      - select_user_id (mo.ui.dropdown): user selection widget.
      - toggle_polygons (mo.ui.switch): polygon/point view toggle widget.
    """
    from activitygraphs.dataprocessing import add_user_cols

    # Restrict to users whose home is assignable to a known subsector
    user_ids = gva_data.with_filter("subsector").user_ids
    # Dropdown to pick which individual to visualise on the map
    select_user_id = mo.ui.dropdown(user_ids, value=user_ids[0], label="User ID:", searchable=True)
    # Toggle between subsector polygon geometry and centroid point geometry
    toggle_polygons = mo.ui.switch(value=False, label="Show subsectors")
    return add_user_cols, select_user_id, toggle_polygons


@app.cell
def _(add_user_cols, gva_data, network_nodes, select_user_id):
    """
    Description:
        Enriches the network nodes GeoDataFrame with per-user indicator columns
        (is_visited, is_home, is_work, is_edu) for the currently selected user,
        then displays the result table.

    Input:
      - add_user_cols (function): column-enrichment helper.
      - gva_data (GenevaData): survey data with per-user location tables.
      - network_nodes (gpd.GeoDataFrame): base network node GeoDataFrame.
      - select_user_id (mo.ui.dropdown): the currently selected user ID.

    Output:
      - indiv_nodes (gpd.GeoDataFrame): network_nodes with per-user columns added.
    """
    indiv_nodes = add_user_cols(select_user_id.value, network_nodes, gva_data.location_visits, gva_data.home_locations,
                                gva_data.work_locations, gva_data.edu_locations)
    indiv_nodes
    return (indiv_nodes,)


@app.cell
def _(indiv_nodes):
    """
    Description:
        Builds a searchable dropdown widget populated with the column names of
        indiv_nodes so the user can choose which attribute to colour on the map.

    Input:
      - indiv_nodes (gpd.GeoDataFrame): node table with per-user columns.

    Output:
      - select_node_col (mo.ui.dropdown): column selector widget.
    """
    select_node_col = mo.ui.dropdown(list(indiv_nodes.columns), searchable=True, label="Column:", value="purpose")
    return (select_node_col,)


@app.cell
def _(
    indiv_nodes,
    network_edges,
    preds,
    select_node_col,
    select_user_id,
    toggle_polygons,
):
    """
    Description:
        Attaches the GAT model's per-node predictions to indiv_nodes and renders
        them on an interactive Folium map.  The "predictions" column (visit probabilities)
        determines node colour.  Network edges are shown in grey and widgets for user
        selection, column choice, and polygon toggle appear above the map.

    Input:
      - indiv_nodes (gpd.GeoDataFrame): network nodes with per-user columns.
      - network_edges (gpd.GeoDataFrame): edge geometries (drawn grey).
      - preds (np.ndarray): GAT per-node visit probabilities, shape (num_nodes, 1).
      - select_node_col (mo.ui.dropdown): column selector widget.
      - select_user_id (mo.ui.dropdown): user ID selector widget.
      - toggle_polygons (mo.ui.switch): whether to use polygon or point geometry.

    Output:
      - (displayed in notebook, nothing returned to other cells)
    """
    _pred_nodes = indiv_nodes.copy().sort_index()
    _pred_nodes["predictions"] = preds  # attach model predictions as a new column

    # Use polygon subsector geometry or default point geometry based on the toggle
    nodes = _pred_nodes.set_geometry("original_geometry") if toggle_polygons.value else _pred_nodes
    _m = network_edges.explore(color="gray", tiles="Cartodb Positron")
    _m = nodes.explore(m=_m, column="predictions", marker_kwds={"radius": 5})

    mo.vstack([
        mo.hstack([select_user_id, select_node_col, toggle_polygons], justify="start"),
        _m,
    ])
    return


if __name__ == "__main__":
    app.run()
