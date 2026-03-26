import marimo

__generated_with = "0.21.0"
app = marimo.App(width="full")

with app.setup:
    import matplotlib.pyplot as plt
    import altair as alt

    plt.style.use("default")
    alt.theme.enable("default")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    import networkx as nx
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    import torch
    import torch_geometric as pyg
    import torch.nn.functional as F


    def sigmoid(z, s=1.0):
        return 1 / (1 + np.exp(-z / s))


    def torch_sigmoid(z, s=1.0):
        return 1 / (1 + np.exp(-z / s))

    import os
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


    SEED = 512
    return F, SEED, cm, mcolors, mo, np, nx, pl, sigmoid, torch


@app.cell(hide_code=True)
def _(mo):
    slider_I = mo.ui.slider(
        start=0, stop=10000, step=100, value=2500, show_value=True, label="Number of individuals $I=$", debounce=True
    )
    slider_N = mo.ui.slider(
        steps=[i**2 for i in range(10)], value=5**2, show_value=True, label="Number of nodes $N =$", debounce=True
    )
    slider_asc = mo.ui.slider(start=0, stop=1, step=0.05, value=0.15, show_value=True, label="$\\alpha=$", debounce=True)
    slider_bn = mo.ui.slider(start=0, stop=5, step=0.05, value=2, show_value=True, label="$\\beta_1=$", debounce=True)
    slider_bh = mo.ui.slider(start=0, stop=5, step=0.05, value=0.8, show_value=True, label="$\\beta_2=$", debounce=True)
    slider_dist = mo.ui.slider(start=0, stop=5, step=0.05, value=0.5, show_value=True, label="$\\beta_d=$", debounce=True)
    return slider_I, slider_N, slider_asc, slider_bh, slider_bn, slider_dist


@app.cell
def _(asc, beta_dist, beta_home, beta_node, mo, noise_scale):
    md_utility = mo.md(
        "Utility: $\\eta_n^i = \\beta_1 X^i (X_n - \\bar{X_n}) - \\beta_2 | X_n - X^i_h | - \\beta_d X^i \\text{dist}(n, \\text{home}_i) - \\alpha$"
    )
    md_utility_num = mo.md(
        f"\t   : $\\eta_n^i = {beta_node:.2f} X^i (X_n - \\bar{{X}}) - {beta_home:.2f} | X_n - X^i_h | - {beta_dist:.2f}\\, X^i \\text{{dist}}(n, \\text{{home}}_i) - {asc:.2f}$"
    )
    md_noise = mo.md(f"Noise: $\\varepsilon^i_n \\sim \\text{{Logistic}}(0, {noise_scale})$")
    return md_noise, md_utility, md_utility_num


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Data generation process
    """)
    return


@app.cell
def _(slider_I, slider_N, slider_asc, slider_bh, slider_bn, slider_dist):
    N_sqrt = int(slider_N.value**0.5)
    noise_scale = 0.1
    N = N_sqrt**2
    I = slider_I.value
    asc = slider_asc.value
    beta_node = slider_bn.value
    beta_home = slider_bh.value
    beta_dist = slider_dist.value
    return I, N, N_sqrt, asc, beta_dist, beta_home, beta_node, noise_scale


@app.cell
def _(N_sqrt, nx):
    G = nx.navigable_small_world_graph(N_sqrt, seed=11)
    G = nx.convert_node_labels_to_integers(G)
    return (G,)


@app.cell
def _(G, np, nx):
    distances = np.array([
        [x[1] for x in sorted(row[1].items(), key=lambda x: x[0])] for row in nx.all_pairs_shortest_path_length(G)
    ])
    distances = distances / distances.max()
    return (distances,)


@app.cell
def _(I, N, SEED, asc, beta_dist, beta_home, beta_node, distances, mo, np):
    _rng = np.random.default_rng(seed=SEED)

    min_node_feature, max_node_feature = (-1, 1)
    min_indi_feature, max_indi_feature = (0, 1)

    home_locations = _rng.choice(N, I)
    node_feature = _rng.uniform(min_node_feature, max_node_feature, size=(N, 1))
    indi_feature = _rng.uniform(min_indi_feature, max_indi_feature, size=(I, 1))
    home_feature = node_feature[home_locations]

    mean_node_feature = node_feature.mean()

    _exp_node_feature = min_node_feature + (max_node_feature - min_node_feature) / 2
    _exp_indi_feature = min_indi_feature + (max_indi_feature - min_indi_feature) / 2
    _exp_home_feature = _exp_node_feature

    _exp_dist_feature = distances.mean()

    expected_utility = (
        beta_node * _exp_indi_feature * (_exp_node_feature - _exp_node_feature)
        - beta_home * np.abs(_exp_node_feature - _exp_node_feature)
        - beta_dist * _exp_dist_feature
        - asc
    )
    mo.md(f"Expected utility: $\\mathbb{{E}}[\\eta_n^i] = {expected_utility:.4f}$")
    return (
        home_feature,
        home_locations,
        indi_feature,
        mean_node_feature,
        node_feature,
    )


@app.cell
def _(G, node_feature, nx):
    nx.set_node_attributes(G, {i: f.item() for i, f in enumerate(node_feature)}, "node_feature")
    pos = nx.kamada_kawai_layout(G)
    return (pos,)


@app.cell
def _(
    I,
    N,
    SEED,
    asc,
    beta_dist,
    beta_home,
    beta_node,
    distances,
    home_feature,
    home_locations,
    indi_feature,
    mean_node_feature,
    node_feature,
    noise_scale,
    np,
):
    _rng = np.random.default_rng(seed=SEED)

    utility = (
        beta_node * indi_feature * (node_feature.T - mean_node_feature)
        - beta_home * np.abs(node_feature.T - home_feature)
        - beta_dist * indi_feature * distances[home_locations, :]
        - asc
    )
    noise = _rng.logistic(0, noise_scale, size=(I, N))
    score = utility + noise
    visited_nodes = np.where(score >= 0, 1, 0)
    return noise, score, utility, visited_nodes


@app.cell
def _(cm):
    cmap = cm.BrBG
    return (cmap,)


@app.cell
def _(
    G,
    cm,
    cmap,
    home_locations,
    mcolors,
    nx,
    pos,
    slider_indiv,
    utility,
    visited_nodes,
):
    fig_indiv_util, _ax = plt.subplots()
    fig_indiv_util.set_size_inches(5, 6)


    _i = slider_indiv.value
    _home_node = home_locations[_i].item()

    _all_other = [i for i in G.nodes if i != _home_node]

    _cmap = cmap

    nx.draw_networkx_nodes(
        G,
        nodelist=_all_other,
        pos=pos,
        node_color=utility[_i, _all_other],
        cmap=_cmap,
        ax=_ax,
        vmin=-1,
        vmax=1,
    )
    nx.draw_networkx_nodes(
        G,
        nodelist=[_home_node],
        pos=pos,
        node_color="ForestGreen",
        node_shape="s",
        edgecolors="green" if visited_nodes[_i, _home_node].item() else None,
        linewidths=1.5 if visited_nodes[_i, _home_node].item() else None,
        ax=_ax,
    )

    """nx.draw_networkx_nodes(
        G,
        nodelist=_visited_nodes,
        pos=pos,
        node_color=utility[_i, _visited_nodes],
        cmap=_cmap,
        ax=_ax,
        vmin=-1,
        vmax=1,
        edgecolors="green",
        linewidths=1.5,
    )"""

    nx.draw_networkx_labels(G, pos, ax=_ax, font_color="white")
    nx.draw_networkx_edges(G, pos=pos, ax=_ax)

    _sm = cm.ScalarMappable(cmap=_cmap, norm=mcolors.Normalize(vmin=-1, vmax=1))
    _sm.set_array(utility[_i])
    _cbar = fig_indiv_util.colorbar(_sm, ax=_ax, location='bottom', shrink=0.8)
    _cbar.set_label("Individual-specific node utility $\\eta_{{i, n}}$")

    _ax.set_axis_off()
    _ax.set_title(f"Network graph with node utilities for individual $i = {_i}$")

    None
    return (fig_indiv_util,)


@app.cell
def _(
    G,
    cm,
    cmap,
    fig_indiv_util,
    home_locations,
    mcolors,
    noise,
    nx,
    pos,
    slider_indiv,
    utility,
    visited_nodes,
):
    fig_noise, _ax = plt.subplots()
    fig_noise.set_size_inches(5, 6)

    _i = slider_indiv.value
    _home_node = home_locations[_i].item()

    _all_other = [i for i in G.nodes if i != _home_node]

    _cmap = cmap

    nx.draw_networkx_nodes(
        G,
        nodelist=_all_other,
        pos=pos,
        node_color=noise[_i, _all_other],
        cmap=_cmap,
        ax=_ax,
        vmin=-1,
        vmax=1,
    )
    nx.draw_networkx_nodes(
        G,
        nodelist=[_home_node],
        pos=pos,
        node_color="ForestGreen",
        node_shape="s",
        edgecolors="green" if visited_nodes[_i, _home_node].item() else None,
        linewidths=1.5 if visited_nodes[_i, _home_node].item() else None,
        ax=_ax,
    )

    """nx.draw_networkx_nodes(
        G,
        nodelist=_visited_nodes,
        pos=pos,
        node_color=utility[_i, _visited_nodes],
        cmap=_cmap,
        ax=_ax,
        vmin=-1,
        vmax=1,
        edgecolors="green",
        linewidths=1.5,
    )"""

    nx.draw_networkx_labels(G, pos, ax=_ax, font_color="k")
    nx.draw_networkx_edges(G, pos=pos, ax=_ax)

    _sm = cm.ScalarMappable(cmap=_cmap, norm=mcolors.Normalize(vmin=-1, vmax=1))
    _sm.set_array(utility[_i])
    _cbar = fig_indiv_util.colorbar(_sm, ax=_ax, location='bottom', shrink=0.8)
    _cbar.set_label("Noise $\\varepsilon_{i, n} \\sim \\text{Gumbel}(0,0.1)$")

    _ax.set_axis_off()
    _ax.set_title(f"Additive noise for individual $i = {_i}$")

    None
    return (fig_noise,)


@app.cell
def _(
    G,
    cm,
    cmap,
    home_locations,
    mcolors,
    np,
    nx,
    pos,
    score,
    slider_indiv,
    utility,
):
    fig_visit, _ax = plt.subplots()
    fig_visit.set_size_inches(5, 6)

    _i = slider_indiv.value
    _home_node = home_locations[_i].item()

    _score_nodes = np.where(score[_i] >= 0, 1, 0)

    _unvisited_nodes = [i for i in np.nonzero(1 - _score_nodes)[0].tolist() if i != _home_node]
    _visited_nodes = [i for i in np.nonzero(_score_nodes)[0].tolist() if i != _home_node]

    _cmap = cmap

    nx.draw_networkx_nodes(
        G,
        nodelist=_unvisited_nodes,
        pos=pos,
        node_color="white",
        ax=_ax,
        edgecolors="lightgray"
    )

    nx.draw_networkx_nodes(
        G,
        nodelist=_visited_nodes,
        pos=pos,
        node_color="DarkCyan",
        ax=_ax,
    )

    nx.draw_networkx_nodes(
        G,
        nodelist=[_home_node],
        pos=pos,
        node_color="ForestGreen",
        node_shape="s",
        ax=_ax,
    )

    nx.draw_networkx_labels(G, pos, ax=_ax, font_color="k")
    nx.draw_networkx_edges(G, pos=pos, ax=_ax)

    _sm = cm.ScalarMappable(cmap=_cmap, norm=mcolors.Normalize(vmin=-1, vmax=1))
    _sm.set_array(utility[_i])
    _cbar = fig_visit.colorbar(_sm, ax=_ax, location='bottom', shrink=0.8)

    _cbar.solids.set_alpha(0)
    _cbar.outline.set_alpha(0)
    _cbar.ax.tick_params(labelcolor='none', color='none')
    _cbar.set_label("")

    _ax.set_axis_off()
    _ax.set_title(f"Nodes visited by individual $i = {_i}$")

    None
    return (fig_visit,)


@app.cell(hide_code=True)
def _(I, mo):
    slider_indiv = mo.ui.dropdown(range(I), value=0, searchable=True, label="Selected individual $i =$")
    return (slider_indiv,)


@app.cell(hide_code=True)
def _(
    md_noise,
    md_utility,
    md_utility_num,
    mo,
    slider_I,
    slider_N,
    slider_asc,
    slider_bh,
    slider_bn,
    slider_dist,
):
    mo.vstack([slider_I, slider_N, slider_asc, slider_bn, slider_bh, slider_dist, md_utility, md_utility_num, md_noise])
    return


@app.cell
def _(fig_indiv_util, fig_noise, fig_poisson, fig_pps, fig_preds, fig_visit):
    from pathlib import Path

    _path = Path("reports/figures")
    _dpi = 200

    fig_indiv_util.savefig(_path / "synth_indiv_util.png", dpi=_dpi)
    fig_noise.savefig(_path / "synth_noise.png", dpi=_dpi)
    fig_visit.savefig(_path / "synth_visited.png", dpi=_dpi)
    fig_preds.savefig(_path / "synth_preds.png", dpi=_dpi)
    fig_poisson.savefig(_path / "synth_poisson.png", dpi=_dpi)
    fig_pps.savefig(_path / "synth_pps.png", dpi=_dpi)
    return


@app.cell(hide_code=True)
def _(
    fig_indiv_util,
    fig_noise,
    fig_visit,
    home_locations,
    indi_feature,
    mo,
    node_feature,
    slider_indiv,
):
    _x_i = indi_feature[slider_indiv.value].item()
    _x_h = node_feature[home_locations[slider_indiv.value]].item()

    mo.vstack([
        slider_indiv,
        mo.md(f"Value of individual feature: $X^i={_x_i:.4f}$"),
        mo.md(f"Value of home node feature: $X^i_h={_x_h:.4f}$"),
        mo.hstack([fig_indiv_util, fig_noise, fig_visit], justify="start"),
    ])
    return


@app.cell
def _(fig_poisson, fig_pps, fig_preds, mo):
    mo.hstack([fig_preds, fig_poisson, fig_pps], justify="start"),
    return


@app.cell(hide_code=True)
def _(mo, visited_nodes):
    num_visits = visited_nodes.sum(axis=1).mean()
    mo.md(f"Mean number of visits per individual: $\\sum_{{i = 1}}^{{I}}N^i_v = {num_visits:.2f}$")
    return


@app.cell(hide_code=True)
def _(mo, noise_scale, sigmoid, utility):
    expected_num_visits = sigmoid(utility, s=noise_scale).sum(axis=1).mean()
    mo.md(f"Expected number of visits per individual: $\\mathbb{{E}}[N^i_v] = {expected_num_visits:.2f}$")
    return


@app.cell
def _(noise_scale, sigmoid, utility, visited_nodes):
    _x = sigmoid(utility, s=noise_scale).sum(axis=1)
    _y = visited_nodes.sum(axis=1)

    plt.scatter(_x, _y, s=5)
    plt.plot([_x.min(), _x.max()], [_x.min(), _x.max()], c="red")

    plt.yticks(range(_y.min(), _y.max() + 2, 2))
    plt.xlabel("Expected number of visits")
    plt.ylabel("Actual number of visits")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## PyG Dataset generation
    """)
    return


@app.cell
def _(G, distances, home_locations, indi_feature, np, torch, visited_nodes):
    from torch_geometric.data import Data, InMemoryDataset
    from torch_geometric.utils import from_networkx

    base_data = from_networkx(G, group_node_attrs=["node_feature"])


    class SyntheticDataset(InMemoryDataset):
        def __init__(
            self,
            base_data: Data,
            indi_feature: np.ndarray,
            home_locations: np.ndarray,
            visited_nodes: np.ndarray,
            distances: np.ndarray,
        ):
            super().__init__()

            self._base_data = base_data.clone()
            self._indi_feature = indi_feature.copy()
            self._home_locations = home_locations.copy()
            self._visited_nodes = torch.tensor(visited_nodes, dtype=int)
            self._distances = torch.tensor(distances)

        @property
        def num_classes(self):
            return 1

        def len(self) -> int:
            return self._indi_feature.shape[0]

        def get(self, idx: int) -> Data:
            base_x = self._base_data.x
            base_edge_index = self._base_data.edge_index

            is_home = torch.zeros_like(base_x, dtype=int)
            is_home[self._home_locations[idx]] = 1

            indi_feature = self._indi_feature[idx].item()
            indi_node_feature = torch.full_like(base_x, indi_feature)

            home_feature = base_x[self._home_locations[idx]]
            distances = self._distances[self._home_locations[idx]].unsqueeze(1).float()

            x = torch.cat([base_x, is_home, indi_node_feature], dim=1)
            y = self._visited_nodes[idx].unsqueeze(1)

            return Data(
                x=x,
                y=y,
                edge_index=base_edge_index,
                indi_feature=indi_feature,
                home_feature=home_feature,
                user_id=idx,
                distances=distances,
            )


    dataset = SyntheticDataset(base_data, indi_feature, home_locations, visited_nodes, distances)
    dataset
    return (dataset,)


@app.cell
def _():
    test_size = 0.2
    batch_size = 128
    return batch_size, test_size


@app.cell
def _(batch_size, dataset, test_size):
    from torch_geometric.loader import DataLoader
    from sklearn.model_selection import train_test_split

    _train_indices, _test_indices = train_test_split(range(len(dataset)), test_size=test_size)

    train_dataset = dataset[_train_indices]
    test_dataset = dataset[_test_indices]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return DataLoader, test_loader, train_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Finding a good GCN for this task
    """)
    return


@app.cell
def _(dataset):
    from activitygraphs.models import NodeMLP, GCN, GCNPlus, GCNRes, GCNSkip, GATSkip

    hidden_channels = 64
    lr = 0.01
    dropout = 0.0
    epochs = 50
    verbose = 5

    min_gcn_layers, max_gcn_layers = (2, 5)
    gcnplus_lin_layers = 3
    mlp_layers = 3

    gcn_models = {
        f"GCN-{n}": GCN(
            num_layers=n,
            in_channels=dataset.num_features,
            hidden_channels=hidden_channels,
            out_channels=dataset.num_classes,
            dropout=dropout,
        )
        for n in range(min_gcn_layers, max_gcn_layers + 1)
    }

    gcn_plus_models = {
        f"GCNPlus-{n}": GCNPlus(
            num_gcn=n,
            num_lin=gcnplus_lin_layers,
            in_channels=dataset.num_features,
            hidden_channels=hidden_channels,
            out_channels=dataset.num_classes,
            dropout=dropout,
        )
        for n in range(min_gcn_layers, max_gcn_layers + 1)
    }

    gcn_res_models = {
        f"GCNRes-{n}": GCNRes(
            num_pre_layers=1,
            num_gcn_layers=n,
            num_post_layers=gcnplus_lin_layers,
            in_channels=dataset.num_features,
            hidden_channels=hidden_channels,
            out_channels=dataset.num_classes,
            dropout=dropout,
        )
        for n in range(min_gcn_layers, max_gcn_layers + 1)
    }

    gcn_skip_models = {
        f"GCNSkip-{n}{'-res' if res else ''}": GCNSkip(
            num_pre_layers=1,
            num_gcn_layers=n,
            num_post_layers=gcnplus_lin_layers,
            in_channels=dataset.num_features,
            hidden_channels=hidden_channels,
            out_channels=dataset.num_classes,
            dropout=dropout,
            residuals=res,
        )
        for n in range(min_gcn_layers, max_gcn_layers + 1)
        for res in [True, False]
    }

    gat_skip_models = {
        f"GATSkip-{n}{'-res' if res else ''}": GATSkip(
            num_pre_layers=1,
            num_gcn_layers=n,
            num_post_layers=gcnplus_lin_layers,
            in_channels=dataset.num_features,
            hidden_channels=hidden_channels,
            out_channels=dataset.num_classes,
            edge_dim=1,
            dropout=dropout,
            residuals=res,
        )
        for n in range(min_gcn_layers, max_gcn_layers + 1)
        for res in [True, False]
    }

    other_models = {
        "MLP": NodeMLP(
            mlp_layers,
            in_channels=dataset.num_features,
            hidden_channels=hidden_channels,
            out_channels=dataset.num_classes,
            dropout=dropout,
        ),
        "MLP-Full": NodeMLP(
            mlp_layers,
            in_channels=dataset.num_features + 2,
            hidden_channels=hidden_channels,
            out_channels=dataset.num_classes,
            dropout=dropout,
        ),
    }

    full_info_models = {"MLP-Full"}

    models = other_models | gcn_models | gcn_plus_models | gcn_res_models | gcn_skip_models | gat_skip_models
    list(models.keys())
    return (
        GATSkip,
        dropout,
        epochs,
        full_info_models,
        gcnplus_lin_layers,
        hidden_channels,
        lr,
        models,
        verbose,
    )


@app.cell(hide_code=True)
def _(mo):
    btn_run_experiments = mo.ui.run_button(label="Run model training")
    btn_run_experiments
    return (btn_run_experiments,)


@app.cell
def _(
    btn_run_experiments,
    epochs,
    full_info_models,
    lr,
    mo,
    models,
    pl,
    test_loader,
    train_loader,
    verbose,
):
    from activitygraphs.experiment import run_experiment

    mo.stop(not btn_run_experiments.value)

    _results = {}

    for name, model in models.items():
        _full_info = name in full_info_models
        _results[name] = run_experiment(
            model, train_loader, test_loader, num_epochs=epochs, verbose=verbose, name=name, lr=lr, full_info=_full_info
        )

    results = pl.concat(pl.DataFrame(result) for result in _results.values())
    results.write_parquet("reports/data/synthetic-results.parquet")
    return (run_experiment,)


@app.cell(hide_code=True)
def _(pl):
    _results = pl.read_parquet("reports/data/synthetic-results.parquet")
    # _results = results


    results_aug = _results.with_columns(type=pl.col("name").str.split("-").list.first())
    results_aug
    return (results_aug,)


@app.cell(hide_code=True)
def _(results_aug):
    def plot_results(results, title, column, color="name:N", detail=None, scheme="inferno"):
        kwargs = {} if detail is None else {"detail": detail}
        scheme_kwargs = {} if scheme is None else {"scheme": scheme}

        return (
            alt
            .Chart(results)
            .mark_line(point=True)
            .encode(
                alt.X("epoch:Q").scale(domainMin=1),
                alt.Y(f"{column}:Q").scale(domainMin=0.09),
                color=alt.Color(color).scale(**scheme_kwargs),
                tooltip=["name", column],
                **kwargs,
            )
            .properties(title=title, width=600, height=450)
        )


    plot_results(results_aug, "Training loss", "train") | plot_results(results_aug, "Test loss", "test")
    return (plot_results,)


@app.cell(hide_code=True)
def _(pl, plot_results, results_aug):
    _best_performers = pl.concat([
        results_aug.group_by("type").agg(pl.all().sort_by("test").first()).select("name"),
        pl.DataFrame({"name": "MLP"}),
    ])
    _results = results_aug.join(_best_performers, on="name")

    plot_results(_results, "Training loss", "train") | plot_results(_results, "Test loss", "test")
    return


@app.cell(hide_code=True)
def _(mo, results_aug):
    dropdown_type = mo.ui.dropdown(results_aug["type"].unique(), label="Choose type of GCN model:")
    return (dropdown_type,)


@app.cell(hide_code=True)
def _(dropdown_type, mo, pl, plot_results, results_aug):
    _results = results_aug.with_columns(layers=pl.col("name").str.extract(r"(\d)").fill_null("MLP"))

    if dropdown_type.value:
        _results = _results.filter((pl.col("type") == dropdown_type.value) | (pl.col("type") == "MLP"))


    _fig = plot_results(_results, "Training loss", "train", color="layers", detail="name", scheme=None) | plot_results(
        _results, "Test loss", "test", color="layers", detail="name"
    )

    mo.vstack([dropdown_type, _fig])
    return


@app.cell(hide_code=True)
def _(F, noise_scale, pl, results_aug, sigmoid, torch, utility, visited_nodes):
    _best_possible_theoretical_loss = F.binary_cross_entropy(
        torch.tensor(sigmoid(utility, s=noise_scale)).float(), torch.tensor(sigmoid(utility, s=noise_scale)).float()
    )

    _best_possible_empirical_loss = F.binary_cross_entropy(
        torch.tensor(sigmoid(utility, s=noise_scale)).float(), torch.tensor(visited_nodes).float()
    )

    _random_guessing_loss = F.binary_cross_entropy(
        torch.full_like(torch.tensor(visited_nodes), 0.5).float(), torch.tensor(visited_nodes).float()
    )

    _theoretical_bound = pl.DataFrame({
        "name": "Theoretical bound",
        "epoch": "-",
        "test": _best_possible_theoretical_loss,
        "type": "Baseline",
    })
    _empirical_bound = pl.DataFrame({
        "name": "Empirical bound",
        "epoch": "-",
        "test": _best_possible_empirical_loss,
        "type": "Baseline",
    })
    _random_guessing_bound = pl.DataFrame({
        "name": "Random guessing bound",
        "epoch": "-",
        "test": _random_guessing_loss,
        "type": "Baseline",
    })


    _best_models = (
        results_aug
        .group_by("name")
        .agg(pl.all().sort_by("test").first())
        .sort("test")
        .select("name", pl.col("epoch").cast(pl.String), "test", "type")
    )

    model_comparison = pl.concat([_theoretical_bound, _empirical_bound, _best_models, _random_guessing_bound])
    model_comparison
    return (model_comparison,)


@app.cell(hide_code=True)
def _(model_comparison, pl):
    _bar = (
        alt
        .Chart(model_comparison.filter(~pl.col("name").str.contains("bound")))
        .mark_bar()
        .encode(
            alt.X("name:N").title("Model").sort("y"),
            y=alt.Y("test:Q").title("Test loss"),
            color=alt.Color("type:N").scale(scheme="inferno"),
            tooltip=["name", "test"],
        )
    )

    _theo_rule = (
        alt
        .Chart(model_comparison.filter(pl.col("name") == "Theoretical bound"))
        .mark_rule(color="green")
        .encode(y="min(test):Q", tooltip="min(test):Q")
    )
    _emp_rule = (
        alt
        .Chart(model_comparison.filter(pl.col("name") == "Empirical bound"))
        .mark_rule(color="red")
        .encode(y="min(test):Q", tooltip="min(test):Q")
    )

    _mlp_rule = (
        alt
        .Chart(model_comparison.filter(pl.col("name") == "MLP"))
        .mark_rule(color="red")
        .encode(y="min(test):Q", tooltip="min(test):Q")
    )

    _mlp_rule + _bar + _theo_rule + _emp_rule
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Conclusions:
    - Plain GCN can't beat the MLP
    - GCNPlus can, but struggles
    - Residual connections help, but aren't the best fix
    - Skip connections help the most, almost equal to MLP-Full
    - Skip + Residual not better than just Skip
    - 4 or 5 layers ideal
    - GATConv is more stable, performs the best and converges faster
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualising predictions
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    btn_train_best_model = mo.ui.run_button(label="Train best GCN model")
    btn_train_best_model
    return (btn_train_best_model,)


@app.cell
def _(
    GATSkip,
    btn_train_best_model,
    dataset,
    dropout,
    gcnplus_lin_layers,
    hidden_channels,
    lr,
    mo,
    run_experiment,
    test_loader,
    train_loader,
):
    mo.stop(not btn_train_best_model.value)

    best_num_layers = 5

    gat = GATSkip(
        num_pre_layers=1,
        num_gcn_layers=best_num_layers,
        num_post_layers=gcnplus_lin_layers,
        in_channels=dataset.num_features,
        hidden_channels=hidden_channels,
        out_channels=dataset.num_classes,
        dropout=dropout,
        edge_dim=1,
        residuals=True,
    )

    _ = run_experiment(gat, train_loader, test_loader, num_epochs=50, verbose=10, lr=lr)

    gat
    return (gat,)


@app.cell
def _(DataLoader, dataset, gat, slider_indiv, torch):
    _i = slider_indiv.value
    _loader = DataLoader(dataset[_i:_i + 1])
    batch = next(iter(_loader)).cuda()

    _logits = gat(batch.x, batch.edge_index)
    probs = torch.sigmoid(_logits).detach().cpu().numpy()
    return batch, probs


@app.cell
def _(
    G,
    cm,
    cmap,
    home_locations,
    mcolors,
    nx,
    pos,
    probs,
    slider_indiv,
    visited_nodes,
):
    fig_preds, _ax = plt.subplots()
    fig_preds.set_size_inches(5, 6)


    _i = slider_indiv.value
    _home_node = home_locations[_i].item()

    _all_other = [i for i in G.nodes if i != _home_node]

    _cmap = cmap

    nx.draw_networkx_nodes(
        G,
        nodelist=_all_other,
        pos=pos,
        node_color=probs[_all_other],
        cmap=_cmap,
        ax=_ax,
        vmin=0,
        vmax=1,
    )
    nx.draw_networkx_nodes(
        G,
        nodelist=[_home_node],
        pos=pos,
        node_color="ForestGreen",
        node_shape="s",
        edgecolors="green" if visited_nodes[_i, _home_node].item() else None,
        linewidths=1.5 if visited_nodes[_i, _home_node].item() else None,
        ax=_ax,
    )

    """nx.draw_networkx_nodes(
        G,
        nodelist=_visited_nodes,
        pos=pos,
        node_color=utility[_i, _visited_nodes],
        cmap=_cmap,
        ax=_ax,
        vmin=-1,
        vmax=1,
        edgecolors="green",
        linewidths=1.5,
    )"""

    nx.draw_networkx_labels(G, pos, ax=_ax, font_color="white")
    nx.draw_networkx_edges(G, pos=pos, ax=_ax)

    _sm = cm.ScalarMappable(cmap=_cmap, norm=mcolors.Normalize(vmin=0, vmax=1))
    _sm.set_array(probs)
    _cbar = fig_preds.colorbar(_sm, ax=_ax, location='bottom', shrink=0.8)
    _cbar.set_label("Node visit probability $\\hat{y}_{{i, n}}$")

    _ax.set_axis_off()
    _ax.set_title(f"GATSkip predicted visit probabilites for individual $i = {_i}$")

    fig_preds
    return (fig_preds,)


@app.cell
def _(SEED, batch, gat, torch):
    from activitygraphs.sampling import poisson_sampling, pps_sampling

    _logits = gat(batch.x, batch.edge_index).detach()
    _generator = torch.Generator(device="cuda").manual_seed(SEED)

    num_pps_nodes = 7

    poisson_visits = poisson_sampling(_logits, generator=_generator).cpu().numpy()
    pps_visits = pps_sampling(num_pps_nodes, _logits, batch.batch, generator=_generator).cpu().numpy()

    poisson_visits
    return num_pps_nodes, poisson_visits, pps_visits


@app.cell
def _(fig_poisson):
    fig_poisson
    return


@app.cell
def _(
    G,
    cm,
    cmap,
    home_locations,
    mcolors,
    np,
    nx,
    poisson_visits,
    pos,
    slider_indiv,
    utility,
):
    fig_poisson, _ax = plt.subplots()
    fig_poisson.set_size_inches(5, 6)

    _i = slider_indiv.value
    _home_node = home_locations[_i].item()

    _score_nodes = poisson_visits

    _unvisited_nodes = [i for i in np.nonzero(1 - _score_nodes)[0].tolist() if i != _home_node]
    _visited_nodes = [i for i in np.nonzero(_score_nodes)[0].tolist() if i != _home_node]

    _cmap = cmap

    nx.draw_networkx_nodes(
        G,
        nodelist=_unvisited_nodes,
        pos=pos,
        node_color="white",
        ax=_ax,
        edgecolors="lightgray"
    )

    nx.draw_networkx_nodes(
        G,
        nodelist=_visited_nodes,
        pos=pos,
        node_color="DarkCyan",
        ax=_ax,
    )

    nx.draw_networkx_nodes(
        G,
        nodelist=[_home_node],
        pos=pos,
        node_color="ForestGreen",
        node_shape="s",
        ax=_ax,
    )

    nx.draw_networkx_labels(G, pos, ax=_ax, font_color="k")
    nx.draw_networkx_edges(G, pos=pos, ax=_ax)

    _sm = cm.ScalarMappable(cmap=_cmap, norm=mcolors.Normalize(vmin=-1, vmax=1))
    _sm.set_array(utility[_i])
    _cbar = fig_poisson.colorbar(_sm, ax=_ax, location='bottom', shrink=0.8)

    _cbar.solids.set_alpha(0)
    _cbar.outline.set_alpha(0)
    _cbar.ax.tick_params(labelcolor='none', color='none')
    _cbar.set_label("")

    _ax.set_axis_off()
    _ax.set_title("Poisson-sampled location choice set")

    fig_poisson
    return (fig_poisson,)


@app.cell
def _(
    G,
    cm,
    cmap,
    fig_poisson,
    home_locations,
    mcolors,
    np,
    num_pps_nodes,
    nx,
    pos,
    pps_visits,
    slider_indiv,
    utility,
):
    fig_pps, _ax = plt.subplots()
    fig_pps.set_size_inches(5, 6)

    _i = slider_indiv.value
    _home_node = home_locations[_i].item()

    _score_nodes = pps_visits

    _unvisited_nodes = [i for i in np.nonzero(1 - _score_nodes)[0].tolist() if i != _home_node]
    _visited_nodes = [i for i in np.nonzero(_score_nodes)[0].tolist() if i != _home_node]

    _cmap = cmap

    nx.draw_networkx_nodes(
        G,
        nodelist=_unvisited_nodes,
        pos=pos,
        node_color="white",
        ax=_ax,
        edgecolors="lightgray"
    )

    nx.draw_networkx_nodes(
        G,
        nodelist=_visited_nodes,
        pos=pos,
        node_color="DarkCyan",
        ax=_ax,
    )

    nx.draw_networkx_nodes(
        G,
        nodelist=[_home_node],
        pos=pos,
        node_color="ForestGreen",
        node_shape="s",
        ax=_ax,
    )

    nx.draw_networkx_labels(G, pos, ax=_ax, font_color="k")
    nx.draw_networkx_edges(G, pos=pos, ax=_ax)

    _sm = cm.ScalarMappable(cmap=_cmap, norm=mcolors.Normalize(vmin=-1, vmax=1))
    _sm.set_array(utility[_i])
    _cbar = fig_poisson.colorbar(_sm, ax=_ax, location='bottom', shrink=0.8)

    _cbar.solids.set_alpha(0)
    _cbar.outline.set_alpha(0)
    _cbar.ax.tick_params(labelcolor='none', color='none')
    _cbar.set_label("")

    _ax.set_axis_off()
    _ax.set_title(f"PPS-sampled location choice set of size $m = {num_pps_nodes}$ ")

    fig_pps
    return (fig_pps,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
