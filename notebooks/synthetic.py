import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import altair as alt

    import torch
    import torch_geometric as pyg
    import torch.nn.functional as F


    def sigmoid(z, s=1.0):
        return 1 / (1 + np.exp(-z / s))


    def torch_sigmoid(z, s=1.0):
        return 1 / (1 + np.exp(-z / s))


    SEED = 512
    return F, SEED, alt, cm, mcolors, mo, np, nx, pl, plt, sigmoid, torch


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
    return slider_I, slider_N, slider_asc, slider_bh, slider_bn


@app.cell
def _(asc, beta_home, beta_node, mo, noise_scale):
    md_utility = mo.md("Utility: $\\eta_n^i = \\beta_1 X^i (X_n - \\bar{X}) - \\beta_2 | X_n - X^i_h | - \\alpha$")
    md_utility_num = mo.md(
        f"\t   : $\\eta_n^i = {beta_node:.2f} X^i (X_n - \\bar{{X}}) - {beta_home:.2f} | X_n - X^i_h | - {asc:.2f}$"
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
def _(slider_I, slider_N, slider_asc, slider_bh, slider_bn):
    N_sqrt = int(slider_N.value**0.5)
    noise_scale = 0.1
    N = N_sqrt**2
    I = slider_I.value
    asc = slider_asc.value
    beta_node = slider_bn.value
    beta_home = slider_bh.value
    return I, N, N_sqrt, asc, beta_home, beta_node, noise_scale


@app.cell
def _(I, N, SEED, asc, beta_home, beta_node, mo, np):
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

    expected_utility = (
        beta_node * _exp_indi_feature * (_exp_node_feature - _exp_node_feature)
        - beta_home * np.abs(_exp_node_feature - _exp_node_feature)
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
def _(
    I,
    N,
    SEED,
    asc,
    beta_home,
    beta_node,
    home_feature,
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
        - asc
    )
    noise = _rng.logistic(0, noise_scale, size=(I, N))
    score = utility + noise
    visited_nodes = np.where(score >= 0, 1, 0)
    return score, utility, visited_nodes


@app.cell
def _(N_sqrt, node_feature, nx):
    G = nx.navigable_small_world_graph(N_sqrt, seed=11)
    G = nx.convert_node_labels_to_integers(G)
    nx.set_node_attributes(G, {i: f.item() for i, f in enumerate(node_feature)}, "node_feature")
    pos = nx.kamada_kawai_layout(G)
    return G, pos


@app.cell(hide_code=True)
def _(G, cm, mcolors, node_feature, np, nx, plt, pos):
    fig_base, _ax = plt.subplots()

    _base_cmap = cm.BrBG

    _vmax = np.max(np.abs(node_feature))
    _norm = mcolors.Normalize(vmin=-_vmax, vmax=_vmax)
    _cmap = mcolors.LinearSegmentedColormap.from_list("trunc_Reds", _base_cmap(np.linspace(0, 0.7, 256)))

    nx.draw_networkx(G, pos=pos, node_color=node_feature, cmap=_cmap, ax=_ax, vmin=-_vmax, vmax=_vmax)

    _sm = cm.ScalarMappable(cmap=_cmap, norm=_norm)
    _sm.set_array(node_feature)
    _cbar = fig_base.colorbar(_sm, ax=_ax)
    _cbar.set_label("$X_n$")

    _ax.set_axis_off()
    _ax.set_title("Base network")

    None
    return (fig_base,)


@app.cell(hide_code=True)
def _(
    G,
    cm,
    home_locations,
    mcolors,
    np,
    nx,
    plt,
    pos,
    score,
    slider_indiv,
    visited_nodes,
):
    fig_indiv, _ax = plt.subplots()

    _i = slider_indiv.value
    _home_node = home_locations[_i].item()

    _score_nodes = np.where(score[_i] >= 0, 1, 0)

    _unvisited_nodes = [i for i in np.nonzero(1 - _score_nodes)[0].tolist() if i != _home_node]
    _visited_nodes = [i for i in np.nonzero(_score_nodes)[0].tolist() if i != _home_node]

    _cmap = cm.RdBu


    nx.draw_networkx_nodes(
        G, nodelist=_unvisited_nodes, pos=pos, node_color=score[_i, _unvisited_nodes], cmap=_cmap, ax=_ax, vmin=-1, vmax=1
    )
    nx.draw_networkx_nodes(
        G,
        nodelist=[_home_node],
        pos=pos,
        node_color="ForestGreen",
        edgecolors="green" if visited_nodes[_i, _home_node].item() else None,
        linewidths=1.5 if visited_nodes[_i, _home_node].item() else None,
        ax=_ax,
    )
    nx.draw_networkx_nodes(
        G,
        nodelist=_visited_nodes,
        pos=pos,
        node_color=score[_i, _visited_nodes],
        cmap=_cmap,
        ax=_ax,
        vmin=-1,
        vmax=1,
        edgecolors="green",
        linewidths=1.5,
    )

    nx.draw_networkx_labels(G, pos, ax=_ax)
    nx.draw_networkx_edges(G, pos=pos, ax=_ax)

    _sm = cm.ScalarMappable(cmap=_cmap, norm=mcolors.Normalize(vmin=-1, vmax=1))
    _sm.set_array(score[_i])
    _cbar = fig_indiv.colorbar(_sm, ax=_ax)
    _cbar.set_label("$Z^i_n$")

    _ax.set_axis_off()
    _ax.set_title(f"Network for individual $i = {_i}$")

    None
    return (fig_indiv,)


@app.cell(hide_code=True)
def _(
    G,
    cm,
    home_locations,
    mcolors,
    np,
    nx,
    plt,
    pos,
    slider_indiv,
    utility,
    visited_nodes,
):
    fig_indiv_util, _ax = plt.subplots()

    _i = slider_indiv.value
    _home_node = home_locations[_i].item()

    _utility_nodes = np.where(utility[_i] >= 0, 1, 0)

    _unvisited_nodes = [i for i in np.nonzero(1 - _utility_nodes)[0].tolist() if i != _home_node]
    _visited_nodes = [i for i in np.nonzero(_utility_nodes)[0].tolist() if i != _home_node]

    _cmap = cm.RdBu

    nx.draw_networkx_nodes(
        G,
        nodelist=_unvisited_nodes,
        pos=pos,
        node_color=utility[_i, _unvisited_nodes],
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
        edgecolors="green" if visited_nodes[_i, _home_node].item() else None,
        linewidths=1.5 if visited_nodes[_i, _home_node].item() else None,
        ax=_ax,
    )
    nx.draw_networkx_nodes(
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
    )

    nx.draw_networkx_labels(G, pos, ax=_ax)
    nx.draw_networkx_edges(G, pos=pos, ax=_ax)

    _sm = cm.ScalarMappable(cmap=_cmap, norm=mcolors.Normalize(vmin=-1, vmax=1))
    _sm.set_array(utility[_i])
    _cbar = fig_indiv_util.colorbar(_sm, ax=_ax)
    _cbar.set_label("$\\eta_n^i$")

    _ax.set_axis_off()
    _ax.set_title(f"Network for individual $i = {_i}$ without noise $\\varepsilon^i_n$")

    None
    return (fig_indiv_util,)


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
):
    mo.vstack([slider_I, slider_N, slider_asc, slider_bn, slider_bh, md_utility, md_utility_num, md_noise])
    return


@app.cell(hide_code=True)
def _(
    fig_base,
    fig_indiv,
    fig_indiv_util,
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
        mo.hstack([fig_base, fig_indiv, fig_indiv_util], justify="start"),
    ])
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
def _(noise_scale, plt, sigmoid, utility, visited_nodes):
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
def _(G, home_locations, indi_feature, np, torch, visited_nodes):
    from torch_geometric.data import Data, InMemoryDataset
    from torch_geometric.utils import from_networkx

    base_data = from_networkx(G, group_node_attrs=["node_feature"])


    class SyntheticDataset(InMemoryDataset):
        def __init__(
            self, base_data: Data, indi_feature: np.ndarray, home_locations: np.ndarray, visited_nodes: np.ndarray
        ):
            super().__init__()

            self._base_data = base_data.clone()
            self._indi_feature = indi_feature.copy()
            self._home_locations = home_locations.copy()
            self._visited_nodes = torch.tensor(visited_nodes, dtype=int)

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

            x = torch.cat([base_x, is_home, indi_node_feature], dim=1)
            y = self._visited_nodes[idx].unsqueeze(1)

            return Data(
                x=x, y=y, edge_index=base_edge_index, indi_feature=indi_feature, home_feature=home_feature, user_id=idx
            )


    dataset = SyntheticDataset(base_data, indi_feature, home_locations, visited_nodes)
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
    return test_dataset, test_loader, train_loader


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
            in_channels=dataset.num_features + 1,
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
    return (run_experiment,)


@app.cell(hide_code=True)
def _(pl):
    _results = pl.read_parquet("reports/data/synthetic-results.parquet")
    # _results = results


    results_aug = _results.with_columns(type=pl.col("name").str.split("-").list.first())
    results_aug
    return (results_aug,)


@app.cell(hide_code=True)
def _(alt, results_aug):
    def plot_results(results, title, column, color="name:N", detail=None, scheme="inferno"):
        kwargs = {} if detail is None else {"detail": detail}
        scheme_kwargs = {} if scheme is None else {"scheme": scheme}

        return (
            alt
            .Chart(results)
            .mark_line(point=True)
            .encode(
                alt.X("epoch:Q").scale(domainMin=1),
                alt.Y(f"{column}:Q").scale(domainMin=0.10),
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
def _(alt, model_comparison, pl):
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
        residuals=False,
    )

    _ = run_experiment(gat, train_loader, test_loader, num_epochs=50, verbose=10, lr=lr)

    gat
    return (gat,)


@app.cell(hide_code=True)
def _(mo, test_dataset):
    dropdown_predictions = mo.ui.dropdown(
        range(len(test_dataset)),
        value=0,
        allow_select_none=False,
        searchable=True,
        label="Predictions for test datapoint:",
    )
    return (dropdown_predictions,)


@app.cell(hide_code=True)
def _(
    G,
    cm,
    dropdown_predictions,
    fig_base,
    gat,
    mcolors,
    mo,
    noise_scale,
    np,
    nx,
    plt,
    pos,
    score,
    sigmoid,
    test_dataset,
    torch,
):
    _i = dropdown_predictions.value
    _data = test_dataset[_i]
    _preds = torch.sigmoid(gat(_data.x, _data.edge_index)).squeeze().detach().numpy()

    # ==================

    _fig, ((_ax1, _ax2), (_ax3, _ax4)) = plt.subplots(2, 2, figsize=(10, 6))

    _base_cmap = cm.PiYG
    _norm = mcolors.Normalize(vmin=0, vmax=1)
    _cmap = _base_cmap

    nx.draw_networkx(G, pos=pos, node_color=_preds, cmap=_cmap, ax=_ax1, vmin=0, vmax=1)

    _sm = cm.ScalarMappable(cmap=_cmap, norm=_norm)
    _sm.set_array(_preds)
    _cbar = fig_base.colorbar(_sm, ax=_ax1)
    _cbar.set_label("$\\bar{P}(s^i_n = 1 | X^i, X_n, X_h^i)$")

    _ax1.set_axis_off()
    _ax1.set_title("Predicted probablities of visit")

    # ==================

    _probs = sigmoid(score[_data.user_id], noise_scale)

    nx.draw_networkx(G, pos=pos, node_color=_probs, cmap=_cmap, ax=_ax2, vmin=0, vmax=1)

    _sm = cm.ScalarMappable(cmap=_cmap, norm=_norm)
    _sm.set_array(_probs)
    _cbar = fig_base.colorbar(_sm, ax=_ax2)
    _cbar.set_label("$P(s^i_n = 1 | X^i, X_n, X_h^i)$")

    _ax2.set_axis_off()
    _ax2.set_title("Actual probablities of visit")

    # ==================

    nx.draw_networkx(G, pos=pos, node_color=_data.y, cmap=_cmap, ax=_ax3, vmin=0, vmax=1)

    _sm = cm.ScalarMappable(cmap=_cmap, norm=_norm)
    _sm.set_array(_data.y)
    _cbar = fig_base.colorbar(_sm, ax=_ax3)
    _cbar.set_label("$s^i_n = 1$")

    _ax3.set_axis_off()
    _ax3.set_title("Actual visited nodes")

    # ==================

    _residuals = _probs - _preds

    _vmax = np.abs(_residuals).max()
    _norm = mcolors.Normalize(vmin=-_vmax, vmax=_vmax)

    nx.draw_networkx(G, pos=pos, node_color=_residuals, cmap=cm.RdBu, ax=_ax4, vmin=-_vmax, vmax=_vmax)

    _sm = cm.ScalarMappable(cmap=cm.RdBu, norm=_norm)
    _sm.set_array(_residuals)
    _cbar = fig_base.colorbar(_sm, ax=_ax4)
    _cbar.set_label("$P(s^i_n = 1) - \\bar{P}(s^i_n = 1)$")

    _ax4.set_axis_off()
    _ax4.set_title("Residuals")

    mo.vstack([dropdown_predictions, _fig])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
