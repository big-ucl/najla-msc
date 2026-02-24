import marimo

__generated_with = "0.19.11"
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

    import torch
    import torch_geometric as pyg
    import torch.nn.functional as F


    def sigmoid(z, s=1.0):
        return 1 / (1 + np.exp(-z / s))


    def torch_sigmoid(z, s=1.0):
        return 1 / (1 + np.exp(-z / s))


    SEED = 512
    return F, SEED, cm, mcolors, mo, np, nx, pl, plt, sigmoid, torch


@app.cell(hide_code=True)
def _(mo):
    slider_I = mo.ui.slider(start=0, stop=10000, step=100, value=2500, show_value=True, label="Number of individuals $I=$")
    slider_N = mo.ui.slider(steps=[i**2 for i in range(10)], value=5**2, show_value=True, label="Number of nodes $N =$")
    slider_B = mo.ui.slider(start=0, stop=5, step=0.05, value=0.5, show_value=True, label="Home node penalty $\\beta=$")
    slider_s = mo.ui.slider(
        start=0, stop=1, step=0.01, value=0.1, show_value=True, label="Scale of noise distribution $s=$"
    )

    md_utility = mo.md("Utility: $\\;\\eta_n^i = X^i X_n - \\beta X^i_h$")
    md_noise = mo.md("Noise: $\\varepsilon^i_n \\sim \\text{Logistic}(0, s)$")
    return md_noise, md_utility, slider_B, slider_I, slider_N, slider_s


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Data generation process
    """)
    return


@app.cell
def _(slider_B, slider_I, slider_N):
    N_sqrt = int(slider_N.value**0.5)
    N = N_sqrt**2
    I = slider_I.value
    beta_home = slider_B.value
    return I, N, N_sqrt, beta_home


@app.cell
def _(I, N, SEED, beta_home, mo, np):
    _rng = np.random.default_rng(seed=SEED)

    min_node_feature, max_node_feature = (-0.9, 0.1)
    min_indi_feature, max_indi_feature = (0.2, 0.8)

    home_locations = _rng.choice(N, I)
    node_feature = _rng.uniform(min_node_feature, max_node_feature, size=(N, 1))
    indi_feature = _rng.uniform(min_indi_feature, max_indi_feature, size=(I, 1))
    home_feature = node_feature[home_locations]

    _exp_node_feature = min_node_feature + (max_node_feature - min_node_feature) / 2
    _exp_indi_feature = min_indi_feature + (max_indi_feature - min_indi_feature) / 2
    _exp_home_feature = _exp_node_feature

    expected_utility = _exp_node_feature - beta_home * _exp_home_feature * _exp_indi_feature
    mo.md(f"Expected utility: $\\mathbb{{E}}[\\eta_n^i] = {expected_utility:.4f}$")
    return home_feature, home_locations, indi_feature, node_feature


@app.cell
def _(
    I,
    N,
    SEED,
    beta_home,
    home_feature,
    indi_feature,
    node_feature,
    np,
    slider_s,
):
    _rng = np.random.default_rng(seed=SEED)

    utility = node_feature.T - beta_home * home_feature * indi_feature
    noise = _rng.logistic(0, slider_s.value, size=(I, N))
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
def _(md_noise, md_utility, mo, slider_B, slider_I, slider_N, slider_s):
    mo.vstack([slider_I, slider_N, slider_B, slider_s, md_utility, md_noise])
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
def _(mo, sigmoid, slider_s, utility):
    expected_num_visits = sigmoid(utility, s=slider_s.value).sum(axis=1).mean()
    mo.md(f"Expected number of visits per individual: $\\mathbb{{E}}[N^i_v] = {expected_num_visits:.2f}$")
    return


@app.cell
def _(plt, sigmoid, slider_s, utility, visited_nodes):
    _x = sigmoid(utility, s=slider_s.value).sum(axis=1)
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
    return test_loader, train_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Finding a good GCN for this task
    """)
    return


@app.cell
def _(F, torch):
    from torch_geometric.nn import GCNConv
    import itertools


    def build_module_list(
        module_f, num_layers: int, in_channels: int, out_channels: int, hidden_channels: int | None = None, **module_kwargs
    ):
        if num_layers > 1 and hidden_channels is None:
            raise ValueError(f"Hidden channels not provided: {num_layers=} or {hidden_channels=}")

        if num_layers < 1:
            raise ValueError(f"Invalid number of layers: {num_layers=}")

        layers = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        modules = torch.nn.ModuleList()

        for num_in, num_out in itertools.pairwise(layers):
            modules.append(module_f(num_in, num_out, **module_kwargs))

        return modules


    class GCN(torch.nn.Module):
        def __init__(
            self, num_layers: int, in_channels: int, hidden_channels: int, out_channels: int, dropout=0.2, residuals=False
        ):
            super().__init__()

            if residuals and in_channels != hidden_channels:
                raise ValueError("Number of in channels must match number of hidden channels")

            self.residuals = residuals
            self.dropout = dropout
            self.convs = build_module_list(GCNConv, in_channels, out_channels, hidden_channels)

        def forward(self, x, edge_index):
            for conv in self.convs[:-1]:
                x_res = x
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = conv(x, edge_index).relu()

                if self.residuals:
                    x = x + x_res

            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[-1](x, edge_index)

            return x


    class NodeMLP(torch.nn.Module):
        def __init__(self, num_layers: int, in_channels: int, hidden_channels: int, out_channels: int, dropout=0.2):
            super().__init__()

            self.dropout = dropout
            self.lins = build_module_list(torch.nn.Linear, in_channels, out_channels, hidden_channels)

        def forward(self, x, edge_index):
            for lin in self.lins[:-1]:
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = lin(x).relu()

            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lins[-1](x)

            return x


    class GCNPlus(torch.nn.Module):
        def __init__(
            self, num_gcn: int, num_lin: int, in_channels: int, hidden_channels: int, out_channels: int, dropout=0.2
        ):
            super().__init__()

            self.gcn = GCN(num_gcn, in_channels, hidden_channels, hidden_channels, dropout)
            self.lin = NodeMLP(num_lin, hidden_channels, hidden_channels, out_channels, dropout)

        def forward(self, x, edge_index):
            x = self.gcn(x, edge_index).relu()
            x = self.lin(x, edge_index)

            return x


    class GCNRes(torch.nn.Module):
        def __init__(
            self,
            num_pre_layers: int,
            num_gcn_layers: int,
            num_post_layers: int,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
            dropout=0.2,
        ):
            super().__init__()

            self.dropout = dropout
            self.pre_lin = NodeMLP(num_pre_layers, in_channels, hidden_channels, hidden_channels, dropout)
            self.convs = GCN(num_gcn_layers, in_channels, hidden_channels, hidden_channels, dropout, residuals=True)
            self.post_lin = NodeMLP(num_post_layers, hidden_channels, hidden_channels, out_channels, dropout)

        def forward(self, x, edge_index):
            x = self.pre_lin(x, edge_index)
            x = self.convs(x, edge_index)
            x = self.post_lin(x, edge_index)

            return x

    return GCN, GCNPlus, NodeMLP


@app.cell
def _(F, torch):
    def compute_training_weights(loader):
        num_neg = 0
        num_pos = 0

        for batch in loader:
            num_neg += (batch.y == 0).sum()
            num_pos += batch.y.sum()
        return num_neg / num_pos


    def extract_features(batch, full_info):
        if not full_info:
            return batch.x

        if batch.batch is not None:
            home_feature = batch.home_feature[batch.batch].unsqueeze(1)
        else:
            home_feature = torch.full((batch.x.shape[0], 1), batch.home_feature.item())

        return torch.cat([batch.x, home_feature], dim=1)


    def train(device, model, loader, optimizer, full_info):
        model.train()

        epoch_loss = 0.0
        num_nodes = 0

        for batch in loader:
            batch = batch.to(device)
            x = extract_features(batch, full_info)

            optimizer.zero_grad()
            out = model(x, batch.edge_index)
            loss = F.binary_cross_entropy_with_logits(out, batch.y.float())
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch.num_nodes
            num_nodes += batch.num_nodes

        return epoch_loss / num_nodes


    @torch.no_grad()
    def test(device, model, loader, full_info):
        model.eval()

        epoch_loss = 0.0
        num_nodes = 0

        for batch in loader:
            batch = batch.to(device)
            x = extract_features(batch, full_info)

            out = model(x, batch.edge_index)
            loss = F.binary_cross_entropy_with_logits(out, batch.y.float())

            epoch_loss += loss.item() * batch.num_nodes
            num_nodes += batch.num_nodes

        return epoch_loss / num_nodes

    return test, train


@app.cell
def _(test, test_loader, torch, train, train_loader):
    from torch_geometric.logging import log


    def run_experiment(model, num_epochs=10, verbose=1, name=None, lr=0.001, full_info=False):
        name = name or model.__class__.__name__

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train_losses = []
        test_losses = []

        log(Model=name)

        for epoch in range(1, num_epochs + 1):
            train_loss = train(device, model, train_loader, optimizer, full_info)
            train_eval_loss = test(device, model, train_loader, full_info)
            test_loss = test(device, model, test_loader, full_info)

            train_losses.append(train_eval_loss)
            test_losses.append(test_loss)

            if verbose and (epoch - 1) % verbose == 0:
                log(Epoch=epoch, train_loss=train_loss, train_eval_loss=train_eval_loss, test_loss=test_loss)

        log(Epoch=epoch, train_loss=train_loss, train_eval_loss=train_eval_loss, test_loss=test_loss)

        return {"name": name, "epoch": range(1, num_epochs + 1), "train": train_losses, "test": test_losses}

    return (run_experiment,)


@app.cell
def _(GCN, GCNPlus, NodeMLP, dataset):
    hidden_channels = 128
    lr = 0.01
    epochs = 10
    verbose = 5

    min_gcn_layers, max_gcn_layers = (2, 2)
    gcnplus_lin_layers = 3
    mlp_layers = 3

    gcn_models = {
        f"GCN-{n}": GCN(
            num_layers=n,
            in_channels=dataset.num_features,
            hidden_channels=hidden_channels,
            out_channels=dataset.num_classes,
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
        )
        for n in range(min_gcn_layers, max_gcn_layers + 1)
    }

    mlp_models = {
        "MLP": NodeMLP(
            mlp_layers,
            in_channels=dataset.num_features,
            hidden_channels=hidden_channels,
            out_channels=dataset.num_classes,
        ),
        "MLP-Full": NodeMLP(
            mlp_layers,
            in_channels=dataset.num_features + 1,
            hidden_channels=hidden_channels,
            out_channels=dataset.num_classes,
        ),
    }

    full_info_models = {"MLP-Full"}

    models = mlp_models | gcn_models | gcn_plus_models
    list(models.keys())
    return epochs, full_info_models, hidden_channels, lr, models, verbose


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
    run_experiment,
    verbose,
):
    mo.stop(not btn_run_experiments.value)

    _results = {}

    for name, model in models.items():
        _full_info = name in full_info_models
        _results[name] = run_experiment(model, num_epochs=epochs, verbose=verbose, name=name, lr=lr, full_info=_full_info)

    results = pl.concat(pl.DataFrame(result) for result in _results.values())
    return (results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    TODO
    - Change utility to a better model
    - Add a pre- linear layer to GCN
    - Remove dropout from GCN
    - Add residual connections to GCN
    -
    """)
    return


@app.cell
def _(results):
    import altair as alt

    _scheme = "inferno"


    def _plot(title, column):
        return (
            alt
            .Chart(results)
            .mark_line(point=True)
            .encode(
                alt.X("epoch:Q").scale(domainMin=1),
                alt.Y(f"{column}:Q").scale(domainMin=0.23),
                color=alt.Color("name").scale(scheme=_scheme),
                tooltip=["name", column],
            )
            .properties(title=title, width=400, height=400)
        )


    _plot("Training loss", "train") | _plot("Test loss", "test")
    return (alt,)


@app.cell
def _(F, pl, results, sigmoid, slider_s, torch, utility, visited_nodes):
    _best_possible_theoretical_loss = F.binary_cross_entropy(
        torch.tensor(sigmoid(utility, s=slider_s.value)).float(), torch.tensor(sigmoid(utility, s=slider_s.value)).float()
    )

    _best_possible_empirical_loss = F.binary_cross_entropy(
        torch.tensor(sigmoid(utility, s=slider_s.value)).float(), torch.tensor(visited_nodes).float()
    )

    _random_guessing_loss = F.binary_cross_entropy(
        torch.full_like(torch.tensor(visited_nodes), 0.5).float(), torch.tensor(visited_nodes).float()
    )

    _theoretical_bound = pl.DataFrame({"name": "Theoretical bound", "epoch": "-", "test": _best_possible_theoretical_loss})
    _empirical_bound = pl.DataFrame({"name": "Empirical bound", "epoch": "-", "test": _best_possible_empirical_loss})
    _random_guessing_bound = pl.DataFrame({"name": "Random guessing bound", "epoch": "-", "test": _random_guessing_loss})


    _best_models = (
        results
        .group_by("name")
        .agg(pl.all().sort_by("test").first())
        .sort("test")
        .select("name", pl.col("epoch").cast(pl.String), "test")
    )

    model_comparison = pl.concat([_theoretical_bound, _empirical_bound, _best_models, _random_guessing_bound])
    model_comparison
    return (model_comparison,)


@app.cell
def _(alt, model_comparison, pl):
    _bar = (
        alt
        .Chart(model_comparison.filter(~pl.col("name").str.contains("bound")))
        .mark_bar()
        .encode(alt.X("name:N").title("Model").sort("y"), y=alt.Y("test:Q").title("Test loss"), tooltip=["name", "test"])
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

    _bar + _theo_rule + _emp_rule
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualising predictions
    """)
    return


@app.cell
def _():
    best_num_layers = 4
    return (best_num_layers,)


@app.cell(hide_code=True)
def _(mo):
    btn_train_best_model = mo.ui.run_button(label="Train best GCN model")
    btn_train_best_model
    return (btn_train_best_model,)


@app.cell
def _(
    GCN,
    best_num_layers,
    btn_train_best_model,
    dataset,
    epochs,
    hidden_channels,
    lr,
    mo,
    run_experiment,
):
    mo.stop(not btn_train_best_model.value)

    gcn = GCN(
        num_layers=best_num_layers,
        in_channels=dataset.num_features,
        hidden_channels=hidden_channels,
        out_channels=dataset.num_classes,
    )

    _ = run_experiment(gcn, num_epochs=epochs, verbose=0, lr=lr)

    gcn
    return


if __name__ == "__main__":
    app.run()
