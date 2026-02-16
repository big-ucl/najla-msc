import marimo

__generated_with = "0.19.9"
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


    SEED = 11
    return F, SEED, cm, mcolors, mo, np, nx, pl, plt, sigmoid, torch


@app.cell(hide_code=True)
def _(mo):
    slider_I = mo.ui.slider(start=0, stop=10000, step=100, value=1000, show_value=True, label="Number of individuals $I=$")
    slider_N = mo.ui.slider(steps=[i**2 for i in range(10)], value=5**2, show_value=True, label="Number of nodes $N =$")
    slider_B = mo.ui.slider(start=-20, stop=20, step=0.1, value=10, show_value=True, label="Home node penalty $\\beta=$")
    slider_s = mo.ui.slider(start=0, stop=5, step=0.1, value=1, show_value=True, label="Scale of noise distribution $s=$")

    md_utility = mo.md("Utility: $\\;\\eta_n^i = X^i X_n - \\beta X^i X^i_h$")
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
def _(I, N, SEED, np):
    _rng = np.random.default_rng(seed=SEED)

    home_locations = _rng.choice(N, I)
    node_feature = _rng.uniform(0, 1, size=(N, 1))
    indi_feature = _rng.uniform(0, 1, size=(I, 1))
    home_feature = node_feature[home_locations]
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

    utility = indi_feature @ node_feature.T - beta_home * indi_feature * home_feature
    noise = _rng.logistic(0, slider_s.value, size=(I, N))
    score = utility + noise
    visited_nodes = np.where(score >= 0, 1, 0)
    return utility, visited_nodes


@app.cell
def _(N_sqrt, SEED, node_feature, nx):
    G = nx.navigable_small_world_graph(N_sqrt, seed=SEED)
    G = nx.convert_node_labels_to_integers(G)
    nx.set_node_attributes(G, {i: f.item() for i, f in enumerate(node_feature)}, "node_feature")
    pos = nx.kamada_kawai_layout(G)
    return G, pos


@app.cell(hide_code=True)
def _(cm, mcolors, np):
    _base_cmap = cm.Greys

    norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap = mcolors.LinearSegmentedColormap.from_list("trunc_Reds", _base_cmap(np.linspace(0, 0.7, 256)))
    return cmap, norm


@app.cell(hide_code=True)
def _(G, cm, cmap, node_feature, norm, nx, plt, pos):
    fig_base, _ax = plt.subplots()

    nx.draw_networkx(G, pos=pos, node_color=node_feature, cmap=cmap, ax=_ax)

    _sm = cm.ScalarMappable(cmap=cmap, norm=norm)
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
    cmap,
    home_locations,
    node_feature,
    norm,
    np,
    nx,
    plt,
    pos,
    slider_indiv,
    visited_nodes,
):
    fig_indiv, _ax = plt.subplots()

    _i = slider_indiv.value
    _home_node = home_locations[_i].item()

    _unvisited_nodes = [i for i in np.nonzero(1 - visited_nodes[_i])[0].tolist() if i != _home_node]
    _visited_nodes = [i for i in np.nonzero(visited_nodes[_i])[0].tolist() if i != _home_node]

    nx.draw_networkx_nodes(
        G, nodelist=_unvisited_nodes, pos=pos, node_color=node_feature[_unvisited_nodes], cmap=cmap, ax=_ax, vmax=1
    )
    nx.draw_networkx_nodes(
        G,
        nodelist=[_home_node],
        pos=pos,
        node_color="LightBlue",
        edgecolors="red" if visited_nodes[_i, _home_node].item() else None,
        linewidths=1.5 if visited_nodes[_i, _home_node].item() else None,
        ax=_ax,
    )
    nx.draw_networkx_nodes(
        G,
        nodelist=_visited_nodes,
        pos=pos,
        node_color=node_feature[_visited_nodes],
        cmap=cmap,
        ax=_ax,
        vmax=1,
        edgecolors="red",
        linewidths=1.5,
    )

    nx.draw_networkx_labels(G, pos, ax=_ax)
    nx.draw_networkx_edges(G, pos=pos, ax=_ax)

    _sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    _sm.set_array(node_feature)
    _cbar = fig_indiv.colorbar(_sm, ax=_ax)
    _cbar.set_label("$X_n$")

    _ax.set_axis_off()
    _ax.set_title(f"Network for individual $i = {_i}$")

    None
    return (fig_indiv,)


@app.cell(hide_code=True)
def _(I, mo):
    slider_indiv = mo.ui.dropdown(range(I), value=0, searchable=True, label="Selected individual $i =$")
    return (slider_indiv,)


@app.cell(hide_code=True)
def _(md_noise, md_utility, mo, slider_B, slider_I, slider_N, slider_s):
    mo.vstack([slider_I, slider_N, slider_B, slider_s, md_utility, md_noise])
    return


@app.cell
def _(fig_base, fig_indiv, mo, slider_indiv):
    mo.vstack([slider_indiv, mo.hstack([fig_base, fig_indiv], justify="start")])
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
    plt.plot([0, _x.max()], [0, _x.max()], c="red")

    plt.yticks(range(0, _y.max() + 2, 2))
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
            self._visited_nodes = visited_nodes.copy()

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

            visited_nodes = torch.zeros_like(base_x)
            visited_nodes[self._visited_nodes[idx], :] = 1

            x = torch.cat([base_x, is_home, indi_node_feature], dim=1)
            y = visited_nodes

            return Data(x=x, y=y, edge_index=base_edge_index, indi_feature=indi_feature, user_id=idx)


    dataset = SyntheticDataset(base_data, indi_feature, home_locations, visited_nodes)
    dataset
    return (dataset,)


@app.cell
def _(dataset):
    data = dataset[0]
    data
    return (data,)


@app.cell
def _():
    test_size = 0.2
    batch_size = 16
    return batch_size, test_size


@app.cell
def _(SEED, batch_size, dataset, test_size):
    from torch_geometric.loader import DataLoader
    from sklearn.model_selection import train_test_split

    _train_indices, _test_indices = train_test_split(range(len(dataset)), test_size=test_size, random_state=SEED)

    train_dataset = dataset[_train_indices]
    test_dataset = dataset[_test_indices]

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return test_loader, train_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Machine learning
    """)
    return


@app.cell
def _(F, torch):
    from torch_geometric.nn import GCNConv


    class GCN(torch.nn.Module):
        def __init__(self, num_layers: int, in_channels: int, hidden_channels: int, out_channels: int):
            super().__init__()

            if num_layers < 2:
                raise ValueError(f"Minimum 2 layers, found {num_layers}")

            self.convs = torch.nn.ModuleList()

            self.convs.append(GCNConv(in_channels, hidden_channels))
            for i in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.convs.append(GCNConv(hidden_channels, out_channels))

        def forward(self, x, edge_index):
            for conv in self.convs[:-1]:
                x = F.dropout(x, p=0.5, training=self.training)
                x = conv(x, edge_index).relu()

            x = F.dropout(x, p=0.5, training=self.training)
            x = self.convs[-1](x, edge_index)

            return x


    class NodeMLP(torch.nn.Module):
        def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
            super().__init__()

            self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
            self.lin2 = torch.nn.Linear(hidden_channels, hidden_channels)
            self.lin3 = torch.nn.Linear(hidden_channels, out_channels)

        def forward(self, x, edge_index):
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.lin1(x).relu()
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.lin2(x).relu()
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.lin3(x)

            return x

    return GCN, NodeMLP


@app.cell
def _(F, data, torch):
    def compute_training_weights(loader):
        num_neg = 0
        num_pos = 0

        for batch in loader:
            num_neg += (batch.y == 0).sum()
            num_pos += batch.y.sum()
        return num_neg / num_pos


    def train(device, model, loader, optimizer):
        model.train()

        epoch_loss = 0.0
        num_nodes = 0

        pos_weight = compute_training_weights(loader)

        for batch in loader:
            batch = batch.to(device)

            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.binary_cross_entropy_with_logits(out, data.y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch.num_nodes
            num_nodes += batch.num_nodes

        return epoch_loss / num_nodes


    @torch.no_grad()
    def test(device, model, loader):
        model.eval()

        epoch_loss = 0.0
        num_nodes = 0

        for batch in loader:
            batch = batch.to(device)

            out = model(data.x, data.edge_index)
            loss = F.binary_cross_entropy_with_logits(out, data.y)

            epoch_loss += loss.item() * batch.num_nodes
            num_nodes += batch.num_nodes

        return epoch_loss / num_nodes

    return test, train


@app.cell
def _(epochs, lr, test, test_loader, torch, train, train_loader):
    from torch_geometric.logging import log


    def run_experiment(model, num_epochs=10, verbose=1, name=None):
        name = name or model.__class__.__name__

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train_losses = []
        test_losses = []

        log(Model=name)

        for epoch in range(1, epochs + 1):
            train_loss = train(device, model, train_loader, optimizer)
            test_loss = test(device, model, test_loader)

            train_losses.append(train_loss)
            test_losses.append(test_loss)

            if verbose and (epoch - 1) % verbose == 0:
                log(Epoch=epoch, Train_Loss=train_loss, Test_Loss=test_loss)

        log(Epoch=epoch, Train_Loss=train_loss, Test_Loss=test_loss)

        return {"name": name, "epoch": range(1, epochs + 1), "train": train_losses, "test": test_losses}

    return (run_experiment,)


@app.cell
def _(GCN, NodeMLP, dataset):
    hidden_channels = 16
    lr = 0.001
    epochs = 20
    verbose = 5

    max_gcn_layers = 8
    max_mlp_layers = 4

    gcn_models = {
        f"GCN-{n}": GCN(
            num_layers=n,
            in_channels=dataset.num_features,
            hidden_channels=hidden_channels,
            out_channels=dataset.num_classes,
        )
        for n in range(2, max_gcn_layers + 1)
    }

    mlp_models = {
        "MLP": NodeMLP(in_channels=dataset.num_features, hidden_channels=hidden_channels, out_channels=dataset.num_classes)
    }

    models = gcn_models | mlp_models
    list(models.keys())
    return epochs, lr, models, verbose


@app.cell(hide_code=True)
def _(mo):
    btn_run_experiments = mo.ui.run_button(label="Run model training")
    btn_run_experiments
    return (btn_run_experiments,)


@app.cell
def _(btn_run_experiments, epochs, mo, models, pl, run_experiment, verbose):
    mo.stop(not btn_run_experiments.value)

    _results = {}

    for name, model in models.items():
        _results[name] = run_experiment(model, num_epochs=epochs, verbose=verbose, name=name)

    results = pl.concat(pl.DataFrame(result) for result in _results.values())
    return (results,)


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
                alt.Y(f"{column}:Q").scale(domainMin=0.2),
                color=alt.Color("name").scale(scheme=_scheme),
                tooltip=["name", column],
            )
            .properties(title=title, width=400, height=400)
        )


    _plot("Training loss", "train") | _plot("Test loss", "test")
    return (alt,)


@app.cell
def _(np, pl, results, sigmoid, slider_s, utility, visited_nodes):
    _probs = np.log(sigmoid(utility, s=slider_s.value))
    _y = visited_nodes
    _best_possible_loss = -(_y * _probs).mean()

    _lower_bound = pl.DataFrame({"name": "Lower bound", "epoch": "-", "test": _best_possible_loss})


    _best_models = (
        results
        .group_by("name")
        .agg(pl.all().sort_by("test").first())
        .sort("test")
        .select("name", pl.col("epoch").cast(pl.String), "test")
    )

    model_comparison = pl.concat([_lower_bound, _best_models])
    model_comparison
    return (model_comparison,)


@app.cell
def _(alt, model_comparison, pl):
    _bar = (
        alt
        .Chart(model_comparison.filter(pl.col("name") != "Lower bound"))
        .mark_bar()
        .encode(alt.X("name:N").title("Model").sort("y"), y=alt.Y("test:Q").title("Test loss"), tooltip=["name", "test"])
    )

    _rule = alt.Chart(model_comparison).mark_rule(color="red").encode(y="min(test):Q", tooltip="min(test):Q")
    _bar + _rule
    return


if __name__ == "__main__":
    app.run()
