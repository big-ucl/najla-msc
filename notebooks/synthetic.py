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


    def sigmoid(z, s=1.0):
        return 1 / (1 + np.exp(-z / s))


    SEED = 11
    return SEED, cm, mcolors, mo, np, nx, plt, sigmoid


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


@app.cell
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


@app.cell
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


@app.cell
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Machine learning
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
