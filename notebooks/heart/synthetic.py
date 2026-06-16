"""
Module: notebooks/heart/synthetic.py

Description:
    A Marimo interactive notebook implementing a discrete-choice-theory simulation of
    node visit behaviour on a synthetic network graph.

    Data generation process:
      - A navigable small-world graph of N nodes is created with NetworkX.
      - For each of I individuals, a home node is chosen uniformly at random and an
        individual feature X^i is drawn from Uniform(0, 1).
      - Each node n has a feature X_n from Uniform(-1, 1).
      - An individual-specific utility is computed:
            eta_n^i = beta_1 * X^i * (X_n - mean(X_n))
                    - beta_2 * |X_n - X^i_home|
                    - beta_dist * X^i * dist(n, home_i)
                    - alpha
      - Logistic noise eps ~ Logistic(0, noise_scale) is added: score = eta + eps.
      - Node n is visited if score >= 0.

    The notebook then:
      - Converts the data to a PyTorch Geometric Dataset (SyntheticDataset).
      - Trains multiple GNN architectures (GCN, GCNPlus, GCNRes, GCNSkip, GATSkip, NodeMLP)
        and an MLP baseline with full information.
      - Compares model test losses against theoretical and empirical lower bounds.
      - Visualises predictions and samples (Poisson / PPS) on the graph.

Dependencies:
    - activitygraphs library (ml.models, ml.sampling, ml.experiment)
    - NetworkX, PyTorch Geometric, scikit-learn
    - Marimo, Polars, Altair, Matplotlib, Seaborn
"""

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")

with app.setup:
    import matplotlib.pyplot as plt
    import altair as alt

    plt.style.use("default")
    alt.theme.enable("default")


@app.cell
def _():
    """
    Description:
        Imports all libraries needed for the synthetic data generation and visualisation
        steps, sets the CUDA debugging environment variable, and defines the global
        random seed and two sigmoid helper functions.

    Output:
      - F (module): torch.nn.functional — used for BCE loss computation.
      - SEED (int): global random seed for reproducible data generation.
      - cm (module): matplotlib.cm — used for ScalarMappable colour bars.
      - mcolors (module): matplotlib.colors — used for Normalize in colour bars.
      - mo (module): marimo — used for UI widgets and markdown cells.
      - np (module): numpy — used for all data generation and array operations.
      - nx (module): networkx — used to create and query the synthetic graph.
      - pl (module): polars — used to store and write experiment results.
      - sigmoid (function): logistic sigmoid with a scale parameter (NumPy).
      - torch (module): PyTorch — used for model training and sampling.
    """
    import marimo as mo
    import polars as pl
    import numpy as np
    import networkx as nx
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    import torch
    import torch.nn.functional as F

    def sigmoid(z, s=1.0):
        """
        Description:
            Applies the standard logistic (sigmoid) function to an array, with an optional
            temperature/scale parameter that controls the steepness of the S-curve.

        Input:
          - z (np.ndarray | float): input value(s) to transform.
          - s (float): scale parameter; larger s gives a flatter, less decisive sigmoid.
                Defaults to 1.0 (standard sigmoid).

        Output:
          - (np.ndarray | float): sigmoid-transformed values in the range (0, 1).
        """
        return 1 / (1 + np.exp(-z / s))

    def torch_sigmoid(z, s=1.0):
        """
        Description:
            NumPy-based sigmoid that mirrors the signature of the torch.sigmoid function.
            Used when torch tensors are not available or when operating on NumPy arrays.

        Input:
          - z (np.ndarray | float): input value(s).
          - s (float): scale parameter.  Defaults to 1.0.

        Output:
          - (np.ndarray | float): sigmoid-transformed values.
        """
        return 1 / (1 + np.exp(-z / s))

    import os

    # Enable CUDA synchronous error reporting to get meaningful error messages from GPU ops
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    SEED = 512  # global random seed used for all numpy RNGs and graph generation
    return F, SEED, cm, mcolors, mo, np, nx, pl, sigmoid, torch


@app.cell(hide_code=True)
def _(mo):
    """
    Description:
        Creates all the interactive slider widgets that control the data generation
        process.  The sliders let the user adjust the number of individuals (I),
        nodes (N), the alternative-specific constant (alpha), and the three utility
        coefficients (beta_1, beta_2, beta_d) without re-running the whole notebook.

    Input:
      - mo (module): marimo, used to build the slider widgets.

    Output:
      - slider_I (mo.ui.slider): controls the number of synthetic individuals.
      - slider_N (mo.ui.slider): controls the number of nodes (must be a perfect square).
      - slider_asc (mo.ui.slider): controls the alternative-specific constant alpha.
      - slider_bh (mo.ui.slider): controls beta_2 (home-feature similarity coefficient).
      - slider_bn (mo.ui.slider): controls beta_1 (node-feature interaction coefficient).
      - slider_dist (mo.ui.slider): controls beta_d (distance-from-home coefficient).
    """
    slider_I = mo.ui.slider(
        start=0, stop=10000, step=100, value=2500, show_value=True, label="Number of individuals $I=$", debounce=True
    )
    slider_N = mo.ui.slider(
        steps=[i**2 for i in range(10)], value=5**2, show_value=True, label="Number of nodes $N =$", debounce=True
    )
    slider_asc = mo.ui.slider(
        start=0, stop=1, step=0.05, value=0.15, show_value=True, label="$\\alpha=$", debounce=True
    )
    slider_bn = mo.ui.slider(start=0, stop=5, step=0.05, value=2, show_value=True, label="$\\beta_1=$", debounce=True)
    slider_bh = mo.ui.slider(start=0, stop=5, step=0.05, value=0.8, show_value=True, label="$\\beta_2=$", debounce=True)
    slider_dist = mo.ui.slider(
        start=0, stop=5, step=0.05, value=0.5, show_value=True, label="$\\beta_d=$", debounce=True
    )
    return slider_I, slider_N, slider_asc, slider_bh, slider_bn, slider_dist


@app.cell
def _(asc, beta_dist, beta_home, beta_node, mo, noise_scale):
    """
    Description:
        Builds three Marimo markdown objects that display the utility formula,
        the formula with the current numeric parameter values substituted in,
        and the noise distribution specification.  These are displayed together
        in the UI cell below the sliders so the user can see the exact DGP.

    Input:
      - asc (float): the alternative-specific constant (alpha).
      - beta_dist (float): coefficient for the graph-distance term.
      - beta_home (float): coefficient for the home-feature distance term.
      - beta_node (float): coefficient for the node-feature interaction term.
      - mo (module): marimo.
      - noise_scale (float): scale of the Logistic noise distribution.

    Output:
      - md_noise (mo.md): markdown showing the noise distribution.
      - md_utility (mo.md): markdown showing the symbolic utility formula.
      - md_utility_num (mo.md): markdown showing the formula with numeric values.
    """
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
    """
    Description: Render the data generation process section heading.

    Input:
      - mo (module): marimo.

    Output:
      - (none): renders a Markdown heading in the notebook UI.
    """
    mo.md("""
    ## Data generation process
    """)
    return


@app.cell
def _(slider_I, slider_N, slider_asc, slider_bh, slider_bn, slider_dist):
    """
    Description:
        Reads the current slider values and converts them into Python scalars used
        by the data generation cells.  These variables are recomputed automatically
        whenever a slider is moved.

    Input:
      - slider_I, slider_N, slider_asc, slider_bh, slider_bn, slider_dist: Marimo sliders.

    Output:
      - I (int): number of synthetic individuals.
      - N (int): total number of nodes (= N_sqrt^2).
      - N_sqrt (int): grid side length.
      - asc (float): alternative-specific constant alpha.
      - beta_dist (float): distance-from-home utility coefficient.
      - beta_home (float): home-feature similarity utility coefficient.
      - beta_node (float): node-feature interaction utility coefficient.
      - noise_scale (float): scale of the Logistic noise added to each utility score.
    """
    N_sqrt = int(slider_N.value**0.5)  # side length of the 2-D grid (N = N_sqrt^2)
    noise_scale = 0.1                  # scale of the Logistic noise added to utility
    N = N_sqrt**2                      # total number of nodes in the graph
    I = slider_I.value                 # number of synthetic individuals to simulate
    asc = slider_asc.value             # alternative-specific constant (utility intercept)
    beta_node = slider_bn.value        # utility coefficient for node-feature term
    beta_home = slider_bh.value        # utility coefficient for home-feature distance term
    beta_dist = slider_dist.value      # utility coefficient for graph-distance term
    return I, N, N_sqrt, asc, beta_dist, beta_home, beta_node, noise_scale


@app.cell
def _(N_sqrt, nx):
    """
    Description:
        Generates the synthetic small-world graph used for the simulation.  A
        navigable small-world graph has a 2D grid structure with additional long-range
        edges, mimicking real-world city structure where nearby locations are densely
        connected but distant locations are also reachable.

    Input:
      - N_sqrt (int): the side length of the 2D grid (graph has N_sqrt^2 nodes).
      - nx (module): networkx.

    Output:
      - G (nx.DiGraph): the synthetic navigable small-world graph with integer node labels.
    """
    # seed=11 ensures the same long-range edges are added every time for reproducibility
    G = nx.navigable_small_world_graph(N_sqrt, seed=11)
    G = nx.convert_node_labels_to_integers(G)  # ensure nodes are labeled 0, 1, ..., N-1
    return (G,)


@app.cell
def _(G, np, nx):
    """
    Description:
        Computes the all-pairs shortest-path distance matrix for the synthetic graph,
        then normalises it to the range [0, 1] so it can be used as a feature in the
        utility function without dominating other terms.

    Input:
      - G (nx.DiGraph): the synthetic small-world graph.
      - np (module): numpy.
      - nx (module): networkx.

    Output:
      - distances (np.ndarray): normalised pairwise hop-distance matrix, shape (N, N).
            distances[i, j] is the shortest path in hops from node i to node j,
            divided by the maximum hop distance in the graph.
    """
    # Compute the all-pairs shortest path length (in hops) for the graph.
    # Resulting shape: (N, N), where distances[i, j] = shortest path from node i to j.
    distances = np.array([
        [x[1] for x in sorted(row[1].items(), key=lambda x: x[0])] for row in nx.all_pairs_shortest_path_length(G)
    ])
    # Normalise by the maximum distance so all values lie in [0, 1]
    distances = distances / distances.max()
    return (distances,)


@app.cell
def _(I, N, SEED, asc, beta_dist, beta_home, beta_node, distances, mo, np):
    """
    Description:
        Generates the synthetic population data.  For each of I individuals, a home
        node and a scalar individual feature are drawn at random.  Expected utility
        values are computed analytically and displayed as a Marimo markdown cell.

    Input:
      - I (int): number of individuals to simulate.
      - N (int): number of nodes in the graph.
      - SEED (int): global random seed for reproducibility.
      - asc (float): alternative-specific constant.
      - beta_dist, beta_home, beta_node (float): utility coefficients.
      - distances (np.ndarray): normalised pairwise distance matrix, shape (N, N).
      - mo (module): marimo, used to display the expected utility markdown.
      - np (module): numpy.

    Output:
      - home_feature (np.ndarray): node feature at each individual's home node, shape (I, 1).
      - home_locations (np.ndarray): home node index for each individual, shape (I,).
      - indi_feature (np.ndarray): scalar feature for each individual, shape (I, 1).
      - mean_node_feature (float): population mean of node features (used in utility formula).
      - node_feature (np.ndarray): scalar feature for each node, shape (N, 1).
    """
    # Seeded RNG for reproducible data generation
    _rng = np.random.default_rng(seed=SEED)

    # Feature ranges for node and individual attributes
    min_node_feature, max_node_feature = (-1, 1)   # node features X_n ~ Uniform(-1, 1)
    min_indi_feature, max_indi_feature = (0, 1)    # individual features X^i ~ Uniform(0, 1)

    # Assign a random home node index to each individual: shape (I,)
    home_locations = _rng.choice(N, I)

    # Node features: shape (N, 1), one scalar attribute per node
    node_feature = _rng.uniform(min_node_feature, max_node_feature, size=(N, 1))

    # Individual features: shape (I, 1), one scalar attribute per person
    indi_feature = _rng.uniform(min_indi_feature, max_indi_feature, size=(I, 1))

    # Home feature for each individual: the node feature at their home node, shape (I, 1)
    home_feature = node_feature[home_locations]

    # Population mean node feature; used in the utility formula as the reference point
    mean_node_feature = node_feature.mean()

    # Expected values of each feature used to calculate expected utility analytically
    _exp_node_feature = min_node_feature + (max_node_feature - min_node_feature) / 2
    _exp_indi_feature = min_indi_feature + (max_indi_feature - min_indi_feature) / 2
    _exp_home_feature = _exp_node_feature  # same range as node features

    # Average normalised distance across all pairs of nodes
    _exp_dist_feature = distances.mean()

    # Expected utility E[eta_n^i] evaluated at the expected values of all inputs
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
    """
    Description:
        Attaches the generated node features to the NetworkX graph object so they are
        accessible via G.nodes[i]["node_feature"], and computes a Kamada-Kawai layout
        (spring-like, aesthetically pleasant for small-world graphs) for visualisation.

    Input:
      - G (nx.DiGraph): the synthetic small-world graph.
      - node_feature (np.ndarray): per-node feature values, shape (N, 1).
      - nx (module): networkx.

    Output:
      - pos (dict): node -> (x, y) layout coordinates used by all nx.draw_* calls below.
    """
    nx.set_node_attributes(G, {i: f.item() for i, f in enumerate(node_feature)}, "node_feature")
    # Kamada-Kawai positions nodes to minimise a spring-energy objective; looks good for small graphs
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
    """
    Description:
        Computes the individual-specific utilities for all (individual, node) pairs,
        adds logistic noise, and derives the binary visit matrix.

        The utility formula is:
          eta[i, n] = beta_1 * X^i * (X_n - mean(X_n))
                    - beta_2 * |X_n - X^i_home|
                    - beta_dist * X^i * dist(n, home_i)
                    - alpha

        A node is visited if: eta[i, n] + epsilon[i, n] >= 0,
        where epsilon[i, n] ~ Logistic(0, noise_scale).

    Input:
      - I (int): number of individuals.
      - N (int): number of nodes.
      - SEED (int): random seed for the noise RNG.
      - asc, beta_dist, beta_home, beta_node (float): utility parameters.
      - distances (np.ndarray): normalised distance matrix, shape (N, N).
      - home_feature (np.ndarray): home node feature for each individual, shape (I, 1).
      - home_locations (np.ndarray): home node index for each individual, shape (I,).
      - indi_feature (np.ndarray): individual-level features, shape (I, 1).
      - mean_node_feature (float): population mean node feature (used in utility).
      - node_feature (np.ndarray): per-node features, shape (N, 1).
      - noise_scale (float): scale of the Logistic noise.
      - np (module): numpy.

    Output:
      - noise (np.ndarray): logistic noise draws, shape (I, N).
      - score (np.ndarray): total score = utility + noise, shape (I, N).
      - utility (np.ndarray): deterministic utility values, shape (I, N).
      - visited_nodes (np.ndarray): binary visit indicators (1=visited), shape (I, N).
    """
    _rng = np.random.default_rng(seed=SEED)

    # Compute the deterministic utility matrix: shape (I, N)
    # Each element eta[i, n] represents the utility of individual i visiting node n.
    utility = (
        beta_node * indi_feature * (node_feature.T - mean_node_feature)  # node-feature interaction
        - beta_home * np.abs(node_feature.T - home_feature)               # home-feature similarity
        - beta_dist * indi_feature * distances[home_locations, :]         # distance from home
        - asc                                                              # constant disutility
    )

    # Logistic noise: shape (I, N).  The scale matches the choice of noise_scale slider.
    noise = _rng.logistic(0, noise_scale, size=(I, N))

    # Total score = utility + noise.  Individual i visits node n if score[i, n] >= 0.
    score = utility + noise

    # Binary visit matrix: shape (I, N).  1 = visited, 0 = not visited.
    visited_nodes = np.where(score >= 0, 1, 0)
    return noise, score, utility, visited_nodes


@app.cell
def _(sns):
    """
    Description:
        Defines shared visualisation constants for the graph plots below.
        The diverging colour map (vlag_r) maps negative utilities to blue and
        positive utilities to red, making the sign of the utility visually obvious.

    Input:
      - sns (module): seaborn, used to obtain the colour map.

    Output:
      - cmap (matplotlib.colors.Colormap): a reversed "vlag" diverging colour map
            used to colour nodes by utility or visit probability.
    """
    cmap = sns.color_palette("vlag_r", as_cmap=True)  # diverging blue-white-red
    best_i = 26  # example individual index for static reference plots (not exported)
    return (cmap,)


@app.cell
def _(
    G,
    cm,
    cmap,
    figsize,
    home_locations,
    mcolors,
    nx,
    pos,
    slider_indiv,
    tab10,
    utility,
):
    """
    Description:
        Draws the network graph coloured by the deterministic utility values for the
        selected individual.  Non-home nodes are coloured on the diverging vlag_r scale
        (blue = negative utility, red = positive utility) and the home node is drawn
        as an orange square.  Saves the figure as fig_indiv_util for later export.

    Input:
      - G (nx.DiGraph): the synthetic graph.
      - cm (module): matplotlib.cm for ScalarMappable colour bar.
      - cmap: the diverging colour map.
      - figsize (tuple): figure size in inches.
      - home_locations (np.ndarray): home node index for each individual.
      - mcolors (module): matplotlib.colors for Normalize.
      - nx (module): networkx for drawing.
      - pos (dict): node layout positions.
      - slider_indiv (mo.ui.dropdown): currently selected individual index.
      - tab10 (list): Seaborn tab10 colour palette list.
      - utility (np.ndarray): deterministic utility matrix, shape (I, N).

    Output:
      - fig_indiv_util (matplotlib.figure.Figure): the utility map figure.
    """
    fig_indiv_util, _ax = plt.subplots(figsize=figsize, constrained_layout=True)

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
        node_color=tab10[1],
        node_shape="s",
        # edgecolors="green" if visited_nodes[_i, _home_node].item() else None,
        # linewidths=1.5 if visited_nodes[_i, _home_node].item() else None,
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
    _cbar = fig_indiv_util.colorbar(_sm, ax=_ax, location="bottom", shrink=0.8)
    _cbar.set_label("Individual-specific node utility $\\eta_{{i, n}}$")

    _ax.set_axis_off()
    _ax.set_title("Network graph with node utilities for example individual $i$")

    None
    return (fig_indiv_util,)


@app.cell
def _(
    G,
    cm,
    cmap,
    fig_indiv_util,
    figsize,
    home_locations,
    mcolors,
    noise,
    nx,
    pos,
    slider_indiv,
    tab10,
    utility,
):
    """
    Description:
        Draws the network graph coloured by the additive Logistic noise values for the
        selected individual.  Illustrates how the noise perturbs the utility scores,
        causing some high-utility nodes to be unvisited and vice versa.

    Input:
      - G (nx.DiGraph): the synthetic graph.
      - cm (module): matplotlib.cm for ScalarMappable.
      - cmap: the diverging colour map.
      - fig_indiv_util (Figure): the utility figure (used to attach the colour bar).
      - figsize (tuple): figure size in inches.
      - home_locations (np.ndarray): home node indices.
      - mcolors (module): matplotlib.colors for Normalize.
      - noise (np.ndarray): logistic noise draws, shape (I, N).
      - nx (module): networkx.
      - pos (dict): node layout positions.
      - slider_indiv (mo.ui.dropdown): currently selected individual.
      - tab10 (list): Seaborn tab10 colour palette.
      - utility (np.ndarray): utility matrix (only used for ScalarMappable range).

    Output:
      - fig_noise (matplotlib.figure.Figure): the noise map figure.
    """
    fig_noise, _ax = plt.subplots(figsize=figsize, constrained_layout=True)

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
        node_color=tab10[1],
        node_shape="s",
        # edgecolors="green" if visited_nodes[_i, _home_node].item() else None,
        # linewidths=1.5 if visited_nodes[_i, _home_node].item() else None,
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
    _cbar = fig_indiv_util.colorbar(_sm, ax=_ax, location="bottom", shrink=0.8)
    _cbar.set_label("Noise $\\varepsilon_{i, n} \\sim \\text{Gumbel}(0,0.1)$")

    _ax.set_axis_off()
    _ax.set_title("Additive noise for example individual $i$")

    None
    return (fig_noise,)


@app.cell
def _(
    G,
    cm,
    cmap,
    figsize,
    home_locations,
    mcolors,
    np,
    nx,
    pos,
    score,
    slider_indiv,
    tab10,
    utility,
):
    """
    Description:
        Draws the network graph with nodes coloured categorically to show which nodes
        were visited by the selected individual (score >= 0) and which were not.
        Visited nodes are shown in the first tab10 colour, unvisited in white with a
        light-grey border, and the home node as an orange square.

    Input:
      - G (nx.DiGraph): the synthetic graph.
      - cm, cmap, mcolors: colour map and normalisation helpers (for invisible colour bar).
      - figsize (tuple): figure size.
      - home_locations (np.ndarray): home node indices.
      - np (module): numpy.
      - nx (module): networkx.
      - pos (dict): node layout positions.
      - score (np.ndarray): utility + noise, shape (I, N); visit if >= 0.
      - slider_indiv (mo.ui.dropdown): currently selected individual.
      - tab10 (list): Seaborn tab10 palette.
      - utility (np.ndarray): utility matrix (only used for colour-bar scaling).

    Output:
      - fig_visit (matplotlib.figure.Figure): the visit pattern figure.
    """
    fig_visit, _ax = plt.subplots(figsize=figsize, constrained_layout=True)

    _i = slider_indiv.value
    _home_node = home_locations[_i].item()

    _score_nodes = np.where(score[_i] >= 0, 1, 0)

    _unvisited_nodes = [i for i in np.nonzero(1 - _score_nodes)[0].tolist() if i != _home_node]
    _visited_nodes = [i for i in np.nonzero(_score_nodes)[0].tolist() if i != _home_node]

    _cmap = cmap

    nx.draw_networkx_nodes(G, nodelist=_unvisited_nodes, pos=pos, node_color="white", ax=_ax, edgecolors="lightgray")

    nx.draw_networkx_nodes(
        G,
        nodelist=_visited_nodes,
        pos=pos,
        node_color=tab10[0],
        ax=_ax,
    )

    nx.draw_networkx_nodes(
        G,
        nodelist=[_home_node],
        pos=pos,
        node_color=tab10[1],
        node_shape="s",
        ax=_ax,
    )

    _light_nodes = {n: n for n in _visited_nodes + [_home_node]}
    _dark_nodes = {n: n for n in _unvisited_nodes}

    nx.draw_networkx_labels(G, pos, labels=_light_nodes, font_color="white")
    nx.draw_networkx_labels(G, pos, labels=_dark_nodes, font_color="black")

    nx.draw_networkx_edges(G, pos=pos, ax=_ax)

    _sm = cm.ScalarMappable(cmap=_cmap, norm=mcolors.Normalize(vmin=-1, vmax=1))
    _sm.set_array(utility[_i])
    _cbar = fig_visit.colorbar(_sm, ax=_ax, location="bottom", shrink=0.8)

    _cbar.solids.set_alpha(0)
    _cbar.outline.set_alpha(0)
    _cbar.ax.tick_params(labelcolor="none", color="none")
    _cbar.set_label("")

    _ax.set_axis_off()
    _ax.set_title("Nodes visited by example individual $i$")

    None
    return (fig_visit,)


@app.cell(hide_code=True)
def _(I, mo):
    """
    Description:
        Creates a searchable dropdown widget that allows the user to select which
        synthetic individual to visualise in the graph panels above.

    Input:
      - I (int): total number of individuals; determines the dropdown range.
      - mo (module): marimo.

    Output:
      - slider_indiv (mo.ui.dropdown): individual selector widget, range 0..I-1.
    """
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
    """
    Description:
        Lays out all the data-generation parameter sliders and the three utility formula
        markdown blocks in a vertical stack for display at the top of the notebook section.

    Input:
      - md_noise, md_utility, md_utility_num (mo.md): formula markdown objects.
      - mo (module): marimo.
      - slider_I, slider_N, slider_asc, slider_bh, slider_bn, slider_dist: all sliders.

    Output:
      - (displayed in notebook, nothing exported)
    """
    mo.vstack([slider_I, slider_N, slider_asc, slider_bn, slider_bh, slider_dist, md_utility, md_utility_num, md_noise])
    return


@app.cell
def _(fig_indiv_util, fig_noise, fig_poisson, fig_pps, fig_preds, fig_visit):
    """
    Description:
        Saves all six synthetic experiment figures to PNG files in the reports/figures
        directory at 200 DPI for inclusion in the thesis.  Each figure is named after
        the aspect it illustrates (utility, noise, visited nodes, predictions, samples).

    Input:
      - fig_indiv_util (Figure): individual utility map.
      - fig_noise (Figure): noise map.
      - fig_poisson (Figure): Poisson-sampled choice set.
      - fig_pps (Figure): PPS-sampled choice set.
      - fig_preds (Figure): GATSkip predicted probabilities.
      - fig_visit (Figure): ground-truth visited nodes.

    Output:
      - (six PNG files written to reports/figures/; nothing returned)
    """
    from pathlib import Path

    _path = Path("reports/figures")  # output directory for figures
    _dpi = 200                       # resolution for print-quality PNG export

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
    """
    Description:
        Main display cell for the data generation section.  Shows the individual
        selector dropdown, the numeric feature values for the selected individual,
        and the three side-by-side plots: utility map, noise map, and visit map.

    Input:
      - fig_indiv_util, fig_noise, fig_visit (Figure): the three graph visualisation figures.
      - home_locations (np.ndarray): home node index per individual.
      - indi_feature (np.ndarray): scalar feature per individual, shape (I, 1).
      - mo (module): marimo.
      - node_feature (np.ndarray): scalar feature per node, shape (N, 1).
      - slider_indiv (mo.ui.dropdown): the currently selected individual.

    Output:
      - (displayed in notebook, nothing exported)
    """
    _x_i = indi_feature[slider_indiv.value].item()  # individual feature value for display
    _x_h = node_feature[home_locations[slider_indiv.value]].item()  # home node feature for display

    mo.vstack([
        slider_indiv,
        mo.md(f"Value of individual feature: $X^i={_x_i:.4f}$"),
        mo.md(f"Value of home node feature: $X^i_h={_x_h:.4f}$"),
        mo.hstack([fig_indiv_util, fig_noise, fig_visit], justify="start"),
    ])
    return


@app.cell
def _(fig_poisson, fig_pps, fig_preds, mo):
    """
    Description:
        Display cell — arranges the three sampling/prediction figures side-by-side
        in the notebook: predicted probabilities, Poisson sample, and PPS sample.

    Input:
      - fig_poisson (Figure): Poisson-sampled location choice set.
      - fig_pps (Figure): PPS-sampled location choice set.
      - fig_preds (Figure): GATSkip predicted visit probabilities.
      - mo (module): marimo, used for layout.

    Output:
      - (displayed in notebook, nothing exported)
    """
    (mo.hstack([fig_preds, fig_poisson, fig_pps], justify="start"),)
    return


@app.cell(hide_code=True)
def _(mo, visited_nodes):
    """
    Description:
        Computes and displays the mean number of nodes visited per individual
        as a Marimo markdown cell.  Used as a quick sanity check on the DGP.

    Input:
      - mo (module): marimo.
      - visited_nodes (np.ndarray): binary visit matrix, shape (I, N).

    Output:
      - (displayed as markdown, nothing exported)
    """
    # Sum across nodes for each individual then average across individuals
    num_visits = visited_nodes.sum(axis=1).mean()
    mo.md(f"Mean number of visits per individual: $\\sum_{{i = 1}}^{{I}}N^i_v = {num_visits:.2f}$")
    return


@app.cell(hide_code=True)
def _(mo, noise_scale, sigmoid, utility):
    """
    Description:
        Computes and displays the analytically-expected mean number of visits per
        individual: E[N_v^i] = sum_n sigmoid(eta_n^i / noise_scale) averaged over I.
        Compare with the empirical mean from visited_nodes to verify the DGP.

    Input:
      - mo (module): marimo.
      - noise_scale (float): logistic noise scale used in the DGP.
      - sigmoid (function): scaled sigmoid helper.
      - utility (np.ndarray): deterministic utility matrix, shape (I, N).

    Output:
      - (displayed as markdown, nothing exported)
    """
    # sigmoid(eta / noise_scale) gives the probability that eta + eps >= 0 for Logistic eps
    expected_num_visits = sigmoid(utility, s=noise_scale).sum(axis=1).mean()
    mo.md(f"Expected number of visits per individual: $\\mathbb{{E}}[N^i_v] = {expected_num_visits:.2f}$")
    return


@app.cell
def _(noise_scale, sigmoid, utility, visited_nodes):
    """
    Description:
        Produces a scatter plot comparing the expected number of visits per individual
        (x-axis) with the actual number of visits (y-axis).  A red identity line is
        overlaid; points close to the line indicate the DGP behaves as intended.

    Input:
      - noise_scale (float): logistic noise scale.
      - sigmoid (function): scaled sigmoid helper.
      - utility (np.ndarray): utility matrix, shape (I, N).
      - visited_nodes (np.ndarray): binary visit matrix, shape (I, N).

    Output:
      - (matplotlib scatter plot, displayed in notebook; nothing exported)
    """
    _x = sigmoid(utility, s=noise_scale).sum(axis=1)  # expected visit count per individual
    _y = visited_nodes.sum(axis=1)                     # actual visit count per individual

    plt.scatter(_x, _y, s=5)
    plt.plot([_x.min(), _x.max()], [_x.min(), _x.max()], c="red")  # identity line

    plt.yticks(range(_y.min(), _y.max() + 2, 2))
    plt.xlabel("Expected number of visits")
    plt.ylabel("Actual number of visits")
    return


@app.cell(hide_code=True)
def _(mo):
    """
    Description: Render the PyG dataset generation section heading.

    Input:
      - mo (module): marimo.

    Output:
      - (none): renders a Markdown heading in the notebook UI.
    """
    mo.md(r"""
    ## PyG Dataset generation
    """)
    return


@app.cell
def _(G, distances, home_locations, indi_feature, np, torch, visited_nodes):
    """
    Description:
        Converts the raw synthetic arrays into a PyTorch Geometric dataset by:
          1. Converting the NetworkX graph to a base PyG Data object (shared topology).
          2. Defining SyntheticDataset, which wraps the simulation arrays and builds
             individual-specific PyG Data objects on demand via the get() method.
          3. Instantiating the dataset with the generated arrays.

    Input:
      - G (nx.DiGraph): the synthetic small-world graph.
      - distances (np.ndarray): normalised pairwise distance matrix, shape (N, N).
      - home_locations (np.ndarray): home node indices, shape (I,).
      - indi_feature (np.ndarray): individual features, shape (I, 1).
      - np (module): numpy.
      - torch (module): PyTorch.
      - visited_nodes (np.ndarray): binary visit matrix, shape (I, N).

    Output:
      - dataset (SyntheticDataset): the full dataset with I individuals, displayed.
    """
    from torch_geometric.data import Data, InMemoryDataset
    from torch_geometric.utils import from_networkx

    # Convert the NetworkX graph to a PyG base object (edge_index + node_feature)
    base_data = from_networkx(G, group_node_attrs=["node_feature"])

    class SyntheticDataset(InMemoryDataset):
        """
        Description:
            A PyTorch Geometric InMemoryDataset that wraps the synthetic discrete-choice
            simulation data.  Each item in the dataset represents one individual and contains
            the network graph with per-node features and binary visit labels.

            Node features per individual:
              col 0   : raw node feature X_n (same for all individuals)
              col 1   : is_home indicator (1 for the home node, 0 elsewhere)
              col 2   : individual feature X^i broadcast to every node

        Input (constructor):
          - base_data (Data): the shared base PyG graph (edge structure + node features).
          - indi_feature (np.ndarray): individual features, shape (I, 1).
          - home_locations (np.ndarray): home node index for each individual, shape (I,).
          - visited_nodes (np.ndarray): binary visit matrix, shape (I, N).
          - distances (np.ndarray): normalised pairwise distance matrix, shape (N, N).
        """
        def __init__(
            self,
            base_data: Data,
            indi_feature: np.ndarray,
            home_locations: np.ndarray,
            visited_nodes: np.ndarray,
            distances: np.ndarray,
        ):
            """
            Description:
                Initialises the dataset by storing all pre-computed tensors needed
                for training. Clones and copies every array so the dataset is
                independent of the caller's memory.

            Input:
              - base_data (torch_geometric.data.Data): the shared graph structure
                    (node features X_n, edge indices, edge weights) used by every
                    individual.
              - indi_feature (np.ndarray): individual-level scalar features, shape
                    (I, 1), where I is the number of individuals in this split.
              - home_locations (np.ndarray): integer node indices indicating each
                    individual's home node, shape (I,).
              - visited_nodes (np.ndarray): binary matrix indicating which nodes
                    each individual visited, shape (I, N) where N is the graph size.
              - distances (np.ndarray): normalised pairwise node distance matrix,
                    shape (N, N), used as an additional edge/node feature.

            Output:
              - (None): constructor — sets instance attributes only.
            """
            super().__init__()

            self._base_data = base_data.clone()                      # shared graph topology + X_n
            self._indi_feature = indi_feature.copy()                 # individual features, shape (I, 1)
            self._home_locations = home_locations.copy()             # home node indices, shape (I,)
            self._visited_nodes = torch.tensor(visited_nodes, dtype=int)  # binary labels, shape (I, N)
            self._distances = torch.tensor(distances)                # normalised distance matrix, shape (N, N)

        @property
        def num_classes(self):
            """
            Description: Returns the number of output classes per node (always 1 for binary prediction).
            Output: (int): 1.
            """
            return 1

        def len(self) -> int:
            """
            Description: Returns the total number of individuals (samples) in the dataset.
            Output: (int): number of individuals I.
            """
            return self._indi_feature.shape[0]

        def get(self, idx: int) -> Data:
            """
            Description:
                Builds and returns the PyG Data object for individual idx by combining the
                shared base graph features with the individual-specific home indicator and
                individual feature value.

            Input:
              - idx (int): index of the individual to retrieve.

            Output:
              - (Data): a PyG Data object with:
                  x (Tensor): node features of shape (N, 3): [X_n, is_home, X^i]
                  y (Tensor): binary visit labels of shape (N, 1)
                  edge_index (Tensor): the graph edge indices
                  indi_feature (float): the scalar individual feature value
                  home_feature (Tensor): the node feature at the home node, shape (1, 1)
                  user_id (int): the individual index (used as user identifier)
                  distances (Tensor): distances from home node to all other nodes, shape (N, 1)
            """
            base_x = self._base_data.x              # node features X_n, shape (N, 1)
            base_edge_index = self._base_data.edge_index  # edge connectivity

            # is_home: one-hot indicator for the home node; same shape as base_x
            is_home = torch.zeros_like(base_x, dtype=int)
            is_home[self._home_locations[idx]] = 1   # set 1 at the home node

            # Broadcast the scalar individual feature to all nodes (so each node sees X^i)
            indi_feature = self._indi_feature[idx].item()
            indi_node_feature = torch.full_like(base_x, indi_feature)

            # The node feature at the home location (used as context)
            home_feature = base_x[self._home_locations[idx]]
            # Distances from the home node to every other node, shape (N, 1)
            distances = self._distances[self._home_locations[idx]].unsqueeze(1).float()

            # Concatenate all per-node features: [X_n, is_home, X^i], shape (N, 3)
            x = torch.cat([base_x, is_home, indi_node_feature], dim=1)
            # Binary visit labels, shape (N, 1)
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
    """
    Description:
        Defines the dataset splitting and batching hyperparameters shared between
        the train/test split cell and the DataLoader creation cell.

    Output:
      - batch_size (int): number of graphs per training mini-batch.
      - test_size (float): fraction of individuals held out for evaluation.
    """
    test_size = 0.2   # 20% of individuals reserved for testing
    batch_size = 128  # number of individuals processed per training step
    return batch_size, test_size


@app.cell
def _(batch_size, dataset, test_size):
    """
    Description:
        Splits the SyntheticDataset into train and test subsets and wraps each in a
        PyG DataLoader.  The training loader shuffles data each epoch; the test loader
        does not shuffle to ensure reproducible evaluation.

    Input:
      - batch_size (int): number of graphs per mini-batch.
      - dataset (SyntheticDataset): the full synthetic dataset.
      - test_size (float): fraction to hold out for testing.

    Output:
      - DataLoader (class): exported so downstream cells can build new loaders.
      - test_loader (DataLoader): batched loader for the test split.
      - train_loader (DataLoader): batched, shuffled loader for the training split.
    """
    from torch_geometric.loader import DataLoader
    from sklearn.model_selection import train_test_split

    # Randomly split individual indices into train and test groups
    _train_indices, _test_indices = train_test_split(range(len(dataset)), test_size=test_size)

    train_dataset = dataset[_train_indices]   # subset of individuals for training
    test_dataset = dataset[_test_indices]     # subset of individuals for evaluation

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return DataLoader, test_loader, train_loader


@app.cell(hide_code=True)
def _(mo):
    """
    Description: Render the section heading for the GCN architecture comparison experiment.

    Input:
      - mo (module): marimo.

    Output:
      - (none): renders a Markdown heading in the notebook UI.
    """
    mo.md(r"""
    ## Finding a good GCN for this task
    """)
    return


@app.cell
def _(dataset):
    """
    Description:
        Instantiates all model variants for the GCN architecture comparison experiment.
        Five families of models are built (GCN, GCNPlus, GCNRes, GCNSkip, GATSkip) across
        2–5 layers, each with and without residuals where applicable, plus two MLP baselines
        (MLP without home/distance features, MLP-Full with full information).

        The resulting dict maps model names (e.g. "GCN-3", "GATSkip-5-res") to nn.Module
        instances ready for training.

    Input:
      - dataset (SyntheticDataset): used to infer num_features and num_classes.

    Output:
      - GATSkip (class): exported so the best-model cell can instantiate it directly.
      - dropout (float): dropout rate (0.0 for this synthetic experiment).
      - epochs (int): number of training epochs per model.
      - full_info_models (set[str]): model names that receive the full-information input.
      - gcnplus_lin_layers (int): number of post-GCN linear layers.
      - hidden_channels (int): width of hidden layers.
      - lr (float): initial learning rate for Adam optimiser.
      - models (dict[str, nn.Module]): all model variants to compare.
      - verbose (int): print training progress every this many epochs.
    """
    from ml.models import NodeMLP, GCN, GCNPlus, GCNRes, GCNSkip, GATSkip

    hidden_channels = 64  # number of hidden units per layer — kept small for the synthetic task
    lr = 0.01             # Adam learning rate
    dropout = 0.0         # no dropout for the synthetic experiment (small dataset)
    epochs = 50           # training epochs per model
    verbose = 5           # print loss every 5 epochs

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
    """
    Description:
        Renders a user-triggered "Run model training" button.  The expensive training
        loop below is gated on this button so it does not run automatically when the
        notebook is opened or when sliders are adjusted.

    Input:
      - mo (module): marimo.

    Output:
      - btn_run_experiments (mo.ui.run_button): the trigger button widget.
    """
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
    """
    Description:
        Runs the full model comparison experiment — trains every model in `models` for
        the specified number of epochs and collects per-epoch train/test loss.
        Only executes when the user presses the "Run model training" button.
        Results are concatenated into a Polars DataFrame and saved to Parquet.

    Input:
      - btn_run_experiments (mo.ui.run_button): gating button; cell stops if not pressed.
      - epochs, lr, verbose: training hyperparameters.
      - full_info_models (set[str]): model names that receive full-information features.
      - mo (module): marimo (for mo.stop).
      - models (dict[str, nn.Module]): all model variants to train.
      - pl (module): polars, used to concatenate and write results.
      - test_loader, train_loader (DataLoader): data for training and evaluation.

    Output:
      - run_experiment (function): exported so the best-model cell can reuse it.
      - (results Parquet file written to reports/data/synthetic-results.parquet)
    """
    from ml.experiment import run_experiment

    mo.stop(not btn_run_experiments.value)

    _results = {}

    for name, model in models.items():
        _full_info = name in full_info_models   # MLP-Full gets home/distance features
        _results[name] = run_experiment(
            model, train_loader, test_loader, num_epochs=epochs, verbose=verbose, name=name, lr=lr, full_info=_full_info
        )

    # Concatenate all per-model result DataFrames and persist to disk
    results = pl.concat(pl.DataFrame(result) for result in _results.values())
    results.write_parquet("reports/data/synthetic-results.parquet")
    return (run_experiment,)


@app.cell(hide_code=True)
def _(pl):
    """
    Description:
        Loads the saved experiment results Parquet file and adds a "type" column
        that extracts the model family prefix (e.g. "GCN", "GATSkip", "MLP") from
        the model name.  Used for colour-coding in the comparison charts below.

    Input:
      - pl (module): polars.

    Output:
      - results_aug (pl.DataFrame): the results table with an added "type" column.
    """
    _results = pl.read_parquet("reports/data/synthetic-results.parquet")
    # _results = results  # uncomment to use freshly-trained results instead

    # Extract the model family from the name (e.g. "GATSkip-5-res" -> "GATSkip")
    results_aug = _results.with_columns(type=pl.col("name").str.split("-").list.first())
    results_aug
    return (results_aug,)


@app.cell(hide_code=True)
def _(results_aug):
    """
    Description:
        Defines the plot_results helper function and immediately uses it to render
        a side-by-side Altair chart comparing training and test loss for all models.

    Input:
      - results_aug (pl.DataFrame): results table with "type" column for colour coding.

    Output:
      - plot_results (function): reusable Altair chart builder exported to downstream cells.
      - (side-by-side Altair chart displayed in notebook)
    """
    def plot_results(results, title, column, color="name:N", detail=None, scheme="inferno"):
        """
        Description:
            Builds a line chart (with point markers) showing a training metric over epochs
            for all models.  Supports optional detail (for multi-line grouping) and colour
            scheme customisation.

        Input:
          - results (pl.DataFrame): filtered results DataFrame.
          - title (str): chart title.
          - column (str): the metric column to plot on the y-axis (e.g. "train", "test").
          - color (str): Altair colour encoding specification.  Defaults to "name:N".
          - detail (str | None): Altair detail field for grouping multiple lines.
          - scheme (str | None): Altair colour scheme name; None uses default colours.

        Output:
          - (alt.Chart): a single Altair line chart.
        """
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
    """
    Description:
        Selects the single best-performing model from each family (by minimum test loss)
        plus the MLP baseline, then plots their training and test loss curves side-by-side.
        This gives a clean "one winner per family" comparison without the clutter of all variants.

    Input:
      - pl (module): polars.
      - plot_results (function): Altair chart builder.
      - results_aug (pl.DataFrame): augmented results with "type" column.

    Output:
      - (Altair side-by-side chart, displayed in notebook; nothing exported)
    """
    # For each model family, keep only the row with the lowest test loss
    _best_performers = pl.concat([
        results_aug.group_by("type").agg(pl.all().sort_by("test").first()).select("name"),
        pl.DataFrame({"name": "MLP"}),  # always include the MLP baseline
    ])
    _results = results_aug.join(_best_performers, on="name")

    plot_results(_results, "Training loss", "train") | plot_results(_results, "Test loss", "test")
    return


@app.cell(hide_code=True)
def _(mo, results_aug):
    """
    Description:
        Creates a dropdown widget to filter the detailed layer-comparison chart to a single
        model family (e.g. show only GATSkip variants across 2–5 layers).

    Input:
      - mo (module): marimo.
      - results_aug (pl.DataFrame): used to get the list of unique model family types.

    Output:
      - dropdown_type (mo.ui.dropdown): model-family filter widget.
    """
    dropdown_type = mo.ui.dropdown(results_aug["type"].unique(), label="Choose type of GCN model:")
    return (dropdown_type,)


@app.cell(hide_code=True)
def _(dropdown_type, mo, pl, plot_results, results_aug):
    """
    Description:
        Filters the results to the selected model family (plus MLP as a baseline),
        extracts the number of layers from the model name as a colour dimension, and
        renders side-by-side train/test loss charts with the family selector dropdown.

    Input:
      - dropdown_type (mo.ui.dropdown): selected model family filter.
      - mo (module): marimo.
      - pl (module): polars.
      - plot_results (function): Altair chart builder.
      - results_aug (pl.DataFrame): augmented results table.

    Output:
      - (Altair chart + dropdown displayed in notebook; nothing exported)
    """
    # Extract the layer count from names like "GCN-3" -> "3"; MLP gets "MLP" as label
    _results = results_aug.with_columns(layers=pl.col("name").str.extract(r"(\d)").fill_null("MLP"))

    if dropdown_type.value:
        # Keep only the selected family and the MLP baseline for comparison
        _results = _results.filter((pl.col("type") == dropdown_type.value) | (pl.col("type") == "MLP"))

    _fig = plot_results(_results, "Training loss", "train", color="layers", detail="name", scheme=None) | plot_results(
        _results, "Test loss", "test", color="layers", detail="name"
    )

    mo.vstack([dropdown_type, _fig])
    return


@app.cell(hide_code=True)
def _(F, noise_scale, pl, results_aug, sigmoid, torch, utility, visited_nodes):
    """
    Description:
        Computes three benchmark loss values to contextualise model performance:
          1. Theoretical bound: BCE using the true sigmoid probabilities as both
             predictions and targets — the minimum achievable loss given the DGP.
          2. Empirical bound: BCE using the true probabilities as predictions but
             binary realisations as targets — the Bayes-optimal achievable loss.
          3. Random guessing: BCE using p = 0.5 for all nodes as predictions.

        These bounds are combined with the best test loss for each trained model to
        produce the model comparison bar chart.

    Input:
      - F (module): torch.nn.functional.
      - noise_scale (float): logistic noise scale used in the DGP.
      - pl (module): polars.
      - results_aug (pl.DataFrame): augmented training results.
      - sigmoid (function): scaled sigmoid helper.
      - torch (module): PyTorch.
      - utility (np.ndarray): deterministic utility matrix.
      - visited_nodes (np.ndarray): binary visit matrix.

    Output:
      - model_comparison (pl.DataFrame): a table combining the three bounds and the
            best test loss per trained model for the bar chart below.
    """
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
    """
    Description:
        Produces an Altair bar chart comparing the best test loss for each model,
        annotated with horizontal rule lines for the theoretical bound (green),
        empirical bound (red), and MLP baseline (red).  Models are sorted by test loss.

    Input:
      - model_comparison (pl.DataFrame): combined bounds + best model losses table.
      - pl (module): polars, used to filter the DataFrame.

    Output:
      - (Altair layered chart, displayed in notebook; nothing exported)
    """
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
    """
    Description: Render the architecture comparison conclusions as a bulleted Markdown list.

    Input:
      - mo (module): marimo.

    Output:
      - (none): renders the conclusion bullets in the notebook UI.
    """
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
    """
    Description: Render the prediction visualisation section heading.

    Input:
      - mo (module): marimo.

    Output:
      - (none): renders a Markdown heading in the notebook UI.
    """
    mo.md(r"""
    ## Visualising predictions
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    """
    Description:
        Renders a user-triggered "Train best GCN model" button.  The retraining cell
        below is gated on this button so it does not run automatically.

    Input:
      - mo (module): marimo.

    Output:
      - btn_train_best_model (mo.ui.run_button): the trigger button widget.
    """
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
    """
    Description:
        Trains the best-performing GATSkip model variant (5 layers, with residuals)
        from scratch for 50 epochs.  Only executes when the user presses the
        "Train best GCN model" button.  The trained model is exported for the
        prediction visualisation cells below.

    Input:
      - GATSkip (class): the GATSkip model class.
      - btn_train_best_model (mo.ui.run_button): gating button.
      - dataset (SyntheticDataset): used to infer in_channels and num_classes.
      - dropout, gcnplus_lin_layers, hidden_channels, lr: hyperparameters.
      - mo (module): marimo (for mo.stop).
      - run_experiment (function): trains a model and returns metrics dict.
      - test_loader, train_loader (DataLoader): data for training and evaluation.

    Output:
      - gat (nn.Module): the trained GATSkipRes model on CUDA in eval mode.
    """
    mo.stop(not btn_train_best_model.value)

    best_num_layers = 5  # 5 GAT layers was identified as optimal in the comparison above

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
    """
    Description:
        Runs the trained GATSkip model on the currently selected individual to produce
        per-node visit probability predictions.  The individual's graph is moved to CUDA
        before the forward pass and results are brought back to CPU as NumPy.

    Input:
      - DataLoader (class): PyG DataLoader used to create the single-item batch.
      - dataset (SyntheticDataset): used to retrieve the selected individual's Data.
      - gat (nn.Module): the trained GATSkipRes model on CUDA.
      - slider_indiv (mo.ui.dropdown): currently selected individual index.
      - torch (module): PyTorch.

    Output:
      - batch (pyg.data.Batch): the CUDA batch containing the selected individual.
      - probs (np.ndarray): per-node predicted visit probabilities, shape (N,).
    """
    _i = slider_indiv.value                         # index of the selected individual
    _loader = DataLoader(dataset[_i : _i + 1])      # DataLoader for a single individual
    batch = next(iter(_loader)).cuda()              # move to GPU for inference

    # Forward pass: edge_attr not used by this model (no edge features in synthetic data)
    _logits = gat(batch.x, batch.edge_index)
    probs = torch.sigmoid(_logits).detach().cpu().numpy()  # convert logits to probabilities
    return batch, probs


@app.cell
def _(
    G,
    cm,
    cmap,
    figsize,
    home_locations,
    mcolors,
    nx,
    pos,
    probs,
    slider_indiv,
    tab10,
):
    """
    Description:
        Draws the network graph coloured by the GATSkip predicted visit probabilities
        for the selected individual.  Nodes are coloured on the diverging vlag_r scale
        (blue = low probability, red = high probability) and the home node is drawn as
        an orange square.  Saves the figure as fig_preds for later export.

    Input:
      - G (nx.DiGraph): the synthetic graph.
      - cm, cmap, mcolors: colour map and normalisation helpers.
      - figsize (tuple): figure size.
      - home_locations (np.ndarray): home node indices.
      - nx (module): networkx.
      - pos (dict): node layout positions.
      - probs (np.ndarray): predicted visit probabilities from the GAT model.
      - slider_indiv (mo.ui.dropdown): currently selected individual.
      - tab10 (list): Seaborn tab10 palette (orange used for home node).

    Output:
      - fig_preds (matplotlib.figure.Figure): the predicted probability map figure.
    """
    fig_preds, _ax = plt.subplots(figsize=figsize, constrained_layout=True)

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
        node_color=tab10[1],
        node_shape="s",
        # edgecolors="green" if visited_nodes[_i, _home_node].item() else None,
        # linewidths=1.5 if visited_nodes[_i, _home_node].item() else None,
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
    _cbar = fig_preds.colorbar(_sm, ax=_ax, location="bottom", shrink=0.8)
    _cbar.set_label("Node visit probability $\\hat{y}_{{i, n}}$")

    _ax.set_axis_off()
    _ax.set_title("GATSkip predicted visit probabilites for example individual $i$")

    fig_preds
    return (fig_preds,)


@app.cell
def _(SEED, batch, gat, torch):
    """
    Description:
        Applies both sampling strategies to the GATSkip predictions for the selected
        individual:
          - Poisson sampling: each node is independently included with probability
            sigmoid(logit_n), yielding a variable-size location choice set.
          - PPS sampling: exactly num_pps_nodes nodes are sampled without replacement,
            with inclusion probability proportional to sigmoid(logit_n).

    Input:
      - SEED (int): used to seed the CUDA random generator for reproducible samples.
      - batch (pyg.data.Batch): CUDA batch for the selected individual.
      - gat (nn.Module): the trained GATSkipRes model.
      - torch (module): PyTorch.

    Output:
      - num_pps_nodes (int): the fixed PPS sample size (7 nodes).
      - poisson_visits (np.ndarray): Poisson sample binary indicators, shape (N,).
      - pps_visits (np.ndarray): PPS sample binary indicators, shape (N,).
    """
    from ml.sampling import poisson_sampling, pps_sampling

    _logits = gat(batch.x, batch.edge_index).detach()  # raw logits, no gradient needed
    _generator = torch.Generator(device="cuda").manual_seed(SEED)

    num_pps_nodes = 7

    poisson_visits = poisson_sampling(_logits, generator=_generator).cpu().numpy()
    pps_visits = pps_sampling(num_pps_nodes, _logits, batch.batch, generator=_generator).cpu().numpy()

    poisson_visits
    return num_pps_nodes, poisson_visits, pps_visits


@app.cell
def _():
    """
    Description:
        Imports Seaborn and extracts the default tab10 colour palette as a list.
        tab10 provides ten distinct, accessible colours used to distinguish categorical
        node classes (e.g. tab10[0] = blue for visited, tab10[1] = orange for home).

    Output:
      - sns (module): seaborn, available for further palette lookups.
      - tab10 (list): list of (R, G, B) tuples for the default tab10 colour cycle.
    """
    import seaborn as sns

    tab10 = sns.color_palette()  # returns 10 (R, G, B) tuples: [blue, orange, green, ...]
    return sns, tab10


@app.cell
def _(
    G,
    cm,
    cmap,
    figsize,
    home_locations,
    mcolors,
    np,
    nx,
    poisson_visits,
    pos,
    slider_indiv,
    tab10,
    utility,
):
    """
    Description:
        Draws the network graph showing the Poisson-sampled location choice set for
        the selected individual.  Sampled (visited) nodes are shown in the first tab10
        colour, unsampled nodes in white with a light-grey border, and the home node
        as an orange square.

    Input:
      - G (nx.DiGraph): the synthetic graph.
      - cm, cmap, mcolors: colour helpers (used for invisible colour bar).
      - figsize (tuple): figure size.
      - home_locations (np.ndarray): home node indices.
      - np (module): numpy.
      - nx (module): networkx.
      - poisson_visits (np.ndarray): binary Poisson sample indicators.
      - pos (dict): node layout positions.
      - slider_indiv (mo.ui.dropdown): currently selected individual.
      - tab10 (list): Seaborn tab10 palette.
      - utility (np.ndarray): only used for colour-bar scaling (invisible).

    Output:
      - fig_poisson (matplotlib.figure.Figure): the Poisson-sampled choice set figure.
    """
    fig_poisson, _ax = plt.subplots(figsize=figsize, constrained_layout=True)

    _i = slider_indiv.value
    _home_node = home_locations[_i].item()

    _score_nodes = poisson_visits

    _unvisited_nodes = [i for i in np.nonzero(1 - _score_nodes)[0].tolist() if i != _home_node]
    _visited_nodes = [i for i in np.nonzero(_score_nodes)[0].tolist() if i != _home_node]

    _cmap = cmap

    nx.draw_networkx_nodes(G, nodelist=_unvisited_nodes, pos=pos, node_color="white", ax=_ax, edgecolors="lightgray")

    nx.draw_networkx_nodes(
        G,
        nodelist=_visited_nodes,
        pos=pos,
        node_color=tab10[0],
        ax=_ax,
    )

    nx.draw_networkx_nodes(
        G,
        nodelist=[_home_node],
        pos=pos,
        node_color=tab10[1],
        node_shape="s",
        ax=_ax,
    )

    _light_nodes = {n: n for n in _visited_nodes + [_home_node]}
    _dark_nodes = {n: n for n in _unvisited_nodes}

    nx.draw_networkx_labels(G, pos, labels=_light_nodes, font_color="white")
    nx.draw_networkx_labels(G, pos, labels=_dark_nodes, font_color="black")

    nx.draw_networkx_edges(G, pos=pos, ax=_ax)

    _sm = cm.ScalarMappable(cmap=_cmap, norm=mcolors.Normalize(vmin=-1, vmax=1))
    _sm.set_array(utility[_i])
    _cbar = fig_poisson.colorbar(_sm, ax=_ax, location="bottom", shrink=0.8)

    _cbar.solids.set_alpha(0)
    _cbar.outline.set_alpha(0)
    _cbar.ax.tick_params(labelcolor="none", color="none")
    _cbar.set_label("")

    _ax.set_axis_off()
    _ax.set_title("Poisson-sampled location choice set")

    fig_poisson
    return (fig_poisson,)


@app.cell
def _():
    """
    Description:
        Defines the figure size used for the single-panel network visualisation plots
        in this notebook section (utility, noise, visit, predictions, Poisson, PPS).

    Output:
      - figsize (tuple[int, int]): (5, 5) — square figure for individual graph panels.
    """
    figsize = (5, 5)  # square figures for single-panel network visualisations
    return (figsize,)


@app.cell
def _(
    G,
    cm,
    cmap,
    fig_poisson,
    figsize,
    home_locations,
    mcolors,
    np,
    num_pps_nodes,
    nx,
    pos,
    pps_visits,
    slider_indiv,
    tab10,
    utility,
):
    """
    Description:
        Draws the network graph showing the PPS-sampled location choice set of fixed
        size num_pps_nodes for the selected individual.  Sampled nodes are shown in the
        first tab10 colour, unsampled nodes in white, and the home node as an orange square.

    Input:
      - G (nx.DiGraph): the synthetic graph.
      - cm, cmap, mcolors: colour helpers (invisible colour bar for layout consistency).
      - fig_poisson (Figure): used to attach the invisible colour bar (for alignment).
      - figsize (tuple): figure size.
      - home_locations (np.ndarray): home node indices.
      - np (module): numpy.
      - num_pps_nodes (int): the number of nodes in the fixed PPS sample.
      - nx (module): networkx.
      - pos (dict): node layout positions.
      - pps_visits (np.ndarray): binary PPS sample indicators.
      - slider_indiv (mo.ui.dropdown): currently selected individual.
      - tab10 (list): Seaborn tab10 palette.
      - utility (np.ndarray): used only for colour-bar scaling range.

    Output:
      - fig_pps (matplotlib.figure.Figure): the PPS-sampled choice set figure.
    """
    fig_pps, _ax = plt.subplots(figsize=figsize, constrained_layout=True)

    _i = slider_indiv.value
    _home_node = home_locations[_i].item()

    _score_nodes = pps_visits

    _unvisited_nodes = [i for i in np.nonzero(1 - _score_nodes)[0].tolist() if i != _home_node]
    _visited_nodes = [i for i in np.nonzero(_score_nodes)[0].tolist() if i != _home_node]

    _cmap = cmap

    nx.draw_networkx_nodes(G, nodelist=_unvisited_nodes, pos=pos, node_color="white", ax=_ax, edgecolors="lightgray")

    nx.draw_networkx_nodes(
        G,
        nodelist=_visited_nodes,
        pos=pos,
        node_color=tab10[0],
        ax=_ax,
    )

    nx.draw_networkx_nodes(
        G,
        nodelist=[_home_node],
        pos=pos,
        node_color=tab10[1],
        node_shape="s",
        ax=_ax,
    )

    _light_nodes = {n: n for n in _visited_nodes + [_home_node]}
    _dark_nodes = {n: n for n in _unvisited_nodes}

    nx.draw_networkx_labels(G, pos, labels=_light_nodes, font_color="white")
    nx.draw_networkx_labels(G, pos, labels=_dark_nodes, font_color="black")

    nx.draw_networkx_edges(G, pos=pos, ax=_ax)

    _sm = cm.ScalarMappable(cmap=_cmap, norm=mcolors.Normalize(vmin=-1, vmax=1))
    _sm.set_array(utility[_i])
    _cbar = fig_poisson.colorbar(_sm, ax=_ax, location="bottom", shrink=0.8)

    _cbar.solids.set_alpha(0)
    _cbar.outline.set_alpha(0)
    _cbar.ax.tick_params(labelcolor="none", color="none")
    _cbar.set_label("")

    _ax.set_axis_off()
    _ax.set_title(f"PPS-sampled location choice set of size $m = {num_pps_nodes}$ ")

    fig_pps
    return (fig_pps,)


if __name__ == "__main__":
    app.run()
