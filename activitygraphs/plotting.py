import base64
import itertools
from io import BytesIO
from typing import Protocol

import altair as alt
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import polars as pl
from exploration.dataprocessing import Purpose
from exploration.graphs import ActivityGraph
from exploration.metrics import Metrics
from matplotlib.axes import Axes

PURPOSE_IMPORTANCE = [Purpose.HOME, Purpose.WORK, Purpose.EDUCATION]

alt.data_transformers.enable("vegafusion")


def line_styles_by_key(G: nx.MultiDiGraph, key: str = "person_id"):
    person_ids = [person_id for _, _, person_id in G.edges.data(data=key)]
    person_id_set = set(person_ids)

    line_styles = itertools.cycle(["-", ":", "-.", "--"])
    line_person_mapping = {k: ls for k, ls in zip(person_id_set, line_styles)}

    return [line_person_mapping[person_id] for person_id in person_ids]


def _find_main_value(values, incomplete_ordering):
    values_set = set(values)

    for elem in incomplete_ordering:
        if elem in values_set:
            return elem

    return next(filter(lambda v: v not in incomplete_ordering, values))


def node_colours_by_purpose(G: nx.MultiDiGraph):
    node_colours = []

    for _, purposes in G.nodes.data(data="purposes"):
        main_purpose = _find_main_value(purposes, PURPOSE_IMPORTANCE)
        colour = _map_purpose_to_colour(main_purpose)
        node_colours.append(colour)

    return node_colours


def node_short_labels_by_purpose(G: nx.MultiDiGraph) -> dict[str, str]:
    node_labels = {}

    for n, purposes in G.nodes.data(data="purposes"):
        main_purpose = _find_main_value(purposes, PURPOSE_IMPORTANCE)
        short_label = _map_purpose_to_short_label(main_purpose)
        node_labels[n] = short_label

    return node_labels


def _map_purpose_to_colour(purpose: Purpose):
    match purpose:
        case Purpose.HOME:
            return "#6929c4"
        case Purpose.WORK | Purpose.EDUCATION:
            return "#1192e8"
        case Purpose.WORK_DELIVERY | Purpose.WORK_OTHER:
            return "#005d5d"
        case Purpose.ENTERTAINMENT | Purpose.SPORT | Purpose.LEISURE:
            return "#9f1853"
        case Purpose.SHOPPING_FOOD | Purpose.SHOPPING_OTHER:
            return "#fa4d56"
        case Purpose.PERSONAL_BUSINESS:
            return "#570408"
        case Purpose.HOTEL:
            return "#198038"
        case Purpose.ESCORT_WORK | Purpose.ESCORT_HEALTH | Purpose.ESCORT_SCHOOL | Purpose.ESCORT_OTHER:
            return "#002d9c"
        case Purpose.WORSHIP:
            return "#ee538b"
        case Purpose.OTHER:
            return "#b28600"
        case Purpose.HEALTH:
            return "#009d9a"
        case Purpose.SOCIAL_VISIT | Purpose.SOCIAL_OTHER:
            return "#012749"
        case _:
            raise ValueError(purpose)


def _map_purpose_to_short_label(purpose: Purpose):
    match purpose:
        case Purpose.HOME:
            return "H"
        case Purpose.WORK:
            return "W"
        case Purpose.EDUCATION:
            return "Ed"
        case Purpose.WORK_DELIVERY:
            return "Wd"
        case Purpose.WORK_OTHER:
            return "Wo"
        case Purpose.ENTERTAINMENT | Purpose.SPORT | Purpose.LEISURE:
            return "L"
        case Purpose.SHOPPING_FOOD | Purpose.SHOPPING_OTHER:
            return "Sh"
        case Purpose.PERSONAL_BUSINESS | Purpose.HOTEL:
            return "P"
        case Purpose.ESCORT_WORK | Purpose.ESCORT_HEALTH | Purpose.ESCORT_SCHOOL | Purpose.ESCORT_OTHER:
            return "Es"
        case Purpose.WORSHIP:
            return "Wo"
        case Purpose.OTHER:
            return "O"
        case Purpose.HEALTH:
            return "H"
        case Purpose.SOCIAL_VISIT | Purpose.SOCIAL_OTHER:
            return "So"
        case _:
            raise ValueError(purpose)


def _ax_centered_text(text: str, ax: plt.Axes):
    left, width = 0.25, 0.5
    bottom, height = 0.25, 0.5
    right = left + width
    top = bottom + height

    ax.text(
        0.5 * (left + right),
        0.5 * (bottom + top),
        text,
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
        color="black",
    )


def draw_hh_graph(
    G: nx.MultiDiGraph,
    hh_id=None,
    line_style_key="person_id",
    node_colours=None,
    node_labels=None,
    use_coords=False,
):
    title = "Activity graph" if hh_id is None else f"Act. graph of household: {hh_id}"

    fig, ax = plt.subplots()
    fig.set_facecolor("white")
    fig.set_size_inches(8, 6)

    ax.set_title(title, loc="left", color="black")
    ax.axis("off")

    line_styles = line_styles_by_key(G, key=line_style_key)
    node_colours = node_colours_by_purpose(G) if node_colours is None else node_colours
    node_labels = node_short_labels_by_purpose(G) if node_labels is None else node_labels

    if len(G.nodes) == 0:
        _ax_centered_text("No activities.", ax)

    if use_coords:
        pos = {node: (data["lon"], data["lat"]) for node, data in G.nodes(data=True)}
    else:
        pos = nx.layout.kamada_kawai_layout(G, weight="distance")

    nx.draw_networkx_nodes(G, pos=pos, ax=ax, node_color=node_colours)
    nx.draw_networkx_labels(G, pos=pos, ax=ax, labels=node_labels, font_color="white")
    nx.draw_networkx_edges(G, pos=pos, ax=ax, connectionstyle="arc3,rad=0.1", style=line_styles)

    return fig, ax


def _plot_metric_histogram(metric_col: str, results: pl.DataFrame, bin_count=20) -> alt.Chart:
    return results.plot.bar(alt.X(metric_col).bin(maxbins=bin_count), alt.Y("count()"))


def plot_metric_histograms(metrics: Metrics, results: pl.DataFrame, n_cols=2) -> alt.Chart:
    chart = alt.vconcat()

    for batch in itertools.batched(metrics.names(), n_cols):
        row = alt.hconcat()
        for metric_col in batch:
            row |= _plot_metric_histogram(metric_col, results)
        chart &= row

    return chart.resolve_scale("independent")


def geo_plot_mean_stat_by_postcode(
    mean_stats_by_postcode: pl.DataFrame,
    geo_postcode_shapes: gpd.GeoDataFrame,
    postcode_split: str,
    stat: str,
):
    mean_gdf = geo_postcode_shapes.merge(
        mean_stats_by_postcode.to_pandas(),
        left_on="name",
        right_on=postcode_split,
    )

    title = f"Average HH Graph {stat} by uk postcode {postcode_split}"

    if postcode_split != "sector":
        return (
            alt.Chart(mean_gdf, title=title)
            .mark_geoshape()
            .encode(color=stat, tooltip=["name", stat, "n_samples"])
            .properties(width=500, height=500)
        )

    else:
        fig, ax = plt.subplots()

        ax = mean_gdf.plot(
            ax=ax,
            column=stat,
            legend=True,
            legend_kwds={"label": stat, "orientation": "horizontal"},
        )

        ax.set_title(title)

        return fig


def geo_plot_mean_stat_by_municipality(
    mean_stats_by_municipality: pl.DataFrame,
    geo_municipality_shapes: gpd.GeoDataFrame,
    stat: str,
):
    mean_gdf = geo_municipality_shapes.merge(
        mean_stats_by_municipality.to_pandas(), left_on="LAD24CD", right_on="municipality_id", how="right"
    ).to_crs("EPSG:4326")

    title = f"Average HH Graph {stat} by municipality"

    return (
        alt.Chart(mean_gdf, title=title)
        .mark_geoshape()
        .encode(color=stat, tooltip=["municipality_id", "municipality_name", stat, "n_samples"])
        .properties(width=500, height=500)
    )


def build_dash_graph_scatter(results: pl.DataFrame, graph: ActivityGraph):
    import plotly.graph_objects as go
    from dash import Dash, Input, Output, callback, dcc, html, no_update

    _fig = go.Figure(
        go.Scatter3d(
            x=results["x"],
            y=results["y"],
            z=results["z"],
            mode="markers",
            marker=dict(
                colorscale="viridis",
                color=results["c"],
                line={"color": "#444"},
                reversescale=True,
                sizeref=45,
                sizemode="diameter",
                opacity=0.8,
            ),
        ),
        layout=dict(
            width=1500,
            height=1000,
        ),
    )

    _fig.update_traces(hoverinfo="none", hovertemplate=None)

    app = Dash()

    app.layout = html.Div([
        dcc.Graph(id="graph-basic-2", figure=_fig, clear_on_unhover=True),
        dcc.Tooltip(id="graph-tooltip"),
    ])

    @callback(
        Output("graph-tooltip", "show"),
        Output("graph-tooltip", "bbox"),
        Output("graph-tooltip", "children"),
        Input("graph-basic-2", "hoverData"),
    )
    def display_hover(hoverData):
        if hoverData is None:
            return False, no_update, no_update

        # demo only shows the first point, but other points may also be available
        pt = hoverData["points"][0]
        bbox = pt["bbox"]
        num = pt["pointNumber"]

        df_row = results.row(num, named=True)
        hh_id = df_row["hh_id"]

        buf = BytesIO()
        fig, _ = draw_hh_graph(graph.to_nx(hh_id), hh_id)
        fig.savefig(buf, format="png")
        plt.close()
        buf.seek(0)

        # noinspection PyTypeChecker
        image_string = base64.b64encode(buf.getvalue()).decode()
        image_string = f"data:image/png;base64,{image_string}"

        children = [
            html.Div(
                [
                    html.H2(f"HH Graph #{hh_id}", style={"color": "darkblue", "overflow-wrap": "break-word"}),
                    # html.P(f"Cluster: {c}"),
                    html.Img(src=image_string, style={"width": "100%"}),
                ],
                style={"width": "400px", "white-space": "normal"},
            )
        ]

        return True, bbox, children

    return app


""" ============================================================================ """


def _default_axes(ax: Axes = None) -> Axes:
    return ax if ax is not None else plt.subplots(figsize=(10, 5))[1]


class Graph(Protocol):
    """Protocol class emulating SyntheticGraph, as argument to `draw_synthetic_network`"""

    WEIGHT_NAME: str

    @property
    def G(self) -> nx.Graph:
        pass

    @property
    def G_full(self) -> nx.Graph:
        pass


def draw_synthetic_network(graph: Graph, full=False, ax: Axes = None):
    """Draws a SyntheticGraph with shopping and work nodes highlighted

    Args:
        graph (Graph): the graph to draw
        full (bool, optional): draw the fully-connected version of the graph. Defaults to False.
        ax (Axes, optional): the Axes on which to draw. Creates a new Axes if None. Defaults to None.

    Returns:
        Axes: the drawn axes
    """

    def _node_colour(node_attrs: dict) -> str:
        if "is_shopping" not in node_attrs or "is_workplace" not in node_attrs:
            return "tab:gray"

        if node_attrs["is_shopping"] and node_attrs["is_workplace"]:
            return "orangered"
        if node_attrs["is_shopping"]:
            return "orange"
        if node_attrs["is_workplace"]:
            return "tomato"

        return "tab:blue"

    G = graph.G_full if full else graph.G
    ax = _default_axes(ax)

    pos = nx.spring_layout(G, seed=42, weight=graph.WEIGHT_NAME)
    edge_labels = nx.get_edge_attributes(G, graph.WEIGHT_NAME)

    colors = [_node_colour(attrs) for _, attrs in G.nodes(data=True)]

    nx.draw_networkx(G, pos, node_color=colors, ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax)

    return ax


class Schedules(Protocol):
    """Protocol class emulating SyntheticGraph, as argument to `draw_synthetic_trip`"""

    graph: Graph
    trip_df: pl.DataFrame


def draw_synthetic_trip(schedules: Schedules, person_id: int, full=False, ax: Axes = None):
    """Draws the SyntheticGraph in schedules with a person's schedule and trips overlain.

    Args:
        schedules (Schedules): The schedules from which to draw from
        person_id (int): the ID of the person whose schedule you want to draw
        full (bool, optional): draw the fully-connected version of the graph. Defaults to False.
        ax (Axes, optional): the Axes on which to draw. Creates a new Axes if None. Defaults to None.

    Returns:
        Axes: the drawn axes
    """

    def _activities_to_colors(types: pl.Series):
        if "H" in types:
            return "tab:blue"
        if "W" in types and ("S1" in types or "S2" in types):
            return "orangered"
        if "S1" in types or "S2" in types:
            return "orange"
        if "W" in types:
            return "tomato"

        raise NotImplementedError("Impossible")

    graph = schedules.graph
    G = graph.G_full if full else graph.G
    ax = _default_axes(ax)

    pos = nx.spring_layout(G, seed=42, weight=graph.WEIGHT_NAME)
    edge_labels = nx.get_edge_attributes(G, graph.WEIGHT_NAME)

    trips = schedules.trip_df.filter(pl.col("person_id") == person_id)
    edgelist = trips.select("from_loc_id", "to_loc_id").rows()
    node_colours = (
        pl.concat([
            trips.select("from_loc_id", "from_type").rename({"from_loc_id": "loc_id", "from_type": "type"}),
            trips.select("to_loc_id", "to_type").rename({"to_loc_id": "loc_id", "to_type": "type"}),
        ])
        .group_by("loc_id")
        .agg(pl.col("type").map_batches(_activities_to_colors, return_dtype=pl.String).first())
        .join(pl.DataFrame({"loc_id": list(G.nodes())}), on="loc_id", how="right")
        .with_columns(pl.col("type").fill_null("tab:gray"))
    )["type"].to_list()

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colours)
    nx.draw_networkx_labels(G, pos, ax=ax)
    nx.draw_networkx_edges(G, pos, ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax)
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=edgelist,
        arrows=True,
        arrowstyle="-|>",
        style="--",
        connectionstyle="arc3,rad=0.2",
        edge_color="red",
        ax=ax,
    )

    return ax


class Results(Protocol):
    """Protocol class emulating experiment.Results, as argument to `plot_training_progress`"""

    n_epochs: int
    name: str

    def final_losses(self) -> pl.DataFrame:
        pass

    def train_losses(self) -> pl.DataFrame:
        pass

    def val_losses(self) -> pl.DataFrame:
        pass

    def test_loss(self) -> float:
        pass

    def has_training_history(self) -> bool:
        pass


def plot_training_progress(results: Results, ax: Axes = None):
    """Plots the training, validation and test losses w.r.t the epochs.

    Args:
        results (experiment.Results): the results to be plotted
        ax (Axes, optional): the Axes on which to draw. Creates a new Axes if None. Defaults to None.

    Returns:
        Axes: the drawn axes
    """
    ax = _default_axes(ax)

    epochs = list(range(1, results.n_epochs + 1))
    _, _, test_loss = results.final_losses()

    ax.grid()
    ax.plot(epochs, results.train_losses(), label="Train")
    ax.plot(epochs, results.val_losses(), label="Validation")
    ax.plot([1, results.n_epochs], [test_loss, test_loss], label="Test", linestyle="dashed", linewidth=1)
    ax.set_title(f"{results.name} losses")
    ax.set_xlabel("Epoch")
    ax.set_xlim((1, results.n_epochs))
    ax.set_ylabel("CE Loss")
    ax.legend()

    return ax


def draw_prediction(graph: Graph, x: list, y_prob: list, full=False, labels=True, ax: Axes = None):
    """Draws the predictions of an ML model over the graph.

    Args:
        graph (Graph): the graph on which to draw the predictions
        x (list): the input node indicators (each node truthy if already selected, of length N)
        y_prob (list): the probabilities for each node (of length N)
        full (bool, optional): draw the fully-connected version of the graph. Defaults to False.
        labels (bool, optional): add the prediction values as node labels. Defaults to True.
        ax (Axes, optional): the Axes on which to draw. Creates a new Axes if None. Defaults to None.

    Returns:
        Axes: the drawn axes
    """

    def _node_colour(x, y):
        if x:
            return "tab:blue"

        return mpl.colormaps["grey_r"](y)

    ax = _default_axes(ax)

    G = graph.G_full if full else graph.G
    pos = nx.spring_layout(G, seed=42, weight=graph.WEIGHT_NAME)
    edge_labels = nx.get_edge_attributes(G, graph.WEIGHT_NAME)
    colors = [_node_colour(_x, _y) for _x, _y in zip(x, y_prob)]

    nx.draw_networkx(G, pos, node_color=colors, ax=ax, edgecolors="gray", font_color="DimGray")
    nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax)

    if labels:
        label_pos = {n: (x, y + 0.15) for n, (x, y) in pos.items()}
        lab = {n: f"{y:.2f}" for n, y in zip(G.nodes(), y_prob)}
        nx.draw_networkx_labels(G, label_pos, lab, font_color="red", font_size=10, ax=ax)

    return ax


def plot_model_comparisons(*results: Results, how="bar", ax: Axes = None):
    """Plots a comparison between results from different models as a bar chart or as a line chart with training history.

    Args:
        how (str, optional): "bar" or "line". Defaults to "bar".
        ax (Axes, optional): the Axes on which to draw. Creates a new Axes if None. Defaults to None.

    Returns:
        Axes: the drawn axes
    """
    ax = _default_axes(ax)

    if how == "bar":
        return _plot_model_comparisons_bar(results, ax)
    elif how == "line":
        return _plot_model_comparisons_line(results, ax)

    raise KeyError(f"Unknown plot '{how}'. Valid entries are 'bar' or 'line'")


def _plot_model_comparisons_bar(results: tuple[Results, ...], ax: Axes = None):
    xs = [r.name for r in results]
    heights = [r.test_loss() for r in results]
    labels = [f"{h:.4f}" for h in heights]

    colors = ["tab:orange" if r.has_training_history() else "tab:blue" for r in results]

    ax.set_axisbelow(True)
    b = ax.bar(xs, heights, color=colors)
    ax.bar_label(b, labels)
    ax.grid()
    ax.set_title("Model comparison (Test CE loss)")
    ax.set_xlabel("Model")
    ax.set_ylabel("CE Loss")

    return ax


def _plot_model_comparisons_line(results: tuple[Results, ...], ax: Axes = None):
    max_epochs = max(res.n_epochs for res in results)
    trained_results = [res for res in results if res.has_training_history()]
    benchmark_results = [res for res in results if not res.has_training_history()]

    ax.grid()

    for trained_res in trained_results:
        test = trained_res.test_loss()

        epochs = list(range(1, trained_res.n_epochs + 1))
        line = ax.plot(epochs, trained_res.val_losses(), label=f"{trained_res.name} (val)")
        ax.plot(
            [1, trained_res.n_epochs],
            [test, test],
            linestyle="dashed",
            linewidth=1,
            color=line[0].get_color(),
            label=trained_res.name,
        )

    for benchmark_res in benchmark_results:
        test = benchmark_res.test_loss()
        ax.plot([1, max_epochs], [test, test], label=benchmark_res.name, linestyle="dashed", linewidth=1)

    ax.set_title("Model losses")
    ax.set_xlabel("Epoch")
    ax.set_xlim((1, max_epochs))
    ax.set_ylabel("BCE Loss")
    ax.legend()

    return ax
