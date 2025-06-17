import itertools

import altair as alt
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.style
import networkx as nx
import polars as pl

from dataprocessing import Purpose
from metrics import Metrics

PURPOSE_IMPORTANCE = [Purpose.HOME, Purpose.WORK, Purpose.EDUCATION]

alt.data_transformers.enable("vegafusion")

matplotlib.style.use("dark_background")
alt.theme.enable("dark")


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
    line_styles=None,
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


def _plot_metric_historgram(metric_col: str, results: pl.DataFrame, bin_count=20) -> alt.Chart:
    return results.plot.bar(alt.X(metric_col).bin(maxbins=bin_count), alt.Y("count()"))


def plot_metric_histograms(metrics: Metrics, results: pl.DataFrame, n_cols=2) -> alt.Chart:
    chart = alt.vconcat()

    for batch in itertools.batched(metrics.names(), n_cols):
        row = alt.hconcat()
        for metric_col in batch:
            row |= _plot_metric_historgram(metric_col, results)
        chart &= row

    return chart


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
