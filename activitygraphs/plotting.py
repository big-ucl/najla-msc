import matplotlib.pyplot as plt
import networkx as nx


def ax_centered_text(text: str, ax: plt.Axes):
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
    )


def draw_hh_graph(G: nx.MultiDiGraph, hh_id=None):
    title = "Activity graph" if hh_id is None else f"Act. graph of household: {hh_id}"

    fig, ax = plt.subplots()
    ax.set_title(title, loc="left")

    if len(G.nodes) == 0:
        ax_centered_text("No activities.", ax)

    pos = nx.layout.kamada_kawai_layout(G, weight="distance")
    nx.draw_networkx(G, pos=pos, ax=ax, with_labels=True, connectionstyle="arc3,rad=0.1")

    return fig
