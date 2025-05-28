import networkx as nx
import matplotlib.pyplot as plt
import itertools

def line_styles_by_key(G: nx.Graph, key: str="person_id"):
    person_ids = [person_id for _, _, person_id in G.edges.data(data="person_id")]
    person_id_set = set(person_ids)

    line_styles = itertools.cycle(["-", ":", "-.", "--"])
    line_person_mapping = { k: ls for k, ls in zip(person_id_set, line_styles)}

    return [line_person_mapping[person_id] for person_id in person_ids]


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
    )

def draw_hh_graph(G: nx.MultiDiGraph, hh_id=None, line_styles=None):
    title = "Activity graph" if hh_id is None else f"Act. graph of household: {hh_id}"

    fig, ax = plt.subplots()
    ax.set_title(title, loc="left")

    if len(G.nodes) == 0:
        _ax_centered_text("No activities.", ax)

    pos = nx.layout.kamada_kawai_layout(G, weight="distance")
    nx.draw_networkx_nodes(G, pos=pos, ax=ax)
    nx.draw_networkx_labels(G, pos=pos, ax=ax)
    nx.draw_networkx_edges(G, pos=pos, ax=ax, connectionstyle="arc3,rad=0.1", style=line_styles)

    return fig