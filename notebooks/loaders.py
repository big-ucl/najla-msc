import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    return mo, plt


@app.cell
def _(mo):
    from config import load_config

    cfg = load_config(mo.notebook_dir().parent)
    return


@app.cell
def _():
    import networkx as nx
    return (nx,)


@app.cell
def _(nx):
    edges = {
        ("A", "B"): 5,
        ("B", "C"): 5,
        ("C", "A"): 5,
        ("C", "D"): 15,
        ("D", "E"): 3,
        ("E", "F"): 7,
        ("F", "D"): 9,
    }

    G = nx.Graph()
    G.add_weighted_edges_from(((u, v, w) for ((u, v), w) in edges.items()), weight="distance")

    nodes = list(G.nodes())
    return G, nodes


@app.cell
def _(G, nx, plt):
    def draw_network(G):
        pos = nx.spring_layout(G, seed=42, weight="distance")
        edge_labels = nx.get_edge_attributes(G, "distance")

        nx.draw_networkx(G, pos)
        nx.draw_networkx_edge_labels(G, pos, edge_labels)
        plt.show()


    draw_network(G)
    return (draw_network,)


@app.cell
def _(G, nodes, nx):
    import numpy as np


    def node_ordering(nodes: list[str]):
        return dict((node, idx) for idx, node in enumerate(nodes))


    def distance_matrix(G: nx.Graph, nodes: list[str]):
        matrix = np.empty((len(nodes), len(nodes)), dtype=np.float32)
        ordering = node_ordering(nodes)
        distances = nx.shortest_path_length(G, weight="distance")

        for node, distance in distances:
            idx = ordering[node]
            row_items = sorted(distance.items(), key=lambda x: ordering[x[0]])
            row = [dist for _, dist in row_items]

            matrix[idx, :] = row

        return matrix


    distances = distance_matrix(G, nodes)
    distances
    return (distance_matrix,)


@app.cell
def _(G, distance_matrix, draw_network, nodes, nx):
    def fully_connected_graph(G: nx.Graph, nodes: list[str]):
        distances = distance_matrix(G, nodes)
        return nx.from_numpy_array(distances, edge_attr="distance", nodelist=nodes)


    G_full = fully_connected_graph(G, nodes)
    draw_network(G_full)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
