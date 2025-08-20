import multiprocessing as mp
import os

import exploration.graphs as g
import networkx as nx


def _convert_to_nx(graph: g.ActivityGraph) -> dict[str, nx.MultiDiGraph]:
    nx_graphs = {}

    for hh_id, nx_graph in graph.to_nxs():
        nx_graphs[hh_id] = nx_graph

    return nx_graphs


def parallel_to_nx(activity_graph: g.ActivityGraph) -> dict[str, nx.MultiDiGraph]:
    n_proc = os.cpu_count()
    partitions = activity_graph.partition_by_hh_id(n_chunks=n_proc)

    ctx = mp.get_context("spawn")

    with ctx.Pool(processes=n_proc) as pool:
        res = pool.map(_convert_to_nx, partitions)

    return {hh_id: nx_graph for gs in res for hh_id, nx_graph in gs.items()}
