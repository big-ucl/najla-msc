"""
Module: activitygraphs/archive/exploration/multi.py

Description:
    Provides multiprocessing utilities for converting large ActivityGraphs to
    NetworkX MultiDiGraph objects in parallel.

    Converting thousands of household activity graphs from Polars DataFrames to
    NetworkX objects is CPU-bound and can be slow when done sequentially.  This
    module uses Python's multiprocessing module to split the households into
    equal-sized chunks and process each chunk in a separate worker process,
    then merges the results back into a single dictionary.

    The "spawn" multiprocessing context is used (rather than "fork") to avoid
    issues with CUDA/GPU libraries or shared state that can cause deadlocks on
    Linux and macOS when forking a process that has already imported complex libraries.

    Usage example:
        from archive.exploration.multi import parallel_to_nx
        from archive.exploration.graphs import ActivityGraph

        graph = ActivityGraph.from_dataset(dataset)
        nx_graphs = parallel_to_nx(graph)  # dict[hh_id -> nx.MultiDiGraph]
"""

import multiprocessing as mp
import os

import archive.exploration.graphs as g
import networkx as nx


def _convert_to_nx(graph: g.ActivityGraph) -> dict[str, nx.MultiDiGraph]:
    """
    Description:
        Worker function executed in a separate process by the multiprocessing pool.
        Converts all household subgraphs in a given ActivityGraph chunk into
        NetworkX MultiDiGraph objects and returns them as a dictionary.

        This function is designed to be called via pool.map() and must be
        picklable (no lambda or closures), which is why it is a module-level function.

    Input:
      - graph (ActivityGraph): an ActivityGraph chunk (typically one of several
            partitions created by partition_by_hh_id()). Contains a subset of
            all households.

    Output:
      - (dict[str, nx.MultiDiGraph]): mapping from household ID string to the
            corresponding NetworkX directed multigraph for that household.
    """
    nx_graphs = {}  # Accumulator dict: hh_id -> NetworkX graph

    # Iterate over all households in this chunk and convert each to a NetworkX graph
    for hh_id, nx_graph in graph.to_nxs():
        nx_graphs[hh_id] = nx_graph

    return nx_graphs


def parallel_to_nx(activity_graph: g.ActivityGraph) -> dict[str, nx.MultiDiGraph]:
    """
    Description:
        Converts an entire ActivityGraph (potentially thousands of households) into
        a dictionary of NetworkX MultiDiGraph objects using all available CPU cores.

        Strategy:
          1. Determine the number of available logical CPU cores (os.cpu_count()).
          2. Partition the ActivityGraph into n_proc equal-sized chunks using
             partition_by_hh_id().
          3. Spawn a multiprocessing pool and distribute one chunk per worker process.
          4. Merge all per-worker result dicts into a single flat dictionary.

        Using "spawn" context ensures that each worker starts fresh without
        inheriting the parent's memory state, which is safer for complex ML/GIS
        library environments.

    Input:
      - activity_graph (ActivityGraph): the full dataset ActivityGraph to convert.
            Must contain at least os.cpu_count() households for full parallelism.

    Output:
      - (dict[str, nx.MultiDiGraph]): mapping from household ID string to its
            NetworkX directed multigraph, covering all households in activity_graph.
    """
    # Use all logical CPU cores available on this machine
    n_proc = os.cpu_count()

    # Split households evenly across n_proc chunks for parallel processing
    partitions = activity_graph.partition_by_hh_id(n_chunks=n_proc)

    # Use "spawn" to start fresh worker processes (safer than "fork" for our libraries)
    ctx = mp.get_context("spawn")

    # Launch n_proc worker processes; each converts one chunk to a dict of NetworkX graphs
    with ctx.Pool(processes=n_proc) as pool:
        res = pool.map(_convert_to_nx, partitions)  # res is a list of dicts, one per worker

    # Flatten the list of per-worker dicts into a single dict: hh_id -> nx.MultiDiGraph
    return {hh_id: nx_graph for gs in res for hh_id, nx_graph in gs.items()}
