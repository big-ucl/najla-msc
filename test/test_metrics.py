"""
Module: test/test_metrics.py

Description:
    Unit tests for three additional metric and model components:

    1. FullyConnectedMLP — a flat (non-graph) MLP that processes all nodes at once,
       used as a full-information baseline.  Tests verify output shape and that it
       rejects inputs with the wrong number of nodes.

    2. compute_home_hop_distance — computes the shortest-path hop distance from every
       node to every home node in the graph.  Tests verify correctness on a simple path
       graph and that disconnected nodes get distance = inf.

    3. Hop-band metrics — ActivityGraphModule can optionally compute metrics separately
       for nodes within 0-2 hops of home vs 3-5 hops.  Tests verify the expected metric
       keys are logged and that hop-band metrics are absent when home_hop_distance is None.
"""

import torch
import torch_geometric as pyg

from activitygraphs.ml.lightning_module import ActivityGraphModule
from activitygraphs.ml.metrics import compute_home_hop_distance
from activitygraphs.ml.models import FullyConnectedMLP, NodeMLP

# A simple undirected path graph: 0 -- 1 -- 2 -- 3 -- 4
# Represented as edge_index with both directions so BFS/shortest paths work correctly.
# Hop distance from node 0: [0, 1, 2, 3, 4].  From node 2: [2, 1, 0, 1, 2].
PATH_EDGE_INDEX = torch.tensor([[0, 1, 2, 3, 1, 2, 3, 4], [1, 2, 3, 4, 0, 1, 2, 3]])


class TestFullyConnectedMLP:
    """
    Description:
        Tests for FullyConnectedMLP, a flat MLP baseline that receives all nodes of a
        single graph concatenated into a single input vector (treating the graph as a
        fixed-size table rather than a graph structure).
    """

    def test_forward_shape(self):
        """
        Description:
            Verifies that the model produces an output tensor with the same number of
            rows as the input and 1 column (binary prediction per node).

        Input:
          Setup: 2 graphs of 4 nodes each, 3 input features per node.

        Assertion:
          - output.shape == (8, 1) for 2 * 4 = 8 total nodes.
        """
        num_nodes, in_features, num_graphs = 4, 3, 2
        model = FullyConnectedMLP(num_nodes, in_features, hidden_channels=8, num_layers=3, dropout=0.0)
        x = torch.rand(num_graphs * num_nodes, in_features)
        out = model(x, torch.zeros(2, 0, dtype=torch.long), None, None)
        assert out.shape == (num_graphs * num_nodes, 1)

    def test_rejects_wrong_node_count(self):
        """
        Description:
            Verifies that passing an input tensor whose row count is NOT a multiple of
            num_nodes raises a ValueError.  FullyConnectedMLP requires a fixed graph
            size because it concatenates all node features into one vector.

        Input:
          Setup: model built with num_nodes=4 but input has 5 rows.

        Assertion:
          - A ValueError is raised (not silently producing garbage output).
        """
        model = FullyConnectedMLP(num_nodes=4, in_features=3, hidden_channels=8)
        x = torch.rand(5, 3)  # 5 is not a multiple of num_nodes=4
        try:
            model(x, torch.zeros(2, 0, dtype=torch.long), None, None)
            raise AssertionError("expected a ValueError for a non-multiple node count")
        except ValueError:
            pass


class TestHomeHopDistance:
    """
    Description:
        Tests for compute_home_hop_distance, which computes the all-pairs shortest-path
        hop distance matrix for a graph.  This matrix is used to define "hop bands"
        (e.g. 0-2 hops from home vs 3-5 hops) for per-band metric computation.
    """

    def test_path_graph(self):
        """
        Description:
            Verifies correctness of hop distances on the simple path graph
            0 -- 1 -- 2 -- 3 -- 4.  Distances from node 0 should be [0,1,2,3,4]
            and from node 2 should be [2,1,0,1,2].

        Input:
          Setup: PATH_EDGE_INDEX (the bidirectional 5-node path graph defined at module level).

        Assertions:
          - hop[0] == [0, 1, 2, 3, 4].
          - hop[2] == [2, 1, 0, 1, 2].
        """
        hop = compute_home_hop_distance(PATH_EDGE_INDEX, num_nodes=5)
        assert hop[0].tolist() == [0, 1, 2, 3, 4]
        assert hop[2].tolist() == [2, 1, 0, 1, 2]

    def test_disconnected_is_inf(self):
        """
        Description:
            Verifies that unreachable nodes (no path exists) get hop distance = inf.
            This prevents them from being assigned to any finite hop band.

        Input:
          Setup: a minimal 3-node graph with only edge 0 -> 1; node 2 is isolated.

        Assertion:
          - hop[0, 2] == float("inf").
        """
        edge_index = torch.tensor([[0], [1]])  # only 0-1 connected; node 2 isolated
        hop = compute_home_hop_distance(edge_index, num_nodes=3)
        assert hop[0, 2] == float("inf")


class TestHopBandMetrics:
    """
    Description:
        Tests for the hop-band metric computation in ActivityGraphModule.  When a
        home_hop_distance matrix is provided, the module logs test metrics separately
        for nodes within 0-2 hops of home and for nodes 3-5 hops away.
    """

    def _make_graph(self):
        """
        Description:
            Constructs a minimal 5-node test graph for hop-band metric tests.
            Node 0 is the home node (x[:, 0] = 1), and positive labels are placed at
            nodes 0 (hop 0, in band 0-2) and 3 (hop 3, in band 3-5).

        Output:
          - (pyg.data.Data): a 5-node graph with x (features), y (labels), edge_index,
                graph_x (demographics), and user_id attributes.
        """
        x = torch.zeros(5, 2)
        x[0, 0] = 1.0  # is_home at column 0, node 0
        x[:, 1] = torch.rand(5)
        y = torch.zeros(5, 1)
        y[0] = 1.0
        y[3] = 1.0
        graph = pyg.data.Data(x=x, edge_index=torch.zeros(2, 0, dtype=torch.long), y=y)
        graph.graph_x = torch.zeros(1, 2)  # demographics
        graph.user_id = torch.tensor([0])
        return graph

    def test_hop_band_keys_logged(self, monkeypatch):
        """
        Description:
            Verifies that when home_hop_distance and is_home_idx are provided to
            ActivityGraphModule, the test metrics include separate per-band keys for
            both band 0-2 and band 3-5 (recall@k, ndcg@k, bce_weighted, n_pos).
            Also verifies that the n_pos count for band 0-2 equals 1 (only node 0).

        Input:
          Setup: a PATH_EDGE_INDEX 5-node graph with home at node 0; _make_graph()
          provides one test graph with positives at nodes 0 and 3.

        Assertions:
          - All expected hop-band metric keys are in logged.keys().
          - All logged hop-band values are finite.
          - test_hop_0-2_n_pos == 1.0 (only node 0 is positive in band 0-2).
        """
        hop = compute_home_hop_distance(PATH_EDGE_INDEX, num_nodes=5)
        model = NodeMLP(num_layers=2, in_channels=4, hidden_channels=8, out_channels=1)
        module = ActivityGraphModule(
            model=model,
            lr=1e-3,
            pos_weight=torch.tensor(1.0),
            home_hop_distance=hop,
            is_home_idx=0,
        )

        logged: dict = {}
        monkeypatch.setattr(module, "log", lambda key, val, **kw: logged.__setitem__(key, val))
        monkeypatch.setattr(module, "log_dict", lambda mapping, **kw: logged.update(mapping))

        batch = pyg.data.Batch.from_data_list([self._make_graph()])
        module.eval()
        with torch.no_grad():
            module.test_step(batch, 0)
        module.on_test_epoch_end()

        k = module.k
        expected = {
            f"test_hop_0-2_recall@{k}",
            f"test_hop_0-2_ndcg@{k}",
            f"test_hop_3-5_recall@{k}",
            "test_hop_0-2_bce_weighted",
            "test_hop_0-2_n_pos",
            "test_hop_3-5_bce_weighted",
        }
        assert expected.issubset(logged.keys())
        assert all(torch.as_tensor(logged[key]).isfinite() for key in expected)
        # node 0 (home) is the only positive in band 0-2
        assert torch.as_tensor(logged["test_hop_0-2_n_pos"]).item() == 1.0

    def test_no_hop_band_metrics_without_hop_distance(self):
        """
        Description:
            Verifies that when home_hop_distance is NOT provided to ActivityGraphModule,
            hop_band_test_metrics is None, meaning no per-band metrics will be computed
            or logged during the test phase.

        Input:
          Setup: ActivityGraphModule with only model, lr, and pos_weight (no hop distance).

        Assertion:
          - module.hop_band_test_metrics is None.
        """
        model = NodeMLP(num_layers=2, in_channels=4, hidden_channels=8, out_channels=1)
        module = ActivityGraphModule(model=model, lr=1e-3, pos_weight=torch.tensor(1.0))
        assert module.hop_band_test_metrics is None
