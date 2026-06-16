"""
Module: test/test_baselines.py

Description:
    Unit tests for the frequency-based baseline models NodeBaseline and
    ConditionalNodeBaseline, and for the helper function compute_ptr_from_batch.

    NodeBaseline learns per-node visit frequencies from the training set and uses them
    as logit predictions (no graph structure used).

    ConditionalNodeBaseline conditions the frequency on which node is the home node
    (determined by a binary feature column).

    compute_ptr_from_batch replicates PyG's built-in .ptr attribute from the .batch
    assignment vector.
"""

import torch
import torch_geometric as pyg
import pytest

from activitygraphs.ml.baselines import (
    NodeBaseline,
    ConditionalNodeBaseline,
    compute_ptr_from_batch,
)


def make_graphs(num_graphs: int, num_nodes: int, num_features: int, seed: int = 0) -> list[pyg.data.Data]:
    """
    Description:
        Creates a list of random synthetic PyG graphs for use in tests.  Each graph
        has random node features, random binary labels, and no edges (so tests focus
        purely on node-level behaviour).

    Input:
      - num_graphs (int): how many graphs to generate.
      - num_nodes (int): number of nodes per graph (all graphs have the same size).
      - num_features (int): number of node features per node.
      - seed (int): seed for the torch Generator to ensure reproducibility.  Defaults to 0.

    Output:
      - (list[pyg.data.Data]): a list of PyG Data objects.
    """
    rng = torch.Generator().manual_seed(seed)  # seeded RNG for reproducible random tensors
    graphs = []
    for _ in range(num_graphs):
        x = torch.rand(num_nodes, num_features, generator=rng)       # random node features
        y = torch.randint(0, 2, (num_nodes, 1), generator=rng).float()  # random binary labels
        edge_index = torch.zeros(2, 0, dtype=torch.long)              # no edges
        graphs.append(pyg.data.Data(x=x, edge_index=edge_index, y=y))
    return graphs


@pytest.fixture
def batch_uniform():
    """
    Description:
        Pytest fixture providing a batched PyG object containing 4 graphs, each
        with 5 nodes.  Used for tests that assume uniform graph size.

    Output:
      - (pyg.data.Batch): a batch of 4 graphs, each with 5 nodes and 3 features.
    """
    graphs = make_graphs(num_graphs=4, num_nodes=5, num_features=3)
    return pyg.data.Batch.from_data_list(graphs)


@pytest.fixture
def batch_variable():
    """
    Description:
        Pytest fixture providing a batched PyG object containing graphs with
        different node counts (3, 5, 2).  Used to test variable-size behaviour.

    Output:
      - (pyg.data.Batch): a batch of 3 graphs with 3, 5, and 2 nodes respectively.
    """
    rng = torch.Generator().manual_seed(42)  # fixed seed for reproducibility
    graphs = []
    for n in [3, 5, 2]:  # graphs of different sizes
        x = torch.rand(n, 3, generator=rng)
        y = torch.randint(0, 2, (n, 1), generator=rng).float()
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        graphs.append(pyg.data.Data(x=x, edge_index=edge_index, y=y))
    return pyg.data.Batch.from_data_list(graphs)


class TestComputePtrFromBatch:
    """
    Description:
        Tests for the compute_ptr_from_batch helper, which reconstructs the PyG
        batch pointer tensor (ptr) from the batch assignment vector.  The ptr tensor
        contains the cumulative node count at the start of each graph (plus a final
        sentinel equal to the total number of nodes).
    """

    def test_matches_pyg_ptr_uniform(self, batch_uniform):
        """
        Description:
            Verifies that the recomputed ptr matches PyG's built-in ptr attribute
            for a batch of identically-sized (uniform) graphs.

        Input:
          - batch_uniform (pyg.data.Batch): 4 graphs each with 5 nodes.

        Assertion:
          - compute_ptr_from_batch(batch.batch) == batch.ptr element-wise.
        """
        computed = compute_ptr_from_batch(batch_uniform.batch)
        assert torch.equal(computed, batch_uniform.ptr)

    def test_matches_pyg_ptr_variable(self, batch_variable):
        """
        Description:
            Verifies that the recomputed ptr matches PyG's built-in ptr for a batch
            of graphs with different node counts (3, 5, 2 nodes).

        Input:
          - batch_variable (pyg.data.Batch): 3 graphs with variable node counts.

        Assertion:
          - compute_ptr_from_batch(batch.batch) == batch.ptr element-wise.
        """
        computed = compute_ptr_from_batch(batch_variable.batch)
        assert torch.equal(computed, batch_variable.ptr)

    def test_single_graph(self):
        """
        Description:
            Edge case: verifies that a batch vector of all zeros (single graph with 7
            nodes) produces ptr = [0, 7].

        Input:
          - batch: a manually constructed single-graph batch tensor.

        Assertion:
          - ptr == tensor([0, 7]).
        """
        batch = torch.zeros(7, dtype=torch.long)  # 7 nodes, all in graph 0
        computed = compute_ptr_from_batch(batch)
        assert torch.equal(computed, torch.tensor([0, 7]))

    def test_ptr_length(self, batch_uniform):
        """
        Description:
            Verifies that the ptr tensor has exactly num_graphs + 1 elements (one
            boundary per graph plus the final sentinel).

        Input:
          - batch_uniform (pyg.data.Batch): 4 uniform graphs.

        Assertion:
          - len(ptr) == num_graphs + 1.
        """
        computed = compute_ptr_from_batch(batch_uniform.batch)
        assert len(computed) == batch_uniform.num_graphs + 1

    def test_ptr_starts_at_zero(self, batch_uniform):
        """
        Description:
            Verifies that the first element of ptr is always 0 (first graph starts at node 0).

        Input:
          - batch_uniform (pyg.data.Batch): 4 uniform graphs.

        Assertion:
          - ptr[0] == 0.
        """
        computed = compute_ptr_from_batch(batch_uniform.batch)
        assert computed[0].item() == 0

    def test_ptr_ends_at_num_nodes(self, batch_uniform):
        """
        Description:
            Verifies that the last element of ptr equals the total number of nodes in
            the batch (the final sentinel value).

        Input:
          - batch_uniform (pyg.data.Batch): 4 uniform graphs.

        Assertion:
          - ptr[-1] == batch.num_nodes.
        """
        computed = compute_ptr_from_batch(batch_uniform.batch)
        assert computed[-1].item() == batch_uniform.num_nodes


class TestNodeBaseline:
    """
    Description:
        Tests for NodeBaseline, a non-graph frequency baseline model that assigns
        each node a logit score based on its empirical visit frequency (per node
        position) observed across the training set.
    """

    def test_node_indices_match_within_graph_position(self, batch_uniform):
        """
        Description:
            Verifies that subtracting each node's graph-start pointer from its global
            index correctly recovers the within-graph position (0..num_nodes-1).
            This is the internal indexing logic NodeBaseline relies on to look up
            the per-node frequency table.

        Input:
          - batch_uniform (pyg.data.Batch): 4 graphs, each with 5 nodes.

        Assertion:
          - node_indices (global_index - ptr[graph_id]) == arange(total) % 5.
        """
        ptr = compute_ptr_from_batch(batch_uniform.batch)
        node_indices = torch.arange(batch_uniform.num_nodes) - ptr[batch_uniform.batch]
        expected = torch.arange(batch_uniform.num_nodes) % 5  # uniform graphs of size 5
        assert torch.equal(node_indices, expected)

    def test_forward_shape(self, batch_uniform):
        """
        Description:
            Verifies that the baseline's forward method returns a tensor of shape
            (num_total_nodes, 1) as expected by the loss function.

        Input:
          - batch_uniform (pyg.data.Batch): 4 graphs, each with 5 nodes.
          Setup: a NodeBaseline fitted on 8 graphs.

        Assertion:
          - output.shape == (20, 1).
        """
        num_nodes = 5
        baseline = NodeBaseline(num_nodes=num_nodes)
        loader = pyg.loader.DataLoader(make_graphs(8, num_nodes, 3), batch_size=4)
        baseline.fit(loader)
        out = baseline(batch_uniform.x, batch_uniform.edge_index, batch=batch_uniform.batch)
        assert out.shape == (batch_uniform.num_nodes, 1)

    def test_forward_values_in_logit_range(self, batch_uniform):
        """
        Description:
            Verifies that all output logits are finite (no NaN or Inf) after fitting.
            NaN logits could arise if any node position was never observed in training.

        Input:
          - batch_uniform (pyg.data.Batch): 4 graphs, each with 5 nodes.
          Setup: a NodeBaseline fitted on 8 graphs.

        Assertion:
          - torch.isfinite(output).all() is True.
        """
        num_nodes = 5
        baseline = NodeBaseline(num_nodes=num_nodes)
        loader = pyg.loader.DataLoader(make_graphs(8, num_nodes, 3), batch_size=4)
        baseline.fit(loader)
        out = baseline(batch_uniform.x, batch_uniform.edge_index, batch=batch_uniform.batch)
        assert torch.isfinite(out).all()


class TestConditionalNodeBaseline:
    """
    Description:
        Tests for ConditionalNodeBaseline, which conditions per-node visit frequency
        estimates on which node is designated as the home node (identified by a binary
        indicator column in the feature matrix).
    """

    def test_forward_shape(self, batch_uniform):
        """
        Description:
            Verifies that ConditionalNodeBaseline's forward method returns a tensor of
            shape (num_total_nodes, 1).  The baseline is set up with a synthetic feature
            matrix where column 2 is the home indicator (1 only at node 0 per graph).

        Input:
          - batch_uniform: provided by fixture but not directly used here.
          Setup: 8 synthetic graphs with node 0 marked as home via feature column 2.

        Assertion:
          - output.shape == (4 * 5, 1) for the 4-graph test batch.
        """
        num_nodes = 5
        # Use a synthetic feature matrix where a chosen column acts as the home indicator.
        is_home_idx = 2  # column 2 of x is the binary home indicator
        num_features = 3
        graphs = make_graphs(num_graphs=8, num_nodes=num_nodes, num_features=num_features)
        for g in graphs:
            g.x[:, is_home_idx] = 0.0   # clear any random values in the home column
            g.x[0, is_home_idx] = 1.0   # mark node 0 as home for all graphs

        loader = pyg.loader.DataLoader(graphs, batch_size=4)
        # Use only the first 4 graphs for the forward-pass batch (matches batch_size=4)
        batched = pyg.data.Batch.from_data_list(graphs[:4])

        baseline = ConditionalNodeBaseline(num_nodes=num_nodes, is_home_idx=is_home_idx)
        baseline.fit(loader)
        out = baseline(batched.x, batched.edge_index, batch=batched.batch)
        assert out.shape == (batched.num_nodes, 1)
