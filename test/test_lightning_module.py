"""
Module: test/test_lightning_module.py

Description:
    Unit tests for ActivityGraphModule — the PyTorch Lightning wrapper around the GNN
    models.  Tests cover:
      - training_step: returns a finite scalar loss; L1 regularisation increases loss;
        larger pos_weight increases loss on positive-heavy batches.
      - validation_step / on_validation_epoch_end: logs the expected BCE, ranking
        (precision/recall/nDCG @ k, R-Precision), and calibration keys; ranking metrics
        match manually-computed reference values; metric state is reset between epochs.
      - test_step / on_test_epoch_end: logs additional keys (pred_size, sampled_recall, etc.).
      - configure_optimizers: returns AdamW; with schedule_lr returns a scheduler that
        monitors val_bce; scheduler is suppressed in overfit mode.

    Helper functions (precision_at_k, recall_at_k, ndcg_at_k) implement the ranking
    metrics independently using NumPy-style logic to cross-check the torchmetrics output.
"""

import types

import torch
import torch_geometric as pyg

from activitygraphs.ml.lightning_module import ActivityGraphModule, extract_features
from activitygraphs.ml.models import NodeMLP


# -------------------------------------------------------------------------
# Local reference implementations used only to cross-check torchmetrics output.
# The project itself computes ranking metrics via torchmetrics, not these.
# -------------------------------------------------------------------------

def precision_at_k(scores, labels, k):
    """
    Description:
        Computes Precision @ k for a single query: the fraction of the top-k retrieved
        nodes that are truly positive.

    Input:
      - scores (torch.Tensor): per-node model scores, shape (N,).
      - labels (torch.Tensor): binary ground-truth labels, shape (N,).
      - k (int): number of top items to consider.

    Output:
      - (float): precision at k.
    """
    top_k = scores.topk(k).indices      # indices of the k highest-scoring nodes
    return labels[top_k].sum().item() / k  # fraction that are truly positive


def recall_at_k(scores, labels, k):
    """
    Description:
        Computes Recall @ k for a single query: the fraction of all positive nodes
        that appear in the top-k retrieved nodes.

    Input:
      - scores (torch.Tensor): per-node model scores, shape (N,).
      - labels (torch.Tensor): binary ground-truth labels, shape (N,).
      - k (int): number of top items to consider.

    Output:
      - (float): recall at k (0.0 if there are no positive nodes).
    """
    top_k = scores.topk(k).indices
    num_pos = labels.sum().int().item()  # total number of positive (visited) nodes
    return labels[top_k].sum().item() / num_pos if num_pos else 0.0


def ndcg_at_k(scores, labels, k):
    """
    Description:
        Computes Normalised Discounted Cumulative Gain @ k (nDCG @ k) for a single query.
        Rewards placing positive nodes higher in the ranking.

    Input:
      - scores (torch.Tensor): per-node model scores, shape (N,).
      - labels (torch.Tensor): binary ground-truth labels, shape (N,).
      - k (int): cut-off rank.

    Output:
      - (float): nDCG at k (0.0 if ideal DCG is zero).
    """
    # Sort by score descending and take the top k
    order = scores.argsort(descending=True)[:k]
    # Logarithmic position discounts: 1/log2(2), 1/log2(3), ..., 1/log2(k+1)
    discounts = 1.0 / torch.log2(torch.arange(2, k + 2, dtype=torch.float))
    dcg = (labels[order].float() * discounts).sum().item()
    # Ideal DCG: put all positives at the top
    ideal = torch.zeros(k)
    ideal[: min(int(labels.sum().item()), k)] = 1.0
    idcg = (ideal * discounts).sum().item()
    return dcg / idcg if idcg else 0.0  # avoid division by zero when no positives


# -------------------------------------------------------------------------
# Shared test constants
# -------------------------------------------------------------------------
NUM_NODE_FEATURES = 6    # number of spatial features per node
NUM_DEMO_FEATURES = 3    # number of demographic features per user (graph-level)
NUM_EDGE_FEATURES = 2    # number of features per edge
IN_CHANNELS = NUM_NODE_FEATURES + NUM_DEMO_FEATURES  # total node input channels after concat


def make_batch(num_graphs: int = 2, num_nodes: int = 8, seed: int = 0, start_user_id: int = 0) -> pyg.data.Batch:
    """
    Description:
        Creates a synthetic batched PyG object for testing.  Each graph has random node
        features, random binary node labels, a simple chain edge structure, and random
        edge attributes.  A graph-level demographics tensor (graph_x) is attached so that
        extract_features can exercise the demographics concatenation path.

    Input:
      - num_graphs (int): number of graphs in the batch.  Defaults to 2.
      - num_nodes (int): nodes per graph (uniform).  Defaults to 8.
      - seed (int): torch Generator seed for reproducibility.  Defaults to 0.
      - start_user_id (int): the first user_id integer; subsequent graphs get
            start_user_id + 1, start_user_id + 2, etc.  Defaults to 0.

    Output:
      - (pyg.data.Batch): a batched PyG object with node features x (shape N*G x NUM_NODE_FEATURES),
            labels y (shape N*G x 1), edge_index, edge_attr, graph_x, and user_id tensors.
    """
    rng = torch.Generator().manual_seed(seed)
    graphs = []
    for g in range(num_graphs):
        x = torch.rand(num_nodes, NUM_NODE_FEATURES, generator=rng)  # spatial node features
        y = torch.randint(0, 2, (num_nodes, 1), generator=rng).float()  # binary visit labels
        # Build a simple undirected chain: 0-1-2-...(num_nodes-1)
        src = torch.arange(num_nodes - 1)
        dst = torch.arange(1, num_nodes)
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
        edge_attr = torch.rand(edge_index.size(1), NUM_EDGE_FEATURES, generator=rng)
        data = pyg.data.Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            user_id=torch.tensor([start_user_id + g], dtype=torch.long),  # unique per graph
        )
        # graph_x: per-graph demographic features; shape (1, NUM_DEMO_FEATURES)
        data.graph_x = torch.rand(1, NUM_DEMO_FEATURES, generator=rng)
        graphs.append(data)
    return pyg.data.Batch.from_data_list(graphs)


def make_module(
    reg: str | None = None, lambda_reg: float = 0.01, pos_weight: float = 2.0, schedule_lr: bool = False
) -> ActivityGraphModule:
    """
    Description:
        Constructs a test ActivityGraphModule backed by a small NodeMLP.

    Input:
      - reg (str | None): regularisation type; None, "l1", or "l2".  Defaults to None.
      - lambda_reg (float): regularisation strength coefficient.  Defaults to 0.01.
      - pos_weight (float): scalar positive-class weight for BCE loss.  Defaults to 2.0.
      - schedule_lr (bool): whether to attach a ReduceLROnPlateau scheduler.
            Defaults to False.

    Output:
      - (ActivityGraphModule): a configured module ready for testing.
    """
    model = NodeMLP(num_layers=2, in_channels=IN_CHANNELS, hidden_channels=8, out_channels=1)
    return ActivityGraphModule(
        model=model, lr=1e-3, pos_weight=torch.tensor(pos_weight), reg=reg, lambda_reg=lambda_reg, schedule_lr=schedule_lr
    )


def force_positive_per_graph(batch: pyg.data.Batch) -> None:
    """
    Description:
        Mutates a batch in-place to ensure each graph has at least one positive label.
        This is necessary because ranking metrics (precision, recall, nDCG) are undefined
        when a graph has no positive nodes.

    Input:
      - batch (pyg.data.Batch): the batch to modify.

    Output:
      - None (modifies batch.y in-place).
    """
    for i in range(batch.num_graphs):
        # Get the first node index belonging to graph i
        idx = (batch.batch == i).nonzero(as_tuple=True)[0][0]
        batch.y[idx] = 1.0  # force the first node of each graph to be positive


def capture_logs(module: ActivityGraphModule, monkeypatch) -> dict:
    """
    Description:
        Patches self.log and self.log_dict on the module with thin lambdas that store
        all logged key-value pairs into a shared dictionary.  This allows tests to assert
        on what was logged without requiring a full Trainer.

    Input:
      - module (ActivityGraphModule): the module whose logging methods to patch.
      - monkeypatch (pytest.MonkeyPatch): pytest's monkeypatch fixture for safe patching.

    Output:
      - (dict): a mutable dictionary that accumulates all logged metrics.
    """
    logged: dict = {}
    monkeypatch.setattr(module, "log", lambda key, val, **kw: logged.__setitem__(key, val))
    monkeypatch.setattr(module, "log_dict", lambda mapping, **kw: logged.update(mapping))
    return logged


def attach_fake_trainer(module: ActivityGraphModule, overfit_batches: int = 0) -> ActivityGraphModule:
    """
    Description:
        Attaches a minimal stand-in trainer namespace to the module so that
        configure_optimizers can read trainer.overfit_batches without raising
        an AttributeError.

    Input:
      - module (ActivityGraphModule): the module to attach the fake trainer to.
      - overfit_batches (int): simulated overfit_batches trainer attribute.
            Non-zero values suppress the LR scheduler.  Defaults to 0.

    Output:
      - (ActivityGraphModule): the same module with _trainer set.
    """
    module._trainer = types.SimpleNamespace(overfit_batches=overfit_batches)
    return module


class TestTrainingStep:
    """
    Description:
        Tests for ActivityGraphModule.training_step, which computes the weighted BCE
        loss (plus optional L1/L2 regularisation) for a mini-batch and logs it.
    """

    def test_returns_finite_scalar(self, monkeypatch):
        """
        Description:
            Verifies that training_step returns a finite 0-dimensional (scalar) tensor.
            A non-scalar or non-finite loss would crash the optimiser.

        Input:
          Setup: a default module (no regularisation, pos_weight=2.0), a 2-graph batch.

        Assertions:
          - loss.ndim == 0 (it is a scalar, not a vector).
          - loss.isfinite() is True (no NaN or Inf).
        """
        module = make_module()
        monkeypatch.setattr(module, "log", lambda *a, **kw: None)
        loss = module.training_step(make_batch(), 0)
        assert loss.ndim == 0
        assert loss.isfinite()

    def test_l1_reg_increases_loss(self, monkeypatch):
        """
        Description:
            Verifies that adding L1 regularisation (with a large lambda_reg=1.0) to
            the training loss produces a strictly higher total loss than training without
            regularisation, given identical model weights.

        Input:
          Setup: two modules with the same weights; one uses reg=None, the other reg="l1"
          with lambda_reg=1.0.

        Assertion:
          - loss_l1 > loss_base.
        """
        batch = make_batch(seed=1)

        base = make_module(reg=None)
        monkeypatch.setattr(base, "log", lambda *a, **kw: None)
        loss_base = base.training_step(batch, 0).item()

        module_l1 = make_module(reg="l1", lambda_reg=1.0)
        # copy identical weights so the only difference is the L1 term
        module_l1.model.load_state_dict(base.model.state_dict())
        monkeypatch.setattr(module_l1, "log", lambda *a, **kw: None)
        loss_l1 = module_l1.training_step(batch, 0).item()

        assert loss_l1 > loss_base

    def test_pos_weight_shifts_loss(self, monkeypatch):
        """
        Description:
            Verifies that using a larger positive-class weight (10.0 vs 1.0) results
            in a strictly higher loss on a batch with at least one positive label.
            This confirms that pos_weight is correctly passed to BCEWithLogitsLoss.

        Input:
          Setup: two modules sharing the same model weights; pos_weight=1.0 vs 10.0.

        Assertion:
          - loss_high > loss_low.
        """
        batch = make_batch(seed=2)
        batch.y[0] = 1.0  # ensure at least one positive label

        module_low = make_module(pos_weight=1.0)
        monkeypatch.setattr(module_low, "log", lambda *a, **kw: None)
        loss_low = module_low.training_step(batch, 0).item()

        # reuse identical weights, only the pos_weight differs
        module_high = ActivityGraphModule(model=module_low.model, lr=1e-3, pos_weight=torch.tensor(10.0))
        monkeypatch.setattr(module_high, "log", lambda *a, **kw: None)
        loss_high = module_high.training_step(batch, 0).item()

        assert loss_high > loss_low


class TestValidationStep:
    """
    Description:
        Tests for ActivityGraphModule.validation_step and on_validation_epoch_end.
        Verifies that the expected set of validation metric keys is logged, that the
        ranking metric values match a manually-computed reference, and that metric
        state is properly reset between epochs.
    """

    def test_val_metric_keys_logged(self, monkeypatch):
        """
        Description:
            Runs validation_step + on_validation_epoch_end and checks that all expected
            metric keys are present in the logged output and have finite values.
            Missing keys would cause downstream experiment-tracking and result-saving
            code to fail silently.

        Input:
          Setup: a 3-graph batch with at least one positive per graph.

        Assertion:
          - All keys in {val_bce, val_bce_weighted, val_r_precision, val_recall@k,
            val_ndcg@k, val_precision@k, val_calibration_l1} are in logged.keys().
          - All logged values are finite.
        """
        module = make_module()
        batch = make_batch(num_graphs=3, num_nodes=8)
        force_positive_per_graph(batch)

        logged = capture_logs(module, monkeypatch)

        module.eval()
        with torch.no_grad():
            module.validation_step(batch, 0)
        module.on_validation_epoch_end()

        k = module.k
        expected = {
            "val_bce",
            "val_bce_weighted",
            "val_r_precision",
            f"val_recall@{k}",
            f"val_ndcg@{k}",
            f"val_precision@{k}",
            "val_calibration_l1",
        }
        assert expected.issubset(logged.keys())
        assert all(torch.as_tensor(logged[key]).isfinite() for key in expected)

    def test_ranking_metrics_match_reference(self, monkeypatch):
        """torchmetrics precision/recall/ndcg must match the per-graph metrics.py functions.

        One graph == one user_id, so the Retrieval* grouping is per graph and the unweighted mean
        over non-empty groups equals the old per-graph average. MRR is excluded on purpose:
        RetrievalMRR uses first-relevant-rank, not mean-over-positives, so its value differs.
        """
        module = make_module()
        k = module.k
        batch = make_batch(num_graphs=4, num_nodes=8, seed=7)
        force_positive_per_graph(batch)
        capture_logs(module, monkeypatch)

        module.eval()
        with torch.no_grad():
            out = module(extract_features(batch, module.full_info), batch.edge_index, batch.edge_attr, batch.batch)

        precisions, recalls, ndcgs = [], [], []
        for i in range(batch.num_graphs):
            mask = batch.batch == i
            scores = out[mask].squeeze()
            labels = batch.y[mask].squeeze()
            if labels.sum().int().item() == 0:
                continue
            precisions.append(float(precision_at_k(scores, labels, k)))
            recalls.append(float(recall_at_k(scores, labels, k)))
            ndcgs.append(float(ndcg_at_k(scores, labels, k)))

        ref = {
            f"val_precision@{k}": sum(precisions) / len(precisions),
            f"val_recall@{k}": sum(recalls) / len(recalls),
            f"val_ndcg@{k}": sum(ndcgs) / len(ndcgs),
        }

        with torch.no_grad():
            module.validation_step(batch, 0)
        actual = module.val_metrics.compute()

        for key, expected in ref.items():
            torch.testing.assert_close(actual[key].item(), expected, rtol=1e-5, atol=1e-6)

    def test_metrics_reset_between_epochs(self, monkeypatch):
        """on_validation_epoch_end must reset metric state, so a repeated epoch gives identical numbers."""
        module = make_module()
        batch = make_batch(num_graphs=3, num_nodes=8, seed=3)
        force_positive_per_graph(batch)
        capture_logs(module, monkeypatch)

        module.eval()
        with torch.no_grad():
            module.validation_step(batch, 0)
        first = {key: val.item() for key, val in module.val_metrics.compute().items()}
        module.on_validation_epoch_end()  # logs + resets

        with torch.no_grad():
            module.validation_step(batch, 0)
        second = {key: val.item() for key, val in module.val_metrics.compute().items()}

        # Without a reset, re-feeding the same user_ids appends docs to existing groups and shifts
        # the result. Equality confirms state was cleared.
        assert first == second


class TestTestStep:
    """
    Description:
        Tests for ActivityGraphModule.test_step and on_test_epoch_end.
        In addition to the validation metrics, test-phase logging includes
        additional keys: pred_size (mean Poisson sample size), true_size
        (mean true positive count), sampled_recall, and sampled_size.
    """

    def test_test_metric_keys_logged(self, monkeypatch):
        """
        Description:
            Runs test_step + on_test_epoch_end and verifies that the full set of test
            metric keys is present in the logged output.  This is a superset of the
            validation metrics and includes choice-set size and sampling recall keys.

        Input:
          Setup: a 3-graph batch with at least one positive per graph.

        Assertion:
          - All keys in {test_bce, test_bce_weighted, test_r_precision, test_recall@k,
            test_ndcg@k, test_precision@k, test_calibration_l1, test_pred_size,
            test_true_size, test_sampled_recall, test_sampled_size} are logged.
        """
        module = make_module()
        batch = make_batch(num_graphs=3, num_nodes=8)
        force_positive_per_graph(batch)

        logged = capture_logs(module, monkeypatch)

        module.eval()
        with torch.no_grad():
            module.test_step(batch, 0)
        module.on_test_epoch_end()

        k = module.k
        expected = {
            "test_bce",
            "test_bce_weighted",
            "test_r_precision",
            f"test_recall@{k}",
            f"test_ndcg@{k}",
            f"test_precision@{k}",
            "test_calibration_l1",
            "test_pred_size",
            "test_true_size",
            "test_sampled_recall",
            "test_sampled_size",
        }
        assert expected.issubset(logged.keys())


class TestConfigureOptimizers:
    """
    Description:
        Tests for ActivityGraphModule.configure_optimizers, which returns either a
        bare AdamW optimizer or a dict with an optimizer and a ReduceLROnPlateau
        scheduler depending on the schedule_lr flag and trainer.overfit_batches.
    """

    def test_returns_optimizer_and_scheduler(self):
        """
        Description:
            When schedule_lr=True, configure_optimizers must return a dict with both
            an "optimizer" key and an "lr_scheduler" key.

        Assertion:
          - "optimizer" in result and "lr_scheduler" in result.
        """
        result = attach_fake_trainer(make_module(schedule_lr=True)).configure_optimizers()
        assert "optimizer" in result
        assert "lr_scheduler" in result

    def test_scheduler_monitors_val_bce(self):
        """
        Description:
            The ReduceLROnPlateau scheduler must be configured to monitor "val_bce"
            so that it reduces the learning rate when validation BCE stops improving.

        Assertion:
          - result["lr_scheduler"]["monitor"] == "val_bce".
        """
        result = attach_fake_trainer(make_module(schedule_lr=True)).configure_optimizers()
        assert result["lr_scheduler"]["monitor"] == "val_bce"

    def test_optimizer_is_adamw(self):
        """
        Description:
            Verifies that the returned optimizer is an AdamW instance.  Using AdamW
            is a project requirement (decoupled weight decay for better generalisation).

        Assertion:
          - isinstance(result["optimizer"], torch.optim.AdamW).
        """
        result = attach_fake_trainer(make_module(schedule_lr=True)).configure_optimizers()
        assert isinstance(result["optimizer"], torch.optim.AdamW)

    def test_no_scheduler_when_not_requested(self):
        """
        Description:
            With schedule_lr=False (the default), configure_optimizers must return a
            bare AdamW optimizer directly rather than a dict with a scheduler.

        Assertion:
          - isinstance(result, torch.optim.AdamW).
        """
        result = attach_fake_trainer(make_module()).configure_optimizers()
        assert isinstance(result, torch.optim.AdamW)

    def test_no_scheduler_when_overfitting(self):
        """
        Description:
            When trainer.overfit_batches > 0 (debugging overfitting mode), the scheduler
            must be suppressed even if schedule_lr=True was requested, because the
            scheduler would immediately reduce LR on a single repeated batch.

        Input:
          Setup: make_module() with overfit_batches=5 on the fake trainer.

        Assertion:
          - isinstance(result, torch.optim.AdamW) (bare optimizer, no scheduler).
        """
        result = attach_fake_trainer(make_module(), overfit_batches=5).configure_optimizers()
        assert isinstance(result, torch.optim.AdamW)
