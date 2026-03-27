import torch
import torch_geometric as pyg

from activitygraphs.base import IS_HOME_COL_IDX


def inverse_sigmoid(prob):
    return torch.log(prob / (1 - prob))


def extract_is_home(x: torch.Tensor) -> torch.Tensor:
    return (x[..., IS_HOME_COL_IDX] > 0.0).bool()


class UniformBaseline(torch.nn.Module):
    """P(y_i = 1) = 0.5 for all nodes."""

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        return torch.zeros(x.shape[0], 1, device=x.device)


class GlobalBaseline(torch.nn.Module):
    """P(y_i = 1) = global visit rate across all training graphs."""

    def __init__(self):
        super().__init__()
        self.logit = None

    def fit(self, loader: pyg.loader.DataLoader):
        num_pos = 0
        num_total = 0

        for batch in loader:
            num_pos += batch.y.sum().item()
            num_total += batch.y.numel()

        p = num_pos / num_total
        self.logit = inverse_sigmoid(torch.tensor(p))

        return self

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        return self.logit.expand(x.shape[0], 1).to(x.device)


class NodeBaseline(torch.nn.Module):
    """P(y_i = 1) = per-node visit frequency across all training graphs."""

    def __init__(self, num_nodes: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.logits = None

    def fit(self, loader: pyg.loader.DataLoader):
        visit_counts = torch.zeros(self.num_nodes)
        graph_counts = torch.zeros(self.num_nodes)

        for batch in loader:
            node_indices = torch.arange(batch.num_nodes) % self.num_nodes
            visit_counts.scatter_add_(0, node_indices, batch.y.float().squeeze().cpu())
            graph_counts.scatter_add_(0, node_indices, torch.ones(batch.num_nodes))

        p = (visit_counts / graph_counts.clamp(min=1)).clamp(1e-6, 1 - 1e-6)
        self.logits = inverse_sigmoid(p)

        return self

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        node_indices = torch.arange(x.shape[0], device=x.device) % self.num_nodes
        return self.logits.to(x.device)[node_indices].unsqueeze(1).to(x.device)


class ConditionalNodeBaseline(torch.nn.Module):
    """P(y_i = 1 | home = h) = per-node visit frequency conditioned on home node."""

    def __init__(self, num_nodes: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.logits = None  # shape: [num_nodes, num_nodes] (node, home)

    def fit(self, loader: pyg.loader.DataLoader):
        visit_counts = torch.zeros(self.num_nodes, self.num_nodes)
        graph_counts = torch.zeros(self.num_nodes)

        for batch in loader:
            node_indices = torch.arange(batch.num_nodes) % self.num_nodes
            home_mask = extract_is_home(batch.x)

            for i in range(batch.num_graphs):
                graph_mask = batch.batch == i
                home_nodes = home_mask[graph_mask].nonzero()

                if len(home_nodes) == 0:
                    continue

                home = home_nodes[0].item()
                visit_counts[node_indices[graph_mask], home] += batch.y[graph_mask].squeeze().float().cpu()
                graph_counts[home] += 1

        p = visit_counts / graph_counts.clamp(min=1).unsqueeze(0)
        p = p.clamp(1e-6, 1 - 1e-6)
        self.logits = inverse_sigmoid(p)

        return self

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        node_indices = torch.arange(x.shape[0], device=x.device) % self.num_nodes
        home_mask = extract_is_home(x)
        home_indices = torch.zeros(batch.max().item() + 1, dtype=torch.long, device=x.device)

        for i in range(batch.max().item() + 1):
            graph_mask = batch == i
            home_nodes = home_mask[graph_mask].nonzero()

            home_indices[i] = 0 if len(home_nodes) == 0 else home_nodes[0].item()

        node_home = home_indices[batch]
        return self.logits.to(x.device)[node_indices, node_home.cpu()].unsqueeze(1).to(x.device)
