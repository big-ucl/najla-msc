import itertools

import torch
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.nn.conv import GATConv, GCNConv


def build_module_list(
    module_f: torch.nn.Module,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    hidden_channels: int | None = None,
    **module_kwargs,
) -> torch.nn.ModuleList:
    if num_layers > 1 and hidden_channels is None:
        raise ValueError(f"Hidden channels not provided for {num_layers=} and {hidden_channels=}")

    if num_layers < 1:
        raise ValueError(f"Invalid number of layers: {num_layers=}")

    layers = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
    modules = torch.nn.ModuleList()

    for num_in, num_out in itertools.pairwise(layers):
        modules.append(module_f(num_in, num_out, **module_kwargs))

    return modules


class GCN(torch.nn.Module):
    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.2,
        residuals: bool = False,
        conv: pyg.nn.conv.MessagePassing = None,
    ):
        super().__init__()

        conv = GCNConv if conv is None else conv

        if residuals and in_channels != hidden_channels:
            raise ValueError("Number of in channels must match number of hidden channels")

        self.residuals = residuals
        self.dropout = dropout
        self.convs = build_module_list(conv, num_layers, in_channels, out_channels, hidden_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs[:-1]:
            x_res = x
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)

            if self.residuals:
                x = x + x_res

            x = F.leaky_relu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)

        return x


class NodeMLP(torch.nn.Module):
    def __init__(
        self, num_layers: int, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.2
    ):
        super().__init__()

        self.dropout = dropout
        self.lins = build_module_list(torch.nn.Linear, num_layers, in_channels, out_channels, hidden_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for lin in self.lins[:-1]:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = lin(x).relu()

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)

        return x


class GCNPlus(torch.nn.Module):
    def __init__(
        self, num_gcn: int, num_lin: int, in_channels: int, hidden_channels: int, out_channels: int, dropout=0.2
    ):
        super().__init__()

        self.gcn = GCN(num_gcn, in_channels, hidden_channels, hidden_channels, dropout)
        self.lin = NodeMLP(num_lin, hidden_channels, hidden_channels, out_channels, dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.gcn(x, edge_index).relu()
        x = self.lin(x, edge_index)

        return x


class GCNRes(torch.nn.Module):
    def __init__(
        self,
        num_pre_layers: int,
        num_gcn_layers: int,
        num_post_layers: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.dropout = dropout
        self.pre_lin = NodeMLP(num_pre_layers, in_channels, hidden_channels, hidden_channels, dropout)
        self.convs = GCN(num_gcn_layers, hidden_channels, hidden_channels, hidden_channels, dropout, residuals=True)
        self.post_lin = NodeMLP(num_post_layers, hidden_channels, hidden_channels, out_channels, dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.pre_lin(x, edge_index)
        x = self.convs(x, edge_index)
        x = self.post_lin(x, edge_index)

        return x


class GCNSkip(torch.nn.Module):
    def __init__(
        self,
        num_pre_layers: int,
        num_gcn_layers: int,
        num_post_layers: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.2,
        conv: pyg.nn.conv.MessagePassing | None = None,
    ):
        super().__init__()

        self.dropout = dropout
        self.pre_lin = NodeMLP(num_pre_layers, in_channels, hidden_channels, hidden_channels, dropout)
        self.convs = GCN(
            num_gcn_layers, hidden_channels, hidden_channels, hidden_channels, dropout, residuals=True, conv=conv
        )
        self.post_lin = NodeMLP(num_post_layers, hidden_channels + in_channels, hidden_channels, out_channels, dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x_skip = x

        x = self.pre_lin(x, edge_index)
        x = self.convs(x, edge_index)

        x = torch.cat([x, x_skip], dim=1)
        x = self.post_lin(x, edge_index)

        return x


class GATSkip(GCNSkip):
    def __init__(
        self,
        num_pre_layers: int,
        num_gcn_layers: int,
        num_post_layers: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.2,
    ):
        super().__init__(
            num_pre_layers,
            num_gcn_layers,
            num_post_layers,
            in_channels,
            hidden_channels,
            out_channels,
            dropout,
            conv=GATConv,
        )
