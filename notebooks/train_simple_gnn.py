import marimo

__generated_with = "0.19.8"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    import torch

    from pathlib import Path

    from activitygraphs.config import load_config

    project_root = Path(mo.notebook_dir().parent)
    cfg = load_config(project_root)


@app.cell
def _():
    from activitygraphs.geometric import ActivityDataset

    dataset = ActivityDataset.from_files(cfg.data, project_root)
    dataset
    return (dataset,)


@app.cell
def _():
    import torch.nn.functional as F
    from torch_geometric.nn import HeteroConv, Linear, SAGEConv


    class HeteroGNN(torch.nn.Module):
        def __init__(self, metadata, hidden_channels, out_channels, num_layers, targets=None):
            super().__init__()

            edge_types = metadata[1]

            self.convs = torch.nn.ModuleList()
            for _ in range(num_layers):
                conv = HeteroConv({edge_type: SAGEConv((-1, -1), hidden_channels) for edge_type in edge_types})
                self.convs.append(conv)

            self.lin = Linear(hidden_channels, out_channels)

            self._targets = targets

        def forward(self, x_dict, edge_index_dict):
            for conv in self.convs:
                x_dict = {key: x.float() for key, x in x_dict.items()}
                x_dict = conv(x_dict, edge_index_dict)
                x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}

            target_layers = x_dict.keys() if self._targets is None else self._targets
            hs = torch.cat([x_dict[l] for l in target_layers], dim=0)

            return self.lin(hs)

    return (HeteroGNN,)


@app.cell
def _(HeteroGNN, dataset):
    from torch_geometric.loader import DataLoader

    device = torch.device('cpu')

    loader = DataLoader(dataset, batch_size=1)
    data = next(iter(loader)).to(device)
    model = HeteroGNN(data.metadata(), hidden_channels=64, out_channels=1, num_layers=2).to(device)

    with torch.no_grad():  # Initialize lazy modules.
        out = model(data.x_dict, data.edge_index_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)
    return data, model, optimizer


@app.cell
def _(data, model, optimizer):
    def train():
        model.train()
        optimizer.zero_grad()

        out = model(data.x_dict, data.edge_index_dict)

    return


if __name__ == "__main__":
    app.run()
