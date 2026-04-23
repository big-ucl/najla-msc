import marimo

__generated_with = "0.19.9"
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
    from archive.locations import ActivityDataset

    network_name = "stops"

    dataset = ActivityDataset.from_files(cfg.data, project_root, name=network_name)
    dataset
    return (dataset,)


@app.cell
def _():
    import torch.nn.functional as F
    from torch_geometric.nn import HeteroConv, Linear, GATConv

    class HeteroGNN(torch.nn.Module):
        def __init__(self, metadata, hidden_channels, out_channels, num_layers, output_types=None):
            super().__init__()

            node_types, edge_types = metadata

            self.convs = torch.nn.ModuleList()
            for _ in range(num_layers):
                conv = HeteroConv({
                    edge_type: GATConv((-1, -1), hidden_channels, add_self_loops=False) for edge_type in edge_types
                })
                self.convs.append(conv)

            self.lin = Linear(hidden_channels, out_channels)
            self.output_types = output_types if output_types is not None else node_types

        def forward(self, x_dict, edge_index_dict, edge_attr_dict):
            for conv in self.convs:
                x_dict = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
                x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}

            splits = [x_dict[t].shape[0] for t in self.output_types]
            hs = torch.cat([x_dict[t] for t in self.output_types], dim=0)

            out = self.lin(hs)
            out = torch.split(out, splits)
            out = {t: o for t, o in zip(self.output_types, out)}

            return out

    return F, HeteroGNN


@app.cell
def _(F, HeteroGNN, dataset):
    from tqdm import tqdm
    from torch_geometric.loader import DataLoader

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    loader = DataLoader(dataset, batch_size=32)
    data = next(iter(loader)).to(device)
    model = HeteroGNN(data.metadata(), hidden_channels=64, out_channels=1, num_layers=2).to(device)

    with torch.no_grad():  # Initialize lazy modules.
        x_dict = {k: x.float() for k, x in data.x_dict.items()}
        _ = model(x_dict, data.edge_index_dict, data.edge_attr_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)

    def train(model, optimizer, loader):
        model.train()
        optimizer.zero_grad()

        losses = []

        for batch in tqdm(loader):
            batch = batch.to(device)

            x_dict = {k: x.float() for k, x in batch.x_dict.items()}
            y_dict = {k: y for k, y in batch.y_dict.items()}
            out = model(x_dict, batch.edge_index_dict, batch.edge_attr_dict)

            out_cat = torch.cat([out[t] for t in model.output_types])
            y_cat = torch.cat([y_dict[t] for t in model.output_types])

            loss = F.binary_cross_entropy_with_logits(out_cat.squeeze(), y_cat)
            loss.backward()
            optimizer.step()

            losses.append(float(loss.detach()))

        return losses

    losses = train(model, optimizer, loader)
    return data, losses, model


@app.cell
def _(losses):
    sum(losses)
    return


@app.cell
def _():
    return


@app.cell
def _():
    import torch_geometric.transforms as T

    data = T.AddSelfLoops()(data)
    return (data,)


@app.cell
def _(data, model):
    _x_dict = {k: x.float() for k, x in data.x_dict.items()}

    model(_x_dict, data.edge_index_dict, data.edge_attr_dict)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
