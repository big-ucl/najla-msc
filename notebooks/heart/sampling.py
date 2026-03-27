import marimo

__generated_with = "0.21.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import pickle

    import torch
    import torch_geometric as pyg

    import pandas as pd
    import geopandas as gpd
    import polars as pl

    import city2graph as c2g
    import pyproj

    from activitygraphs.config import load_config
    from pathlib import Path

    project_root = Path(mo.notebook_dir().parent.parent)
    cfg = load_config(project_root)


@app.cell
def _():
    models_path = project_root / "models"

    batch_size = 64
    test_size = 0.2
    seed = 42
    return batch_size, models_path, seed, test_size


@app.cell
def _():
    from activitygraphs.run import load_dataset

    return (load_dataset,)


@app.cell
def _():
    from activitygraphs.run import build_gat, build_gps, build_mlp

    return (build_gat,)


@app.cell
def _(batch_size, load_dataset, seed, test_size):
    train_set, test_set = load_dataset(cfg, test_size, seed)
    train_loader = pyg.loader.DataLoader(train_set, batch_size=batch_size)
    test_loader = pyg.loader.DataLoader(test_set, batch_size=batch_size)
    return train_loader, train_set


@app.cell
def _(build_gat, models_path, train_set):
    hidden_channels = 128
    dropout = 0.2
    gat_layers = 8
    gps_layers = 4
    mlp_layers = 3

    def load_model(file, builder, layers):
        path = models_path / file

        model = builder(train_set, layers, hidden_channels, dropout)
        model.load_state_dict(torch.load(path, weights_only=True))
        return model.cuda().eval()

    gat = load_model("GATSkip-8-res.pth", build_gat, gat_layers)
    gat
    return (gat,)


@app.cell
def _(gat, train_loader):
    def predict(model, batch):
        return 

    batch = next(iter(train_loader)).cuda()
    logits = gat(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    probs = torch.sigmoid(logits)

    probs
    return batch, logits


@app.cell
def _(batch, logits, seed):
    from activitygraphs.sampling import poisson_sampling, pps_sampling

    generator = torch.Generator(device="cuda").manual_seed(seed)

    poisson_sampling(logits, generator).sum()
    pps_sampling(10, logits, batch.batch, generator).sum()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
