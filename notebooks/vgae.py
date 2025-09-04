import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium", app_title="VGAE")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# VGAE for subgraph query completion""")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import plotting
    return (plotting,)


@app.cell
def _():
    import numpy as np
    import polars as pl
    return (np,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Dataset generation

    Generate a synthetic graph, population and schedules.
    """
    )
    return


@app.cell
def _():
    from synthetic import SyntheticGraph

    synth = SyntheticGraph.example()
    return (synth,)


@app.cell
def _(plotting, synth):
    plotting.draw_synthetic_network(synth)
    return


@app.cell
def _(np, synth):
    from synthetic import SyntheticGenerator

    n_samples = 10000

    _rng = np.random.default_rng(seed=42)
    _generator = SyntheticGenerator(synth, _rng)
    _generator.generate_population(n_samples)
    _generator.generate_schedules()

    schedules = _generator.build()
    schedules
    return (schedules,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Conversion to PyG dataset""")
    return


@app.cell
def _(schedules):
    from datasets import convert_to_pyg_dataset

    dataset = convert_to_pyg_dataset(schedules)
    dataset
    return (dataset,)


@app.cell
def _(dataset):
    from datasets import train_test_split

    train_set, test_set = train_test_split(dataset, random_state=42)
    train_set, test_set
    return test_set, train_set


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Defining the models""")
    return


@app.cell
def _(train_set):
    from models import GCNEncoder

    hidden_channels = 32
    latent_channels = 16

    gcn_encoder = GCNEncoder(
        in_node_channels=train_set.num_features, 
        in_graph_channels=train_set.num_graph_features, 
        hidden_channels=hidden_channels,
        num_layers=2,
        latent_channels=latent_channels
    )

    gcn_encoder
    return gcn_encoder, hidden_channels, latent_channels


@app.cell
def _(hidden_channels, latent_channels, train_set):
    from models import MLPDecoder

    label_decoder = MLPDecoder(
        latent_channels=latent_channels,
        hidden_channels=hidden_channels,
        num_layers=3,
        out_channels=train_set.num_classes,
    )

    label_decoder
    return (label_decoder,)


@app.cell
def _(gcn_encoder, label_decoder):
    from models import NodeLabelVGAE

    vgae = NodeLabelVGAE(gcn_encoder, label_decoder)
    vgae
    return (vgae,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Test that the models output something""")
    return


@app.cell
def _(train_set):
    from torch_geometric.loader import DataLoader

    _train_load = DataLoader(train_set, batch_size=32)

    batch = next(iter(_train_load))
    data = train_set[0]

    batch, data
    return DataLoader, batch


@app.cell
def _(batch, vgae):
    mu, log_std = vgae.forward(batch)
    z = vgae.reparametrize(mu, log_std)
    y = vgae.decode(z)

    y.shape
    return log_std, mu, y


@app.cell
def _(batch, y):
    from losses import recon_loss

    recon = recon_loss(y, batch.y)
    recon
    return (recon_loss,)


@app.cell
def _(log_std, mu):
    from losses import kl_loss

    kl = kl_loss(mu, log_std)
    kl
    return


@app.cell
def _(batch, log_std, mu, y):
    from losses import elbo_loss

    elbo = elbo_loss(mu, log_std, y, batch.y)
    elbo
    return (elbo_loss,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Train the model""")
    return


@app.cell
def _():
    import torch
    return (torch,)


@app.cell
def _(DataLoader, test_set, torch, train_set, vgae):
    lr = 0.01
    n_epochs = 10
    batch_size = 32

    model = vgae
    device = torch.device("cpu")
    train_loader = DataLoader(train_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return device, model, n_epochs, optimizer, test_loader, train_loader


@app.cell
def _(device, elbo_loss, model, n_epochs, optimizer, train_loader):
    def train_epoch(model, device, train_loader, optimizer):
        epoch_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            mu, log_std = model.forward(batch)
            z = model.reparametrize(mu, log_std)
            y = model.decode(z)

            loss = elbo_loss(mu, log_std, y, batch.y)
            loss.backward()

            epoch_loss += loss.item()
            optimizer.step()

        return epoch_loss / len(train_loader)

    train_losses = []

    for epoch in range(n_epochs):
        model.train()

        total_loss = train_epoch(model, device, train_loader, optimizer)
        train_losses.append(total_loss)

        print(f"Epoch {epoch}: elbo={total_loss}")
    return


@app.cell
def _(device, model, recon_loss, test_loader, torch):
    def evaluate_model(model, device, loader, criterion) -> float:
        model.eval()
        total_loss = 0

        for batch in loader:
            batch = batch.to(device)

            with torch.no_grad():
                y = model.infer(batch)
                loss = criterion(y, batch.y)

            total_loss += loss

        return total_loss / len(loader)

    test_loss = evaluate_model(model, device, test_loader, recon_loss)
    test_loss
    return (evaluate_model,)


@app.cell
def _(device, evaluate_model, model, recon_loss, train_loader):
    evaluate_model(model, device, train_loader, recon_loss)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
