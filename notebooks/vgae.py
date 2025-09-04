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

    n_samples = 1000

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
    return (train_set,)


@app.cell
def _(train_set):
    from torch_geometric.loader import DataLoader

    train_load = DataLoader(train_set, batch_size=32)

    batch = next(iter(train_load))
    data = train_set[0]

    batch, data
    return (batch,)


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
        out_num_nodes=train_set[0].num_nodes,
        out_num_classes=train_set.num_classes
    )

    label_decoder
    return (label_decoder,)


@app.cell
def _(gcn_encoder, label_decoder):
    from torch_geometric.nn import VGAE


    vgae = VGAE(gcn_encoder, label_decoder)
    vgae
    return (vgae,)


@app.cell
def _(batch, vgae):
    z = vgae.encode(batch)
    y = vgae.decode(z)

    y.shape
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
