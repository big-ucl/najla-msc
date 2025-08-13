import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    from config import load_config

    cfg = load_config(mo.notebook_dir().parent)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    ## Synthetic data generation

    Construct an example graph and convert it to a fully connected graph with edge weights as shortest path distances.
    Then create activity schedules over this graph and create a corresponding `ActivityDataset`.
    """
    )
    return


@app.cell
def _():
    import networkx as nx
    import numpy as np
    import polars as pl
    import matplotlib.pyplot as plt
    return np, pl, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Locations and connected graph

    Create example NetworkX graph and draw it
    """
    )
    return


@app.cell
def _():
    from synthetic import SyntheticGraph

    edges = {
        ("A", "B"): 5,
        ("B", "C"): 5,
        ("C", "A"): 5,
        ("C", "D"): 15,
        ("D", "E"): 3,
        ("E", "F"): 7,
        ("E", "G"): 2,
        ("F", "G"): 2,
        ("F", "D"): 9,
    }

    workplace_nodes = ["A", "B", "C"]
    shopping_nodes = ["B", "C", "D", "E"]

    synth_graph = SyntheticGraph(edges, workplace_nodes, shopping_nodes)
    synth_graph
    return (synth_graph,)


@app.cell
def _(synth_graph):
    from plotting import draw_synthetic_network

    draw_synthetic_network(synth_graph)
    return (draw_synthetic_network,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Compute distance matrix and create fully connected version of graph above.""")
    return


@app.cell
def _(synth_graph):
    synth_graph.distance_matrix
    return


@app.cell
def _(draw_synthetic_network, synth_graph):
    draw_synthetic_network(synth_graph, full=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Activity sequence generation

    For each individual, generate activity sequence as follows :

    1. Location choices:
        - Pick a home node at random
        - Pick a work node at random in the list of valid work nodes
        - Pick closest shopping nodes to home, and to work
    """
    )
    return


@app.cell
def _(np, synth_graph):
    from synthetic import SyntheticGenerator

    rng = np.random.default_rng(42)
    generator = SyntheticGenerator(synth_graph, rng)

    n_samples = 1000
    exclude_chosen_from_shopping = True


    generator.generate_population(n_samples, exclude_chosen_from_shopping)
    generator.person_choices_df
    return generator, n_samples


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    2. Sequence generation
        - Start at home
        - Choose from available schedules with equal prob. (between 1 and 3 activities, S2 has to be alone or directly beside work)
        - Finish day at home
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Generated schedules:""")
    return


@app.cell
def _(generator):
    generator.generate_schedules()
    generator.schedule_df
    return


@app.cell
def _(generator):
    schedules = generator.build()
    return (schedules,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Trip dataframe corresponding to activity schedules with distance measures.""")
    return


@app.cell
def _(schedules):
    schedules.trip_df
    return


@app.cell
def _(mo, n_samples):
    selected_person = mo.ui.number(start=0, stop=n_samples - 1, label="Person ID: ")
    return (selected_person,)


@app.cell(hide_code=True)
def _(mo, schedules, selected_person):
    from plotting import draw_synthetic_trip

    mo.vstack(
        [
            mo.md("Generated schedules: "),
            mo.hstack(
                [draw_synthetic_trip(schedules, selected_person.value), selected_person],
                align="start",
                justify="start",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## PyG conversion""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Create a dataset of node features and targets based on if the nodes appear in the group""")
    return


@app.cell
def _(schedules):
    from datasets import convert_to_pyg_dataset

    dataset = convert_to_pyg_dataset(schedules)
    dataset
    return (dataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Split into a training and test set""")
    return


@app.cell
def _(dataset, mo):
    from datasets import train_test_split

    train_set, test_set = train_test_split(dataset, test_size=0.15, random_state=42)

    with mo.redirect_stdout():
        print(f"Training samples: {len(train_set)}")
        print(f"Test samples: {len(test_set)}")
    return test_set, train_set


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Model definition""")
    return


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    import torch_geometric.nn as gnn


    # Define the model architecture
    class SimpleGCN(nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels):
            super().__init__()
            self.conv1 = gnn.GCNConv(in_channels, hidden_channels)
            self.conv2 = gnn.GCNConv(hidden_channels, out_channels)

        def forward(self, batch):
            x = batch.x
            edge_index = batch.edge_index
            edge_attr = batch.edge_attr.squeeze()

            x = self.conv1(x, edge_index, edge_attr)
            x = F.relu(x)
            x = self.conv2(x, edge_index, edge_attr)

            return x
    return F, SimpleGCN, nn, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Model training""")
    return


@app.cell
def _(dataset):
    print(f"Number of graphs: {len(dataset)}")
    print(f"Number of features: {dataset.num_features}")
    return


@app.cell
def _(F, SimpleGCN, dataset, torch):
    from sklearn.model_selection import GroupKFold
    from torch_geometric.loader import DataLoader

    k_folds = 2
    batch_size = 32

    gkf = GroupKFold(n_splits=k_folds, shuffle=True, random_state=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleGCN(in_channels=dataset.num_features, hidden_channels=32, out_channels=dataset.num_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = F.binary_cross_entropy_with_logits
    return DataLoader, batch_size, criterion, device, gkf, model, optimizer


@app.cell
def _(DataLoader, batch_size, gkf, train_set):
    from torch.utils.data import SubsetRandomSampler

    _train_idx, _val_idx = next(gkf.split(train_set, groups=train_set.indices()))

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(_train_idx),
    )

    val_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(_val_idx),
    )
    return train_loader, val_loader


@app.cell
def _(
    DataLoader,
    criterion,
    device,
    mo,
    model,
    nn,
    optimizer,
    pl,
    torch,
    train_loader,
    val_loader,
):
    def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module):
        model.train()
        total_loss = 0

        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()

            total_loss += loss.item()
            optimizer.step()

        return total_loss / len(loader)


    def evaluate_model(model: nn.Module, loader: DataLoader, criterion: torch.nn.Module):
        model.eval()
        total_loss = 0

        for batch in loader:
            batch = batch.to(device)

            with torch.no_grad():
                out = model(batch)
                loss = criterion(out, batch.y)

            total_loss += loss

        return total_loss / len(loader)


    n_epochs = 2
    _train_losses = []
    _val_losses = []

    for _epoch in range(n_epochs):
        _train_loss = train_epoch(model, train_loader, optimizer, criterion)
        _val_loss = evaluate_model(model, val_loader, criterion)

        _train_losses.append(_train_loss)
        _val_losses.append(_val_loss)

        if _epoch % 10 == 0:
            print(f"Epoch {_epoch + 1:3}, Training loss: {_train_loss:.4f} | Validation loss: {_val_loss:.4f}")

    losses = pl.DataFrame({"epoch": range(1, n_epochs + 1), "train": _train_losses, "val": _val_losses})

    with mo.redirect_stdout():
        print(
            f"Final losses after {n_epochs} epochs: Training={losses['train'][-1]:.4f} | Validation={losses['val'][-1]:.4f}"
        )
    return evaluate_model, losses, n_epochs


@app.cell
def _(DataLoader, batch_size, criterion, evaluate_model, mo, model, test_set):
    _test_loader = DataLoader(dataset=test_set, batch_size=batch_size)
    test_loss = evaluate_model(model, _test_loader, criterion)

    with mo.redirect_stdout():
        print(f"Test loss: {test_loss:.4f}")
    return (test_loss,)


@app.cell
def _(losses, n_epochs, plt, test_loss):
    plt.figure(figsize=(10, 5))
    plt.grid()
    plt.plot(range(1, n_epochs + 1), losses["train"], label="Train")
    plt.plot(range(1, n_epochs + 1), losses["val"], label="Validation")
    plt.plot([1, n_epochs], [test_loss, test_loss], label="Test", linestyle="dashed", linewidth=1)
    plt.title("Training, validation, and test Losses")
    plt.xlabel("Epoch")
    plt.xlim([1, n_epochs])
    plt.ylabel("BCE Loss")
    plt.legend()
    plt.show()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
