import marimo

__generated_with = "0.14.16"
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
    return np, nx, pl, plt


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
def _(np, nx):
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

    G = nx.Graph()
    G.add_weighted_edges_from(((u, v, w) for ((u, v), w) in edges.items()), weight="distance")

    nodes = np.array(list(G.nodes()))
    home_nodes = np.array(nodes)
    workplace_nodes = np.array(["A", "B", "C"])
    shopping_nodes = np.array(["B", "C", "D", "E"])


    def set_inclusion_attribute(G, included_nodes, attr_name: str):
        nx.set_node_attributes(G, {node: node in included_nodes for node in G.nodes}, attr_name)


    set_inclusion_attribute(G, home_nodes, "is_home")
    set_inclusion_attribute(G, workplace_nodes, "is_workplace")
    set_inclusion_attribute(G, shopping_nodes, "is_shopping")
    return G, home_nodes, nodes, shopping_nodes, workplace_nodes


@app.cell
def _(G, nx, plt):
    def _node_colour(node_attrs: dict) -> str:
        if "is_shopping" not in node_attrs or "is_workplace" not in node_attrs:
            return "tab:gray"

        if node_attrs["is_shopping"] and node_attrs["is_workplace"]:
            return "orangered"
        if node_attrs["is_shopping"]:
            return "orange"
        if node_attrs["is_workplace"]:
            return "tomato"

        return "tab:blue"


    def draw_network(G: nx.Graph):
        fig, ax = plt.subplots()

        pos = nx.spring_layout(G, seed=42, weight="distance")
        edge_labels = nx.get_edge_attributes(G, "distance")

        colors = [_node_colour(attrs) for _, attrs in G.nodes(data=True)]

        nx.draw_networkx(G, pos, node_color=colors, ax=ax)
        nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax)

        return fig, ax


    draw_network(G)
    return (draw_network,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Compute distance matrix and create fully connected version of graph above.""")
    return


@app.cell
def _(G, nodes, np, nx):
    def node_ordering(nodes: list[str]):
        return dict((node, idx) for idx, node in enumerate(nodes))


    def distance_matrix(G: nx.Graph, nodes: list[str]):
        matrix = np.empty((len(nodes), len(nodes)), dtype=np.float32)
        ordering = node_ordering(nodes)
        distances = nx.shortest_path_length(G, weight="distance")

        for node, distance in distances:
            idx = ordering[node]
            row_items = sorted(distance.items(), key=lambda x: ordering[x[0]])
            row = [dist for _, dist in row_items]

            matrix[idx, :] = row

        return matrix


    distances = distance_matrix(G, nodes)
    distances
    return distance_matrix, distances


@app.cell
def _(G, distance_matrix, draw_network, nodes, nx):
    def fully_connected_graph(G: nx.Graph, nodes: list[str]):
        distances = distance_matrix(G, nodes)
        G_full = nx.from_numpy_array(distances, edge_attr="distance", nodelist=nodes)
        nx.set_node_attributes(G_full, dict(G.nodes(data=True)))
        return G_full


    G_full = fully_connected_graph(G, nodes)
    draw_network(G_full)
    return (G_full,)


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
def _(np):
    def select_closest_from_choice(nodes, distances, choices_idx, valid, exclude_chosen=False):
        """Given a list of chosen nodes, selects the closest node from the list of valid nodes"""
        is_valid_mask = np.isin(nodes, valid)
        masked_distances = np.where(is_valid_mask, distances[choices_idx], np.inf)

        if exclude_chosen:
            all_rows = np.arange(masked_distances.shape[0])
            masked_distances[all_rows, choices_idx] = np.inf

        closest_nodes_idx = np.argmin(masked_distances, axis=1)
        return nodes[closest_nodes_idx]
    return (select_closest_from_choice,)


@app.cell
def _(
    distances,
    home_nodes,
    mo,
    nodes,
    np,
    select_closest_from_choice,
    shopping_nodes,
    workplace_nodes,
):
    rng = np.random.default_rng(42)

    n_samples = 1000
    exclude_chosen_from_shopping = True

    home_choice_idx = rng.integers(0, len(home_nodes), size=n_samples)
    home_choice = np.array(home_nodes)[home_choice_idx]

    work_choice_idx = rng.integers(0, len(workplace_nodes), size=n_samples)
    work_choice = rng.choice(workplace_nodes, size=n_samples)

    closest_home_shopping = select_closest_from_choice(
        nodes, distances, choices_idx=home_choice_idx, valid=shopping_nodes, exclude_chosen=exclude_chosen_from_shopping
    )
    closest_work_shopping = select_closest_from_choice(
        nodes, distances, choices_idx=work_choice_idx, valid=shopping_nodes, exclude_chosen=exclude_chosen_from_shopping
    )
    with mo.redirect_stdout():
        print(f"Home:  {home_choice}")
        print(f"Work:  {work_choice}")
        print(f"Shop1: {closest_home_shopping}")
        print(f"Shop2: {closest_work_shopping}")
    return (
        closest_home_shopping,
        closest_work_shopping,
        home_choice,
        n_samples,
        rng,
        work_choice,
    )


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
def _(n_samples, np, rng):
    # import itertools
    # all_permuations = [x for r in range(1, 4) for x in itertools.permutations(available, r=r)]

    available_activities = ["W", "S1", "S2"]
    available_schedules = np.array(
        [
            ["W", "-", "-"],
            ["S1", "-", "-"],
            ["S2", "-", "-"],
            ["W", "S1", "-"],
            ["W", "S2", "-"],
            ["S1", "W", "-"],
            ["S1", "S2", "-"],
            ["S2", "W", "-"],
            ["S2", "S1", "-"],
            # ["W", "S1", "S2"],
            ["W", "S2", "S1"],
            ["S1", "W", "S2"],
            ["S1", "S2", "W"],
            ["S2", "W", "S1"],
            # ["S2", "S1", "W"],
        ]
    )

    chosen_schedules = rng.integers(low=0, high=len(available_schedules), size=n_samples)
    schedules = available_schedules[chosen_schedules]
    home_col = np.repeat("H", n_samples).reshape(n_samples, 1)
    schedules = np.hstack([home_col, schedules, home_col])

    schedules
    return (schedules,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Choices by person:""")
    return


@app.cell
def _(
    closest_home_shopping,
    closest_work_shopping,
    home_choice,
    pl,
    work_choice,
):
    person_choices_df = pl.concat(
        [
            pl.DataFrame(
                {
                    "type": "H",
                    "loc_id": home_choice,
                }
            ).with_row_index("person_id"),
            pl.DataFrame(
                {
                    "type": "W",
                    "loc_id": work_choice,
                }
            ).with_row_index("person_id"),
            pl.DataFrame(
                {
                    "type": "S1",
                    "loc_id": closest_home_shopping,
                }
            ).with_row_index("person_id"),
            pl.DataFrame(
                {
                    "type": "S2",
                    "loc_id": closest_work_shopping,
                }
            ).with_row_index("person_id"),
        ]
    )

    person_choices_df
    return (person_choices_df,)


@app.cell
def _(mo):
    mo.md(r"""Schedules:""")
    return


@app.cell
def _(person_choices_df, pl, schedules):
    schedule_df = (
        pl.DataFrame(schedules, schema=["1", "2", "3", "4", "5"])
        .with_row_index("person_id")
        .unpivot(index="person_id", variable_name="numpy_seq", value_name="type")
        .sort(by=["person_id", "numpy_seq"])
        .filter(pl.col("type") != "-")
        .with_columns(pl.int_range(pl.len()).over("person_id", order_by="numpy_seq").alias("sequence_num"))
        .drop("numpy_seq")
        .join(person_choices_df, on=["person_id", "type"])
        .select("person_id", "sequence_num", "type", "loc_id")
        .sort(by=["person_id", "sequence_num"])
    )

    schedule_df
    return (schedule_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Trip dataframe corresponding to activity schedules with distance measures.""")
    return


@app.cell
def _(distances, nodes, pl, schedule_df):
    def _shifted(name, prefix="to_"):
        return pl.col(name).shift(-1).alias(prefix + name)


    distances_df = (
        pl.DataFrame(distances, schema=list(nodes))
        .with_columns(from_loc_id=nodes)
        .unpivot(index="from_loc_id", variable_name="to_loc_id", value_name="distance")
    )

    trip_df = (
        schedule_df.sort(by=["person_id", "sequence_num"])
        .with_columns(_shifted("loc_id"), _shifted("type"), _shifted("person_id"))
        .filter(pl.col("person_id") == pl.col("to_person_id"))
        .drop("to_person_id")
        .rename({"type": "from_type", "loc_id": "from_loc_id"})
    )

    trip_df = trip_df.join(distances_df, on=["from_loc_id", "to_loc_id"]).sort(by=["person_id", "sequence_num"])
    trip_df
    return (trip_df,)


@app.cell
def _(mo, n_samples):
    selected_person = mo.ui.number(start=0, stop=n_samples - 1, label="Person ID: ")
    return (selected_person,)


@app.cell(hide_code=True)
def _(G, draw_trip, mo, selected_person, trip_df):
    mo.vstack(
        [
            mo.md("Generated schedules: "),
            mo.hstack([draw_trip(G, trip_df, selected_person.value), selected_person], align="start", justify="start"),
        ]
    )
    return


@app.cell
def _(nx, pl, plt):
    def _activities_to_colors(types: list[str]):
        if "H" in types:
            return "tab:blue"
        if "W" in types and ("S1" in types or "S2" in types):
            return "orangered"
        if "S1" in types or "S2" in types:
            return "orange"
        if "W" in types:
            return "tomato"

        raise NotImplementedError("Impossible")


    def draw_trip(G: nx.Graph, trip_df: pl.DataFrame, person_id: int):
        fig, ax = plt.subplots()

        pos = nx.spring_layout(G, seed=42, weight="distance")
        edge_labels = nx.get_edge_attributes(G, "distance")

        trips = trip_df.filter(pl.col("person_id") == person_id)
        edgelist = trips.select("from_loc_id", "to_loc_id").rows()
        node_colours = (
            pl.concat(
                [
                    trips.select("from_loc_id", "from_type").rename({"from_loc_id": "loc_id", "from_type": "type"}),
                    trips.select("to_loc_id", "to_type").rename({"to_loc_id": "loc_id", "to_type": "type"}),
                ]
            )
            .group_by("loc_id")
            .agg(pl.col("type").map_batches(_activities_to_colors, return_dtype=pl.String).first())
            .join(pl.DataFrame({"loc_id": list(G.nodes())}), on="loc_id", how="right")
            .with_columns(pl.col("type").fill_null("tab:gray"))
        )["type"].to_list()

        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colours)
        nx.draw_networkx_labels(G, pos, ax=ax)
        nx.draw_networkx_edges(G, pos, ax=ax)
        nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax)
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edgelist,
            arrows=True,
            arrowstyle="-|>",
            style="--",
            connectionstyle="arc3,rad=0.2",
            edge_color="red",
            ax=ax,
        )

        return fig, ax
    return (draw_trip,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## PyG conversion""")
    return


@app.cell
def _():
    from torch_geometric.data import Dataset, InMemoryDataset, Data
    from torch_geometric.utils import from_networkx, to_networkx
    return Data, InMemoryDataset, from_networkx, to_networkx


@app.cell
def _(G_full, draw_network, from_networkx, to_networkx):
    data = from_networkx(G_full, group_edge_attrs="distance")
    draw_network(to_networkx(data))
    return (data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Create a dataset of node features and targets based on if the nodes appear in the group""")
    return


@app.cell
def _(pl, trip_df):
    import polars.selectors as cs

    _features = (
        trip_df.group_by("person_id")
        .agg(pl.col("from_loc_id").unique(maintain_order=True))
        .explode("from_loc_id")
        .with_columns(pl.int_range(pl.len()).over("person_id").alias("sequence_num"))
        .to_dummies("from_loc_id")
        .with_columns(cs.starts_with("from_loc_id").cum_sum().over("person_id", order_by="sequence_num"))
    )

    _targets = _features.with_columns(pl.col("sequence_num") - 1)
    dataset_df = _features.join(_targets, on=["person_id", "sequence_num"], suffix="_target").sort(
        by=["person_id", "sequence_num"]
    )

    X = dataset_df.select(pl.col("sequence_num"), cs.starts_with("from_loc_id") & ~cs.ends_with("_target")).to_torch()
    y = dataset_df.select(cs.ends_with("_target")).to_torch()
    return X, dataset_df, y


@app.cell
def _(Data, InMemoryDataset, X, data, y):
    class BasicLocationsDataset(InMemoryDataset):
        def __init__(self, data, X, y):
            super().__init__()
            self.edge_index = data.edge_index
            self.edge_attr = data.edge_attr
            self.data = data
            self.graph_x = X[:, 0]
            self.X = X[:, 1:].unsqueeze(2).float()
            self.y = y.unsqueeze(2).float()

        def len(self):
            return len(self.X)

        def get(self, idx):
            return Data(
                x=self.X[idx],
                edge_index=self.edge_index,
                edge_attr=self.edge_attr,
                y=self.y[idx],
                graph_x=self.graph_x[idx],
            )


    dataset = BasicLocationsDataset(data, X, y)
    dataset
    return (dataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Split into a training and test set""")
    return


@app.cell
def _(dataset, dataset_df):
    from sklearn.model_selection import GroupShuffleSplit

    groups = dataset_df["person_id"]
    train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42).split(dataset, groups=groups))

    train_set = dataset[train_idx]
    test_set = dataset[test_idx]

    train_groups = groups[train_idx]
    test_groups = groups[test_idx]

    print(f"Training samples: {len(train_set)}")
    print(f"Test samples: {len(test_set)}")
    return test_set, train_groups, train_set


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
def _(DataLoader, batch_size, gkf, train_groups, train_set):
    from torch.utils.data import SubsetRandomSampler

    _train_idx, _val_idx = next(gkf.split(train_set, groups=train_groups))

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


    n_epochs = 100
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


if __name__ == "__main__":
    app.run()
