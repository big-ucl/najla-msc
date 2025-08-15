import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import plotting
    return mo, plotting


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
    return (np,)


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
def _(plotting, synth_graph):
    plotting.draw_synthetic_network(synth_graph)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Compute distance matrix and create fully connected version of graph above.""")
    return


@app.cell
def _(synth_graph):
    synth_graph.distance_matrix
    return


@app.cell
def _(plotting, synth_graph):
    plotting.draw_synthetic_network(synth_graph, full=True)
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
def _(mo, plotting, schedules, selected_person):
    mo.vstack(
        [
            mo.md("Generated schedules: "),
            mo.hstack(
                [plotting.draw_synthetic_trip(schedules, selected_person.value), selected_person],
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


@app.cell
def _(dataset):
    dataset[0]
    return


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
    mo.md(r"""## Model definition and training""")
    return


@app.cell
def _(dataset, test_set, train_set):
    from models import SimpleGCN
    from experiment import Experiment, run_experiment

    gcn = SimpleGCN(in_channels=dataset.num_features, hidden_channels=32, out_channels=dataset.num_features)

    experiment = Experiment(
        train_set=train_set,
        test_set=test_set,
        n_epochs=50,
        val_size=0.15,
        batch_size=32,
        random_state=42,
    )

    gcn
    return experiment, gcn, run_experiment


@app.cell
def _(experiment, synth_graph):
    from models import EqualProbablity, BestGuess
    from experiment import compute_benchmark

    best_results = compute_benchmark(experiment, BestGuess.from_graph(synth_graph))
    equal_results = compute_benchmark(experiment, EqualProbablity())
    return best_results, equal_results


@app.cell
def _(experiment, gcn, mo, plotting, run_experiment):
    # mo.stop(True)

    results = run_experiment(experiment, gcn)

    with mo.redirect_stdout():
        _train, _val, _test = results.final_losses()

        print(
            f"Final losses after {results.n_epochs} epochs: "
            f"Training={_train:.4f} | Validation={_val:.4f} | Test={_test:.4f}"
        )

    plotting.plot_training_progress(results)
    return (results,)


@app.cell
def _(best_results, equal_results, plotting, results):
    plotting.plot_model_comparisons(results, best_results, equal_results, how="bar")
    return


@app.cell
def _(best_results, equal_results, plotting, results):
    plotting.plot_model_comparisons(results, best_results, equal_results, how="line")
    return


if __name__ == "__main__":
    app.run()
