import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import plotting
    return mo, plotting


@app.cell
def _():
    import math
    import torch
    import torch.nn.functional as F
    return F, math, torch


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
    return np, plt


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


@app.cell(hide_code=True)
def _(mo):
    mo.md("""Define a simple two layer GCN model to get started""")
    return


@app.cell
def _(dataset):
    from models import SimpleGCN

    gcn = SimpleGCN(in_channels=dataset.num_features, hidden_channels=32, out_channels=dataset.num_features)

    gcn
    return (gcn,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We also define an experiment object that holds data on our train and test sets.""")
    return


@app.cell
def _(test_set, train_set):
    from experiment import Experiment, run_experiment

    experiment = Experiment(
        train_set=train_set,
        test_set=test_set,
        n_epochs=50,
        val_size=0.15,
        batch_size=32,
        random_state=42,
    )

    experiment
    return experiment, run_experiment


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Define two benchmark models to compare against. 

    `EqualProbability` assigns equal probability to all nodes that have not yet been selected.

    `BestGuess` knows the data generation process, and thus computes the conditional probability for each node given the sequence number and the pre-selected nodes.
    """
    )
    return


@app.cell
def _(synth_graph):
    from models import EqualProbablity, BestGuess
    from experiment import compute_benchmark

    equal_model = EqualProbablity()
    best_model = BestGuess.from_graph(synth_graph)
    return best_model, compute_benchmark, equal_model


@app.cell
def _(best_model, compute_benchmark, equal_model, experiment, mo):
    equal_results = compute_benchmark(experiment, equal_model)
    best_results = compute_benchmark(experiment, best_model)

    with mo.redirect_stdout():
        print(equal_results)
        print(best_results)
    return best_results, equal_results


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We then train our trainable models (in this case only `SimpleGCN`)""")
    return


@app.cell
def _(experiment, gcn, mo, plotting, run_experiment):
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


@app.cell(hide_code=True)
def _(
    best_model,
    equal_model,
    experiment,
    gcn,
    get_prediction_idx,
    mo,
    n_test_samples,
    next_btn,
    plot_models,
    prev_btn,
):
    sample = experiment.test_set[get_prediction_idx()]

    mo.vstack(
        [
            mo.md("Comparison of predictions between models and benchmarks: "),
            mo.hstack([prev_btn, mo.md(f"Sample #{get_prediction_idx()}/{n_test_samples - 1}"), next_btn], align="center"),
            plot_models(sample, [gcn], [best_model, equal_model]),
        ]
    )
    return


@app.cell(hide_code=True)
def _(experiment, mo):
    def create_prev_next_buttons(n_samples: int):
        get_prediction, set_prediction = mo.state(1)

        def _decrease(_):
            if get_prediction() > 0:
                set_prediction(lambda x: x - 1)

        def _increase(_):
            if get_prediction() < n_samples - 1:
                set_prediction(lambda x: x + 1)

        prev_button = mo.ui.button(on_click=_decrease, label="Prev")
        next_button = mo.ui.button(on_click=_increase, label="Next")

        return prev_button, next_button, get_prediction


    n_test_samples = len(experiment.test_set)
    prev_btn, next_btn, get_prediction_idx = create_prev_next_buttons(n_test_samples)
    return get_prediction_idx, n_test_samples, next_btn, prev_btn


@app.cell(hide_code=True)
def _(F, math, plotting, plt, synth_graph, torch):
    def plot_models(sample, models, benchmarks):
        def _plot(sample, model, ax, is_benchmark):
            y_prob = model(sample).squeeze()

            if not is_benchmark:
                loss = F.nll_loss(F.log_softmax(y_prob, dim=0), sample.y)
                y_prob = F.softmax(y_prob, dim=0)
            else:
                loss = F.nll_loss(torch.log(y_prob), sample.y)

            ax.set_title(f"{type(model).__name__} (loss={loss:.2f})")
            y_prob = y_prob.squeeze().tolist()
            return plotting.draw_prediction(synth_graph, sample.x, y_prob, ax=ax)

        n_boxes = len(models) + len(benchmarks) + 1
        n_rows = math.ceil(n_boxes / 3)
        fig, axs = plt.subplots(figsize=(12, 4 * n_rows), ncols=3, nrows=n_rows, squeeze=False)

        flat_axs = [ax for a in axs for ax in a]

        for ax in flat_axs[n_boxes:]:
            ax.set_axis_off()

        actual_y_prob = F.one_hot(sample.y, num_classes=sample.num_nodes).float()
        plotting.draw_prediction(synth_graph, sample.x, actual_y_prob.tolist(), ax=flat_axs[0], labels=False)
        flat_axs[0].set_title("Actual")

        for ax, model in zip(flat_axs[1 : len(models) + 1], models):
            _ = _plot(sample, model, ax, is_benchmark=False)

        for ax, model in zip(flat_axs[len(models) + 1 :], benchmarks):
            _ = _plot(sample, model, ax, is_benchmark=True)

        fig.suptitle("Comparison of predictions")

        return fig
    return (plot_models,)


if __name__ == "__main__":
    app.run()
