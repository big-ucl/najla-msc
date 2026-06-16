"""
Module: loaders.py

Description:
    A Marimo interactive notebook for loading models and data and making/visualising
    predictions on the Geneva (TPG) dataset.  It primarily loads pretrained GATSkip
    and MLP models from saved .pth files and visualises per-node predicted visit
    probabilities on an interactive map.

    The notebook also demonstrates how to:
      - Initialise the dataset and data loaders.
      - Load and evaluate pre-saved model checkpoints.
      - Add per-user geographic context columns to the network node GeoDataFrame.

Dependencies:
    - activitygraphs library (GenevaData, load_gva_network_graph, build_gat, build_mlp)
    - PyTorch, PyTorch Geometric
    - Marimo, Polars, GeoPandas, Contextily
"""

import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    """
    Description: Import the Marimo notebook framework and the local plotting helpers module.

    Input:
      - (none)

    Output:
      - mo (module): Marimo for reactive UI widgets and Markdown rendering.
      - plotting (module): local module with functions for drawing synthetic graphs,
        model predictions, and training curves.
    """
    import marimo as mo
    import plotting

    return mo, plotting


@app.cell
def _():
    """
    Description: Import PyTorch and related utilities needed for model evaluation and
    loss computation in the prediction visualisation cells.

    Input:
      - (none)

    Output:
      - F (module): ``torch.nn.functional`` for activation functions (softmax, nll_loss).
      - math (module): Python math module for ceiling division when laying out subplots.
      - torch (module): the PyTorch library for tensor operations.
    """
    import math
    import torch
    import torch.nn.functional as F

    return F, math, torch


@app.cell
def _(mo):
    """
    Description: Load the project configuration from the YAML file in the parent directory
    of the notebook. ``cfg`` is kept local (not exported) in this notebook since all
    subsequent cells use it only indirectly through the imported dataset/model objects.

    Input:
      - mo (module): Marimo (provides ``notebook_dir()``).

    Output:
      - (none): ``cfg`` is a local variable (not exported from this cell).
    """
    from config import load_config

    cfg = load_config(mo.notebook_dir().parent)
    return


@app.cell(hide_code=True)
def _(mo):
    """
    Description: Render a section heading for the synthetic data generation part of the notebook.

    Input:
      - mo (module): Marimo.

    Output:
      - (none): renders a Markdown heading in the notebook UI.
    """
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
    """
    Description: Import NumPy and Matplotlib for numerical computation and figure
    rendering in the synthetic-data and prediction-comparison cells.

    Input:
      - (none)

    Output:
      - np (module): NumPy for array operations and random number generation.
      - plt (module): ``matplotlib.pyplot`` for creating and showing figures.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    return np, plt


@app.cell(hide_code=True)
def _(mo):
    """
    Description: Render a sub-section heading for the synthetic network location graph.

    Input:
      - mo (module): Marimo.

    Output:
      - (none): renders a Markdown heading in the notebook UI.
    """
    mo.md(
        r"""
    ### Locations and connected graph

    Create example NetworkX graph and draw it
    """
    )
    return


@app.cell
def _():
    """
    Description: Define a small synthetic transport network and build a ``SyntheticGraph``
    object from it. The graph has 8 nodes (A–H) representing locations and weighted edges
    representing travel times. Two subsets of nodes are designated as workplaces and
    shopping destinations to create realistic activity-scheduling patterns.

    Input:
      - (none)

    Output:
      - synth_graph (SyntheticGraph): the synthetic network object; used by downstream
        cells for data generation, distance-matrix computation, and prediction visualisation.
    """
    from synthetic import SyntheticGraph

    edges = {
        ("A", "B"): 3,
        ("B", "C"): 6,
        ("C", "A"): 5,
        ("C", "D"): 15,
        ("D", "E"): 3,
        ("E", "F"): 7,
        ("E", "G"): 2,
        ("F", "G"): 4,
        ("F", "D"): 9,
        ("F", "H"): 10,
        ("H", "B"): 12,
        ("H", "C"): 16,
    }

    workplace_nodes = ["A", "B", "C"]
    shopping_nodes = ["B", "C", "D", "E"]

    synth_graph = SyntheticGraph(edges, workplace_nodes, shopping_nodes)
    synth_graph
    return (synth_graph,)


@app.cell
def _(plotting, synth_graph):
    """
    Description: Draw the sparse (original) version of the synthetic network showing only
    the explicitly defined edges with their travel-time weights.

    Input:
      - plotting (module): local plotting helpers.
      - synth_graph (SyntheticGraph): the synthetic network.

    Output:
      - (none): renders the network diagram in the notebook UI.
    """
    plotting.draw_synthetic_network(synth_graph)
    return


@app.cell(hide_code=True)
def _(mo):
    """
    Description: Render a note about computing the distance matrix for the fully connected graph.

    Input:
      - mo (module): Marimo.

    Output:
      - (none): renders a Markdown note in the notebook UI.
    """
    mo.md(r"""Compute distance matrix and create fully connected version of graph above.""")
    return


@app.cell
def _(synth_graph):
    """
    Description: Display the all-pairs shortest-path distance matrix for the synthetic
    network. Each cell [i, j] contains the minimum travel time from node i to node j
    using Dijkstra's algorithm. Used to verify connectivity and to build the fully
    connected graph version.

    Input:
      - synth_graph (SyntheticGraph): the synthetic network.

    Output:
      - (none): renders the distance matrix in the notebook UI.
    """
    synth_graph.distance_matrix
    return


@app.cell
def _(plotting, synth_graph):
    """
    Description: Draw the fully connected version of the synthetic network, where edges
    represent the all-pairs shortest-path distances. This is the version used as input
    to the GNN model.

    Input:
      - plotting (module): local plotting helpers.
      - synth_graph (SyntheticGraph): the synthetic network.

    Output:
      - (none): renders the fully-connected network diagram in the notebook UI.
    """
    plotting.draw_synthetic_network(synth_graph, full=True)
    return


@app.cell(hide_code=True)
def _(mo):
    """
    Description: Render a sub-section heading explaining the activity sequence generation rules.

    Input:
      - mo (module): Marimo.

    Output:
      - (none): renders a Markdown heading in the notebook UI.
    """
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
    """
    Description: Generate a synthetic population of 10,000 individuals with randomly
    assigned home and work locations drawn from the synthetic graph. Each person is
    assigned one home node (random) and one work node (random from ``workplace_nodes``).

    Input:
      - np (module): NumPy (for the random number generator).
      - synth_graph (SyntheticGraph): the synthetic network with workplace/shopping node lists.

    Output:
      - generator (SyntheticGenerator): the generator object (used to also generate
        schedules in the next cell and build the final dataset).
      - n_samples (int): the number of individuals generated (10,000); used by the
        sample-selection slider widget.
    """
    from synthetic import SyntheticGenerator

    rng = np.random.default_rng(42)  # deterministic RNG with fixed seed for reproducibility
    generator = SyntheticGenerator(synth_graph, rng)

    n_samples = 10000  # number of synthetic individuals to generate

    generator.generate_population(n_samples)
    generator.person_choices_df
    return generator, n_samples


@app.cell(hide_code=True)
def _(mo):
    """
    Description: Render a step-2 description of the sequence generation rules.

    Input:
      - mo (module): Marimo.

    Output:
      - (none): renders a Markdown list in the notebook UI.
    """
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
    """
    Description: Render a label preceding the generated schedules display cell.

    Input:
      - mo (module): Marimo.

    Output:
      - (none): renders a Markdown label in the notebook UI.
    """
    mo.md(r"""Generated schedules:""")
    return


@app.cell
def _(generator):
    """
    Description: Generate daily activity schedules for each person in the population.
    Each schedule is a sequence of activity locations (home -> activities -> home)
    chosen according to the rules: random schedule type, then location chosen based
    on proximity to home/work. Displays the generated schedule DataFrame.

    Input:
      - generator (SyntheticGenerator): the populated generator from the previous cell.

    Output:
      - (none): renders the schedule DataFrame in the notebook UI. ``generator`` is
        mutated in place.
    """
    generator.generate_schedules()
    generator.schedule_df
    return


@app.cell
def _(generator):
    """
    Description: Finalise the synthetic dataset by calling ``generator.build()``, which
    converts the schedule DataFrame into a ``SyntheticSchedules`` object containing the
    trip DataFrame, distance matrix, and associated metadata needed for PyG conversion.

    Input:
      - generator (SyntheticGenerator): the generator with population and schedules ready.

    Output:
      - schedules (SyntheticSchedules): the finalised synthetic dataset object.
    """
    schedules = generator.build()
    return (schedules,)


@app.cell(hide_code=True)
def _(mo):
    """
    Description: Render a label preceding the trip DataFrame display cell.

    Input:
      - mo (module): Marimo.

    Output:
      - (none): renders a Markdown label in the notebook UI.
    """
    mo.md(r"""Trip dataframe corresponding to activity schedules with distance measures.""")
    return


@app.cell
def _(schedules):
    """
    Description: Display the trip DataFrame from the synthetic schedules. Each row
    represents one trip segment (origin -> destination) for a synthetic individual,
    including the distance computed from the all-pairs shortest-path matrix.

    Input:
      - schedules (SyntheticSchedules): the finalised synthetic dataset.

    Output:
      - (none): renders the trip DataFrame in the notebook UI.
    """
    schedules.trip_df
    return


@app.cell
def _(mo, n_samples):
    """
    Description: Create a numeric input widget that lets the user select a specific person
    ID (0 to n_samples - 1) for viewing their generated activity schedule. The selected
    person ID drives the schedule visualisation in the next cell.

    Input:
      - mo (module): Marimo.
      - n_samples (int): the total number of synthetic individuals (sets the upper bound).

    Output:
      - selected_person (mo.ui.number): the numeric selector widget. Its ``.value``
        attribute (an integer) is used by the schedule-display cell.
    """
    selected_person = mo.ui.number(start=0, stop=n_samples - 1, label="Person ID: ")
    return (selected_person,)


@app.cell(hide_code=True)
def _(mo, plotting, schedules, selected_person):
    """
    Description: Render the interactive schedule view: a label, the person-ID selector
    widget, and the drawn trip diagram for the currently selected individual.

    Input:
      - mo (module): Marimo.
      - plotting (module): local plotting helpers.
      - schedules (SyntheticSchedules): the finalised synthetic dataset.
      - selected_person (mo.ui.number): the person-ID selector widget.

    Output:
      - (none): renders the stacked widget + figure in the notebook UI.
    """
    mo.vstack([
        mo.md("Generated schedules: "),
        selected_person,
        plotting.draw_synthetic_trip(schedules, selected_person.value),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    """
    Description: Render the PyG conversion section heading.

    Input:
      - mo (module): Marimo.

    Output:
      - (none): renders a Markdown heading in the notebook UI.
    """
    mo.md(r"""## PyG conversion""")
    return


@app.cell(hide_code=True)
def _(mo):
    """
    Description: Render a description label for the PyG dataset creation cell below.

    Input:
      - mo (module): Marimo.

    Output:
      - (none): renders a Markdown note in the notebook UI.
    """
    mo.md(r"""Create a dataset of node features and targets based on if the nodes appear in the group""")
    return


@app.cell
def _(schedules):
    """
    Description: Convert the ``SyntheticSchedules`` object into a PyG dataset of
    ``Data`` objects. Each graph in the dataset represents one person's decision problem:
    node features encode location properties, and node labels encode which nodes the
    person actually visited (binary classification target).

    Input:
      - schedules (SyntheticSchedules): the finalised synthetic dataset.

    Output:
      - dataset (BasicLocationsDataset): a PyG dataset with one ``Data`` object per person.
    """
    from datasets import convert_to_pyg_dataset

    dataset = convert_to_pyg_dataset(schedules)
    dataset
    return (dataset,)


@app.cell(hide_code=True)
def _(mo):
    """
    Description: Render a label for the train/test split cell below.

    Input:
      - mo (module): Marimo.

    Output:
      - (none): renders a Markdown note in the notebook UI.
    """
    mo.md(r"""Split into a training and test set""")
    return


@app.cell
def _(dataset, mo):
    """
    Description: Split the PyG dataset into training and test sets using an 85/15 split
    with a fixed random seed for reproducibility. Prints the sizes of both sets.

    Input:
      - dataset (BasicLocationsDataset): the full PyG dataset.
      - mo (module): Marimo (for ``mo.redirect_stdout`` to display print output).

    Output:
      - test_set: the 15% test split of the dataset.
      - train_set: the 85% training split of the dataset.
    """
    from datasets import train_test_split

    train_set, test_set = train_test_split(dataset, test_size=0.15, random_state=42)

    with mo.redirect_stdout():
        print(f"Training samples: {len(train_set)}")
        print(f"Test samples: {len(test_set)}")
    return test_set, train_set


@app.cell(hide_code=True)
def _(mo):
    """
    Description: Render the model definition and training section heading.

    Input:
      - mo (module): Marimo.

    Output:
      - (none): renders a Markdown heading in the notebook UI.
    """
    mo.md(r"""## Model definition and training""")
    return


@app.cell(hide_code=True)
def _(mo):
    """
    Description: Render a short description of the SimpleGCN model used as a baseline.

    Input:
      - mo (module): Marimo.

    Output:
      - (none): renders a Markdown note in the notebook UI.
    """
    mo.md("""Define a simple two layer GCN model to get started""")
    return


@app.cell
def _(dataset):
    """
    Description: Instantiate a simple two-layer Graph Convolutional Network (SimpleGCN)
    sized to the dataset's feature dimensions. This is a homogeneous GCN baseline that
    takes node features and produces per-node outputs.

    Input:
      - dataset (BasicLocationsDataset): provides ``num_features`` (input dimensionality)
        and is used to size the output too.

    Output:
      - gcn (SimpleGCN): the instantiated (untrained) GCN model.
    """
    from ml.models import SimpleGCN

    gcn = SimpleGCN(in_channels=dataset.num_features, hidden_channels=32, out_channels=dataset.num_features)
    gcn
    return (gcn,)


@app.cell
def _(dataset):
    """
    Description: Instantiate a Multi-Layer Perceptron (MLP) baseline model. Unlike the
    GCN, the MLP ignores graph structure and processes flattened node features directly.
    It serves as a non-graph baseline for comparison.

    Input:
      - dataset (BasicLocationsDataset): provides ``num_nodes`` (graph size).

    Output:
      - mlp (MLP): the instantiated (untrained) MLP model.
    """
    from ml.models import MLP

    mlp = MLP(n_nodes=dataset.num_nodes, n_graph_x=1, hidden_channels=32, num_layers=2)
    mlp
    return (mlp,)


@app.cell(hide_code=True)
def _(mo):
    """
    Description: Render a note about the Experiment object that bundles train/test splits.

    Input:
      - mo (module): Marimo.

    Output:
      - (none): renders a Markdown note in the notebook UI.
    """
    mo.md(r"""We also define an experiment object that holds data on our train and test sets.""")
    return


@app.cell
def _(test_set, train_set):
    """
    Description: Create an ``Experiment`` object that bundles the train/test split,
    hyperparameters (number of epochs, batch size, validation fraction), and a random
    seed. The ``Experiment`` is passed to ``run_experiment`` to train any model using
    the same configuration.

    Input:
      - test_set: PyG test dataset.
      - train_set: PyG training dataset.

    Output:
      - experiment (Experiment): the experiment configuration object.
      - run_experiment (callable): the training loop function; exported for the training
        cells below.
    """
    from ml.experiment import Experiment, run_experiment

    experiment = Experiment(
        train_set=train_set,
        test_set=test_set,
        n_epochs=20,        # total training epochs
        val_size=0.15,      # fraction of train_set held out for validation
        batch_size=32,      # number of graphs per mini-batch
        random_state=42,    # seed for validation split
    )

    experiment
    return experiment, run_experiment


@app.cell(hide_code=True)
def _(mo):
    """
    Description: Render descriptions of the two benchmark models used for comparison.

    Input:
      - mo (module): Marimo.

    Output:
      - (none): renders a Markdown description in the notebook UI.
    """
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
    """
    Description: Display the distance matrix as a labelled Pandas DataFrame for easy
    inspection of node-to-node travel times in the synthetic network.

    Input:
      - synth_graph (SyntheticGraph): the synthetic network.

    Output:
      - (none): renders the distance matrix DataFrame in the notebook UI.
    """
    synth_graph.distance_matrix_df
    return


@app.cell
def _(synth_graph):
    """
    Description: Instantiate the two benchmark models for comparison:
      - ``EqualProbability``: assigns equal probability to every unvisited node
        (a simple uniform baseline).
      - ``BestGuess``: a model that knows the data generation process and computes
        the conditional probability of visiting each node given the sequence position.

    Input:
      - synth_graph (SyntheticGraph): the synthetic network (used to initialise BestGuess).

    Output:
      - best_model (BestGuess): the informed benchmark model.
      - compute_benchmark (callable): function that evaluates a benchmark on the experiment.
      - equal_model (EqualProbability): the naive uniform baseline model.
    """
    from ml.models import EqualProbability, BestGuess
    from ml.experiment import compute_benchmark

    equal_model = EqualProbability()               # uniform probability over unvisited nodes
    best_model = BestGuess.from_graph(synth_graph) # oracle that uses generation rules
    return best_model, compute_benchmark, equal_model


@app.cell
def _(best_model, compute_benchmark, equal_model, experiment, mo):
    """
    Description: Evaluate both benchmark models on the experiment's test set and print
    their summary results for comparison with the trained models.

    Input:
      - best_model (BestGuess): the informed benchmark.
      - compute_benchmark (callable): benchmark evaluation function.
      - equal_model (EqualProbability): the uniform baseline.
      - experiment (Experiment): the experiment configuration (provides test_set).
      - mo (module): Marimo (for redirect_stdout).

    Output:
      - best_results: evaluation results object for the BestGuess benchmark.
      - equal_results: evaluation results object for the EqualProbability baseline.
    """
    equal_results = compute_benchmark(experiment, equal_model)
    best_results = compute_benchmark(experiment, best_model)

    with mo.redirect_stdout():
        print(equal_results)
        print(best_results)
    return best_results, equal_results


@app.cell(hide_code=True)
def _(mo):
    """
    Description: Render a note that training is now performed for the SimpleGCN model.

    Input:
      - mo (module): Marimo.

    Output:
      - (none): renders a Markdown note in the notebook UI.
    """
    mo.md(r"""We then train our trainable models (in this case only `SimpleGCN`)""")
    return


@app.cell
def _(experiment, gcn, mo, run_experiment):
    """
    Description: Train the SimpleGCN model using the experiment configuration and print
    the final training, validation, and test losses.

    Input:
      - experiment (Experiment): the experiment configuration with train/val/test splits.
      - gcn (SimpleGCN): the untrained GCN model.
      - mo (module): Marimo (for redirect_stdout).
      - run_experiment (callable): the training loop function.

    Output:
      - gcn_results: the experiment results object with loss history and final metrics.
    """
    gcn_results = run_experiment(experiment, gcn)

    with mo.redirect_stdout():
        _train, _val, _test = gcn_results.final_losses()

        print(
            f"SimpleGCN:"
            f"Final losses after {gcn_results.n_epochs} epochs: "
            f"Training={_train:.4f} | Validation={_val:.4f} | Test={_test:.4f}"
        )
    return (gcn_results,)


@app.cell
def _(gcn_results, plotting):
    """
    Description: Plot the SimpleGCN training progress (loss curves over epochs) for
    visual inspection of convergence and potential overfitting.

    Input:
      - gcn_results: the GCN experiment results object.
      - plotting (module): local plotting helpers.

    Output:
      - (none): renders the training/validation/test loss curves in the notebook UI.
    """
    plotting.plot_training_progress(gcn_results)
    return


@app.cell
def _(experiment, mlp, mo, run_experiment):
    """
    Description: Train the MLP baseline model using the experiment configuration and
    print the final training, validation, and test losses for comparison with the GCN.

    Input:
      - experiment (Experiment): the experiment configuration.
      - mlp (MLP): the untrained MLP model.
      - mo (module): Marimo (for redirect_stdout).
      - run_experiment (callable): the training loop function.

    Output:
      - mlp_results: the MLP experiment results object with loss history.
    """
    mlp_results = run_experiment(experiment, mlp)

    with mo.redirect_stdout():
        _train, _val, _test = mlp_results.final_losses()

        print(
            f"MLP:"
            f"Final losses after {mlp_results.n_epochs} epochs: "
            f"Training={_train:.4f} | Validation={_val:.4f} | Test={_test:.4f}"
        )
    return (mlp_results,)


@app.cell
def _(mlp_results, plotting):
    """
    Description: Plot the MLP training progress (loss curves over epochs).

    Input:
      - mlp_results: the MLP experiment results object.
      - plotting (module): local plotting helpers.

    Output:
      - (none): renders the loss curves in the notebook UI.
    """
    plotting.plot_training_progress(mlp_results)
    return


@app.cell(hide_code=True)
def _(mo):
    """
    Description: Render the model comparisons sub-section heading.

    Input:
      - mo (module): Marimo.

    Output:
      - (none): renders a Markdown heading in the notebook UI.
    """
    mo.md(r"""### Model comparisons""")
    return


@app.cell
def _(best_results, equal_results, gcn_results, mlp_results, plotting):
    """
    Description: Create a bar chart comparing final test losses across all four models
    (GCN, MLP, BestGuess, EqualProbability) for a side-by-side performance overview.

    Input:
      - best_results, equal_results, gcn_results, mlp_results: results objects from
        previous training/evaluation cells.
      - plotting (module): local plotting helpers.

    Output:
      - (none): renders the bar comparison chart in the notebook UI.
    """
    plotting.plot_model_comparisons(gcn_results, mlp_results, best_results, equal_results, how="bar")
    return


@app.cell
def _(best_results, equal_results, gcn_results, mlp_results, plotting):
    """
    Description: Create a line chart comparing training loss curves across all four
    models (GCN, MLP, BestGuess, EqualProbability) over the training epochs.

    Input:
      - best_results, equal_results, gcn_results, mlp_results: results objects.
      - plotting (module): local plotting helpers.

    Output:
      - (none): renders the line comparison chart in the notebook UI.
    """
    plotting.plot_model_comparisons(gcn_results, mlp_results, best_results, equal_results, how="line")
    return


@app.cell(hide_code=True)
def _(
    best_model,
    experiment,
    gcn,
    get_prediction_idx,
    mlp,
    mo,
    n_test_samples,
    next_btn,
    plot_models,
    prev_btn,
):
    """
    Description: Interactive display cell — renders the Prev/Next navigation controls
    alongside a multi-model prediction comparison figure for the currently selected test-set sample.

    Input:
      - best_model (BestGuess): the informed benchmark.
      - experiment (Experiment): provides the test set.
      - gcn (SimpleGCN): the trained GCN model.
      - get_prediction_idx (callable): returns the current sample index.
      - mlp (MLP): the trained MLP model.
      - mo (module): Marimo.
      - n_test_samples (int): total test-set samples (for the index label).
      - next_btn (mo.ui.button): advances to the next sample.
      - plot_models (callable): generates the comparison figure.
      - prev_btn (mo.ui.button): goes back to the previous sample.

    Output:
      - (none): renders the navigation row and figure in the notebook UI.
    """
    sample = experiment.test_set[get_prediction_idx()]

    mo.vstack([
        mo.md("Comparison of predictions between models and benchmarks: "),
        mo.hstack([prev_btn, mo.md(f"Sample #{get_prediction_idx()}/{n_test_samples - 1}"), next_btn], align="center"),
        plot_models(sample, [gcn, mlp], [best_model]),
    ])
    return


@app.cell(hide_code=True)
def _(experiment, mo):
    """
    Description:
        Marimo cell that creates the interactive Prev/Next navigation buttons used
        to page through test-set predictions one sample at a time, and exposes the
        reactive state getter so downstream cells can read the current index.

    Input:
      - experiment: the loaded experiment object containing the test set and models.
      - mo: the Marimo runtime module used to create UI widgets and reactive state.

    Output:
      - get_prediction_idx (callable): reactive getter returning the current sample index.
      - n_test_samples (int): total number of test-set samples (used to clamp navigation).
      - next_btn (mo.ui.button): button that advances to the next sample.
      - prev_btn (mo.ui.button): button that goes back to the previous sample.
    """
    def create_prev_next_buttons(n_samples: int):
        """
        Description:
            Creates a stateful pair of "Prev" / "Next" Marimo buttons and a getter
            function that together let the user page through test-set samples one by one.

        Input:
          - n_samples (int): the total number of samples in the test set; used to clamp
                the index to the valid range [0, n_samples - 1].

        Output:
          - (tuple): a 3-tuple of
              prev_button (mo.ui.button) — clicking decrements the sample index.
              next_button (mo.ui.button) — clicking increments the sample index.
              get_prediction (callable)  — a getter that returns the current index (int).
        """
        # Reactive state: stores the current sample index, starts at 1
        get_prediction, set_prediction = mo.state(1)

        def _decrease(_):
            """
            Description:
                Button click handler that decrements the current sample index by 1,
                clamped so it never goes below 0.

            Input:
              - _ (any): the click event value (ignored; Marimo passes the button value).

            Output:
              - (None): updates reactive state via set_prediction as a side effect.
            """
            # Move to the previous sample, but never go below 0
            if get_prediction() > 0:
                set_prediction(lambda x: x - 1)

        def _increase(_):
            """
            Description:
                Button click handler that increments the current sample index by 1,
                clamped so it never exceeds the last valid index (n_samples - 1).

            Input:
              - _ (any): the click event value (ignored; Marimo passes the button value).

            Output:
              - (None): updates reactive state via set_prediction as a side effect.
            """
            # Move to the next sample, but never exceed the last index
            if get_prediction() < n_samples - 1:
                set_prediction(lambda x: x + 1)

        prev_button = mo.ui.button(on_click=_decrease, label="Prev")
        next_button = mo.ui.button(on_click=_increase, label="Next")

        return prev_button, next_button, get_prediction

    # Total number of test-set samples; used to clamp the navigation range
    n_test_samples = len(experiment.test_set)
    prev_btn, next_btn, get_prediction_idx = create_prev_next_buttons(n_test_samples)
    return get_prediction_idx, n_test_samples, next_btn, prev_btn


@app.cell(hide_code=True)
def _(F, math, plotting, plt, synth_graph, torch):
    """
    Description:
        Marimo cell that defines the plot_models helper and renders a multi-panel
        comparison figure for the sample currently selected by the Prev/Next buttons.
        Each panel shows the synthetic graph coloured by a different model's predicted
        visit probabilities, making it easy to compare GNN and baseline outputs.

    Input:
      - F: the torch.nn.functional module used for softmax normalisation.
      - math: Python's math module for mathematical operations (e.g. ceil).
      - plotting: the project's plotting utilities module.
      - plt: Matplotlib's pyplot module for figure creation.
      - synth_graph: the synthetic activity graph used as the shared graph structure.
      - torch: the PyTorch library for tensor operations.

    Output:
      - plot_models (callable): a function that accepts a sample, a list of trained
            models, and a list of benchmark models and returns a Matplotlib figure.
    """
    def plot_models(sample, models, benchmarks):
        """
        Description:
            Renders a grid of subplots comparing the actual node labels against the
            predictions of multiple trainable models and benchmark models for a single
            data sample.  Each subplot shows the synthetic graph with node colours
            indicating visit probability.

        Input:
          - sample (pyg.data.Data): a single PyG Data object from the test set.
          - models (list[nn.Module]): list of trained models that accept (sample) and
                return raw logits.  Probabilities are obtained via softmax.
          - benchmarks (list[nn.Module]): list of benchmark models that already output
                probabilities (no softmax applied).

        Output:
          - (matplotlib.figure.Figure): a figure with (len(models) + len(benchmarks) + 1)
                subplots arranged in rows of 4.
        """
        def _plot(sample, model, ax, is_benchmark):
            """
            Description:
                Draws a single model's prediction on the given axes, including the NLL loss
                in the subplot title.

            Input:
              - sample (pyg.data.Data): the data sample to predict on.
              - model (nn.Module): the model to use for prediction.
              - ax (matplotlib.axes.Axes): the axes to draw on.
              - is_benchmark (bool): if True, the model already returns probabilities and
                    log is applied directly; otherwise softmax is applied first.

            Output:
              - ax (matplotlib.axes.Axes): the updated axes after drawing.
            """
            model.eval()
            y_prob = model(sample).squeeze()  # raw output from the model

            if not is_benchmark:
                # Trainable model: convert logits to log-probabilities for NLL loss,
                # then convert to probabilities for visualisation
                loss = F.nll_loss(F.log_softmax(y_prob, dim=0), sample.y)
                y_prob = F.softmax(y_prob, dim=0)
            else:
                # Benchmark already outputs probabilities — apply log directly
                loss = F.nll_loss(torch.log(y_prob), sample.y)

            ax.set_title(f"{type(model).__name__} (loss={loss:.2f})")
            y_prob = y_prob.squeeze().tolist()  # convert to plain Python list for plotting
            return plotting.draw_prediction(synth_graph, sample.x, y_prob, ax=ax)

        # Total number of subplots: 1 (actual) + trainable models + benchmarks
        n_boxes = len(models) + len(benchmarks) + 1
        n_rows = math.ceil(n_boxes / 4)  # arrange in rows of 4

        fig, axs = plt.subplots(figsize=(12, 4 * n_rows), ncols=4, nrows=n_rows, squeeze=False)

        # Flatten the 2-D axes array into a simple list for indexed access
        flat_axs = [ax for a in axs for ax in a]

        # Turn off any surplus subplot axes beyond the number of models
        for ax in flat_axs[n_boxes:]:
            ax.set_axis_off()

        # First subplot: ground truth one-hot labels converted to float probabilities
        actual_y_prob = F.one_hot(sample.y, num_classes=sample.num_nodes).float()
        plotting.draw_prediction(synth_graph, sample.x, actual_y_prob.tolist(), ax=flat_axs[0], labels=False)
        flat_axs[0].set_title("Actual")

        # Subplots for trainable models
        for ax, model in zip(flat_axs[1 : len(models) + 1], models):
            _ = _plot(sample, model, ax, is_benchmark=False)

        # Subplots for benchmark models
        for ax, model in zip(flat_axs[len(models) + 1 :], benchmarks):
            _ = _plot(sample, model, ax, is_benchmark=True)

        fig.suptitle("Comparison of predictions")

        return fig

    return (plot_models,)


if __name__ == "__main__":
    app.run()
