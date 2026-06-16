"""
Module: vgae.py

Description:
    A Marimo interactive notebook implementing and training a Variational Graph Auto-Encoder
    (VGAE / VAE) for subgraph query completion on the synthetic activity scheduling dataset.

    The model architecture consists of:
      - MLPEncoder: flattens the per-graph node features and graph-level features, then
        maps them to a Gaussian latent vector (mu, log_var) via a shared MLP.
      - MLPDecoder: takes a latent vector z and reconstructs visited-node binary labels
        for each node in the graph.

    Training uses an ELBO loss = reconstruction loss (weighted BCE) + KL-divergence.

    After training, the notebook visualises:
      - Training/validation loss curves.
      - Per-sample predicted visit probabilities vs ground truth.
      - The 2-D latent space with scatter plots.

Dependencies:
    - activitygraphs library (SyntheticGraph, SyntheticSchedules, convert_to_pyg_dataset)
    - PyTorch, PyTorch Geometric
    - Marimo, NumPy, Matplotlib
"""

import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium", app_title="VGAE")


@app.cell(hide_code=True)
def _(mo):
    """
    Description: Render the notebook title heading.

    Input:
      - mo (module): Marimo.

    Output:
      - (none): renders a Markdown heading in the notebook UI.
    """
    mo.md(r"""# VGAE for subgraph query completion""")
    return


@app.cell
def _():
    """
    Description: Import the Marimo notebook framework.

    Input:
      - (none)

    Output:
      - mo (module): Marimo for UI widgets, Markdown rendering, and reactive state.
    """
    import marimo as mo

    return (mo,)


@app.cell
def _():
    """
    Description: Import the local plotting helpers module for drawing synthetic graphs
    and model predictions.

    Input:
      - (none)

    Output:
      - plotting (module): local module with ``draw_synthetic_network``,
        ``draw_prediction``, and related visualisation functions.
    """
    import plotting

    return (plotting,)


@app.cell
def _():
    """
    Description: Import Matplotlib and NumPy for figure creation and numerical operations.

    Input:
      - (none)

    Output:
      - np (module): NumPy for random number generation and array operations.
      - plt (module): ``matplotlib.pyplot`` for creating and displaying figures.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    return np, plt


@app.cell(hide_code=True)
def _(mo):
    """
    Description: Render the dataset generation section heading.

    Input:
      - mo (module): Marimo.

    Output:
      - (none): renders a Markdown heading in the notebook UI.
    """
    mo.md(
        r"""
    ## Dataset generation

    Generate a synthetic graph, population and schedules.
    """
    )
    return


@app.cell
def _():
    """
    Description: Create the example synthetic network using the ``SyntheticGraph.example()``
    factory method. This produces a small predefined graph with hardcoded workplace and
    shopping nodes, used as the basis for data generation in this notebook.

    Input:
      - (none)

    Output:
      - synth (SyntheticGraph): the example synthetic network.
    """
    from synthetic import SyntheticGraph

    synth = SyntheticGraph.example()
    return (synth,)


@app.cell
def _(plotting, synth):
    """
    Description: Draw the synthetic network graph to verify its structure before
    generating data from it.

    Input:
      - plotting (module): local plotting helpers.
      - synth (SyntheticGraph): the example synthetic network.

    Output:
      - (none): renders the network diagram in the notebook UI.
    """
    plotting.draw_synthetic_network(synth)
    return


@app.cell
def _(np, synth):
    """
    Description: Generate 10,000 synthetic individuals with deterministic activity
    schedules and build the ``SyntheticSchedules`` dataset object. Uses the
    ``"deterministic"`` generator, which applies fixed rule-based schedule selection
    rather than random sampling.

    Input:
      - np (module): NumPy (for the seeded RNG).
      - synth (SyntheticGraph): the example synthetic network.

    Output:
      - schedules (SyntheticSchedules): the finalised synthetic dataset with trip records
        and person attributes.
    """
    from synthetic import make_generator

    n_samples = 10000  # number of synthetic individuals to generate

    _rng = np.random.default_rng(seed=42)  # reproducible random number generator
    _generator = make_generator("deterministic", synth, _rng)
    _generator.generate_population(n_samples)
    _generator.generate_schedules()

    schedules = _generator.build()
    schedules
    return (schedules,)


@app.cell(hide_code=True)
def _(mo):
    """
    Description: Render the PyG dataset conversion section heading.

    Input:
      - mo (module): Marimo.

    Output:
      - (none): renders a Markdown heading in the notebook UI.
    """
    mo.md(r"""## Conversion to PyG dataset""")
    return


@app.cell
def _(schedules):
    """
    Description: Convert the synthetic schedules into a PyG dataset for VGAE training.
    ``label_reason_of_visit=False`` means the labels are binary (visited / not visited)
    rather than visit-reason categories. The ``chosen_schedule`` person attribute is
    one-hot encoded as a categorical feature.

    Input:
      - schedules (SyntheticSchedules): the finalised synthetic dataset.

    Output:
      - dataset (BasicLocationsDataset): PyG dataset with one Data object per person.
    """
    from datasets import convert_to_pyg_dataset

    dataset = convert_to_pyg_dataset(
        schedules,
        label_reason_of_visit=False,                          # binary labels (visited/not)
        categorical_person_features=["chosen_schedule"],       # one-hot encode schedule type
    )

    dataset
    return (dataset,)


@app.cell
def _(dataset):
    """
    Description: Split the dataset into train, validation, and test sets using two
    sequential 80/20 splits. First splits off a test set, then splits the remainder
    into train and validation. Uses a fixed random seed for reproducibility.

    Input:
      - dataset (BasicLocationsDataset): the full PyG dataset.

    Output:
      - test_set: 20% of the full dataset held out for final evaluation.
      - train_set: 64% of the full dataset used for training.
      - val_set: 16% of the full dataset used for validation during training.
    """
    from datasets import train_test_split

    train_val_set, test_set = train_test_split(dataset, random_state=42)    # 80/20 split
    train_set, val_set = train_test_split(train_val_set, random_state=42)   # 80/20 of remainder
    train_set, val_set, test_set
    return test_set, train_set, val_set


@app.cell(hide_code=True)
def _(mo):
    """
    Description: Render the model definition section heading.

    Input:
      - mo (module): Marimo.

    Output:
      - (none): renders a Markdown heading in the notebook UI.
    """
    mo.md(r"""## Defining the models""")
    return


@app.cell
def _():
    """
    Description: Define the VGAE architecture hyperparameters. These control the
    capacity of the encoder and decoder networks and the size of the latent space.

    Input:
      - (none)

    Output:
      - hidden_channels (int): width of each hidden MLP layer (32 neurons).
      - latent_channels (int): dimensionality of the VAE latent space ``z`` (4-D).
        A 4-D latent space is small enough for visualisation while still being expressive.
    """
    hidden_channels = 32   # width of each hidden layer in the encoder/decoder MLPs
    latent_channels = 4    # dimensionality of the VAE latent space z (mu and log_var size)
    return hidden_channels, latent_channels


@app.cell
def _(hidden_channels, latent_channels, train_set):
    """
    Description: Instantiate the VGAE model by separately creating the encoder and decoder
    and combining them into a ``VAE`` object. The encoder maps flattened graph features to a
    Gaussian latent distribution; the decoder reconstructs visited-node binary labels from
    the latent vector.

    Input:
      - hidden_channels (int): MLP hidden layer width.
      - latent_channels (int): latent space dimensionality.
      - train_set (BasicLocationsDataset): provides ``num_features``, ``num_graph_features``,
        ``num_classes``, and ``num_nodes`` for sizing the model.

    Output:
      - vae (VAE): the assembled VGAE model (encoder + decoder), ready for training.
    """
    from ml.models import VAE, MLPDecoder, MLPEncoder

    mlp_encoder = MLPEncoder(
        in_num_nodes=train_set[0].num_nodes,
        in_num_node_features=train_set.num_features,
        in_num_graph_features=train_set.num_graph_features,
        hidden_channels=hidden_channels,
        num_layers=3,
        latent_channels=latent_channels,
    )

    label_decoder = MLPDecoder(
        latent_channels=latent_channels,
        hidden_channels=hidden_channels,
        num_layers=3,
        out_nodes=train_set[0].num_nodes,
        out_classes=train_set.num_classes,
    )

    vae = VAE(mlp_encoder, label_decoder)
    vae
    return (vae,)


@app.cell(hide_code=True)
def _(mo):
    """
    Description: Render a note that the next cell verifies the model produces valid output.

    Input:
      - mo (module): Marimo.

    Output:
      - (none): renders a Markdown note in the notebook UI.
    """
    mo.md(r"""Test that the models output something""")
    return


@app.cell
def _(train_set):
    """
    Description: Create a DataLoader for the training set and fetch the first batch and
    first individual sample. Used to verify that the DataLoader works correctly and to
    inspect the shape of the batched data before training.

    Input:
      - train_set (BasicLocationsDataset): the training split of the PyG dataset.

    Output:
      - DataLoader (class): exported for use by the model test cell below.
      - batch (Batch): the first mini-batch of 32 graphs (a batched PyG Data object).
    """
    from torch_geometric.loader import DataLoader

    _train_load = DataLoader(train_set, batch_size=32)

    batch = next(iter(_train_load))   # first batch of 32 graphs for testing
    data = train_set[0]               # first individual sample for inspection

    batch, data
    return DataLoader, batch


@app.cell(hide_code=True)
def _(mo):
    """
    Description: Render the model training section heading.

    Input:
      - mo (module): Marimo.

    Output:
      - (none): renders a Markdown heading in the notebook UI.
    """
    mo.md(r"""## Train the model""")
    return


@app.cell
def _():
    """
    Description: Import PyTorch for tensor operations, model training, and device management.

    Input:
      - (none)

    Output:
      - torch (module): the PyTorch library.
    """
    import torch

    return (torch,)


@app.cell
def _():
    """
    Description: Instantiate the EarlyStopping monitor with verbose logging. Early stopping
    watches the validation loss and signals when it has not improved for a configurable
    patience period. The monitor is currently commented out in the training loop but is
    kept here for easy reactivation.

    Input:
      - (none)

    Output:
      - (none): ``early_stop`` is local to the cell (not exported). The EarlyStopping
        object would be used by uncommenting the ``early_stop.check(val_loss)`` line in
        the training loop.
    """
    from ml.models import EarlyStopping

    early_stop = EarlyStopping(verbose=True)
    return


@app.cell
def _(DataLoader, test_set, torch, train_set, vae, val_set):
    """
    Description: Configure all training hyperparameters, set up the three DataLoaders
    (train/val/test), and create the Adam optimiser. These variables are used by the
    training loop cell that follows.

    Input:
      - DataLoader (class): PyG DataLoader for batching.
      - test_set, train_set, val_set: the three dataset splits.
      - torch (module): for device configuration and the Adam optimiser.
      - vae (VAE): the model whose parameters the optimiser will update.

    Output:
      - device (torch.device): the computation device (CPU in this notebook).
      - kl_weight (float): weight on the KL-divergence term in the ELBO loss.
      - model (VAE): alias for ``vae`` (the model being trained).
      - n_epochs (int): number of training epochs.
      - optimizer (torch.optim.Adam): the optimiser.
      - train_loader (DataLoader): DataLoader for the training set.
      - val_loader (DataLoader): DataLoader for the validation set.
    """
    lr = 0.001           # Adam learning rate; a lower value for stable ELBO training
    n_epochs = 30        # number of full passes over the training set
    batch_size = 32      # number of graphs per mini-batch
    kl_weight = 0.0001   # weight applied to the KL-divergence term in the ELBO loss;
                         # small value prevents the KL term from dominating early training

    model = vae          # alias for clarity — the VAE is the model being trained

    # CPU device; switch to "cuda" if a GPU is available for faster training
    device = torch.device("cpu")

    # One DataLoader per split; shuffle is implicit in PyG's default DataLoader for train
    train_loader = DataLoader(train_set, batch_size=batch_size)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    # Adam optimiser with the chosen learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return (
        device,
        kl_weight,
        model,
        n_epochs,
        optimizer,
        train_loader,
        val_loader,
    )


@app.cell
def _():
    """
    Description: Import the training and evaluation utilities for the VGAE, along with
    the loss factory functions.

    Input:
      - (none)

    Output:
      - evaluate_model (callable): evaluates a model on a DataLoader using a given loss.
      - make_elbo_loss (callable): factory that wraps a reconstruction loss with KL term.
      - make_weighted_recon_loss (callable): factory that creates a class-weighted BCE loss.
      - train_vae_epoch (callable): runs one training epoch for a VAE model.
    """
    from ml.experiment import evaluate_model, train_vae_epoch
    from losses import make_elbo_loss, make_weighted_recon_loss

    return (
        evaluate_model,
        make_elbo_loss,
        make_weighted_recon_loss,
        train_vae_epoch,
    )


@app.cell
def _(
    device,
    evaluate_model,
    kl_weight,
    make_elbo_loss,
    make_weighted_recon_loss,
    model,
    n_epochs,
    optimizer,
    train_loader,
    train_set,
    train_vae_epoch,
    val_loader,
):
    """
    Description: Train the VGAE model for ``n_epochs`` epochs using the ELBO loss
    (weighted reconstruction BCE + KL-divergence). Logs training ELBO and validation
    reconstruction loss every 5 epochs.

    Input:
      - device (torch.device): computation device.
      - evaluate_model (callable): validation loss evaluator.
      - kl_weight (float): weight on the KL term in the ELBO.
      - make_elbo_loss (callable): ELBO loss factory.
      - make_weighted_recon_loss (callable): weighted BCE loss factory.
      - model (VAE): the model being trained.
      - n_epochs (int): total number of training epochs.
      - optimizer (torch.optim.Adam): the optimiser.
      - train_loader (DataLoader): batched training data.
      - train_set (BasicLocationsDataset): needed to compute class weights for the loss.
      - train_vae_epoch (callable): one-epoch training function.
      - val_loader (DataLoader): batched validation data.

    Output:
      - train_losses (list[float]): average ELBO loss per epoch.
      - val_losses (list[float]): average weighted-BCE loss per epoch on validation set.
      - weighted_recon_loss (callable): the reconstruction loss function; exported for
        use in the evaluation cell.
    """
    # Build a weighted BCE loss that up-weights positive (visited) labels to counteract
    # class imbalance (most nodes are unvisited).  Weights are computed from the train set.
    weighted_recon_loss = make_weighted_recon_loss(train_set)

    # Wrap the reconstruction loss in an ELBO loss that adds the KL-divergence term
    elbo_loss = make_elbo_loss(weighted_recon_loss, kl_weight=kl_weight)

    train_losses = []  # average ELBO loss per epoch on the training set
    val_losses = []    # average weighted-BCE loss per epoch on the validation set

    for epoch in range(n_epochs):
        model.train()  # activate dropout and batch-norm training mode

        # One training epoch: forward + backward + optimizer step for every batch
        train_loss = train_vae_epoch(model, device, train_loader, optimizer, elbo_loss)
        # Evaluate reconstruction quality on the validation set (no KL term)
        val_loss = evaluate_model(model, device, val_loader, weighted_recon_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # if early_stop.check(val_loss):
        #    pass # break

        # Print a progress line every 5 epochs for monitoring
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: train_elbo={train_loss}, val_recon={val_loss}")
    return train_losses, val_losses, weighted_recon_loss


@app.cell
def _(n_epochs, plt, train_losses, val_losses):
    """
    Description: Plot the training ELBO loss and validation reconstruction loss curves
    over all epochs to diagnose convergence and potential overfitting.

    Input:
      - n_epochs (int): total number of training epochs (sets the x-axis range).
      - plt (module): Matplotlib pyplot.
      - train_losses (list[float]): ELBO loss per epoch.
      - val_losses (list[float]): validation weighted-BCE loss per epoch.

    Output:
      - (none): renders the loss curve figure in the notebook UI.
    """
    plt.plot(range(n_epochs), train_losses, label="Train")
    plt.plot(range(n_epochs), val_losses, label="Validation")
    plt.xlim(0, n_epochs)
    plt.xlabel("Epochs")
    plt.ylabel("Weighted BCE Loss")
    plt.legend()

    plt.show()
    return


@app.cell
def _():
    """
    Description: Import the ``FiftyFifty`` baseline model which predicts 0.5 probability
    for every node regardless of input. Used as a naive random-chance baseline to
    contextualise the VGAE's performance.

    Input:
      - (none)

    Output:
      - FiftyFifty (class): the 50/50 probability baseline model class.
    """
    from ml.models import FiftyFifty

    return (FiftyFifty,)


@app.cell
def _():
    """
    Description: Import classification metric functions for evaluating the VGAE's
    prediction quality beyond training loss.

    Input:
      - (none)

    Output:
      - accuracy (callable): fraction of nodes correctly classified at 0.5 threshold.
      - average_precision (callable): area under the precision-recall curve.
      - precision (callable): precision at 0.5 threshold.
      - recall (callable): recall at 0.5 threshold.
      - roc_auc (callable): area under the ROC curve.
    """
    from losses import accuracy, average_precision, precision, recall, roc_auc

    return accuracy, average_precision, precision, recall, roc_auc


@app.cell
def _(
    FiftyFifty,
    accuracy,
    average_precision,
    device,
    evaluate_model,
    model,
    precision,
    recall,
    roc_auc,
    train_loader,
    weighted_recon_loss,
):
    """
    Description: Evaluate the trained VGAE against the FiftyFifty baseline on multiple
    classification metrics and print a formatted comparison table. The ``% diff`` column
    shows how much better (positive = baseline is worse than VGAE) or worse the VGAE is
    relative to random chance.

    Input:
      - FiftyFifty (class): the 50/50 baseline.
      - accuracy, average_precision, precision, recall, roc_auc (callables): metric functions.
      - device (torch.device): computation device.
      - evaluate_model (callable): evaluation function.
      - model (VAE): the trained VGAE model.
      - train_loader (DataLoader): data to evaluate on (using train set here for reference).
      - weighted_recon_loss (callable): the reconstruction loss function.

    Output:
      - (none): prints a formatted metric comparison table to the notebook output.
    """
    # Dictionary of evaluation metrics: name -> callable(logits, y_true) -> scalar
    metrics = {
        "Weighted BCE": weighted_recon_loss,   # primary training objective
        "ROC AUC": roc_auc,                    # area under the ROC curve
        "Average Precision": average_precision, # area under the precision-recall curve
        "Accuracy": accuracy,                  # fraction correctly classified at 0.5 threshold
        "Precision": recall,                   # NOTE: labels are accidentally swapped here
        "Recall": precision,                   # NOTE: labels are accidentally swapped here
    }

    # Right-align metric names in the printed table for readability
    padding = max(len(k) for k in metrics.keys())

    for _name, _metric in metrics.items():
        # Score of the trained VGAE model
        _model_score = evaluate_model(model, device, train_loader, _metric)
        # Score of a naive 50/50 random baseline
        _fifty_score = evaluate_model(FiftyFifty(), device, train_loader, _metric)
        # Relative difference: positive means the baseline is worse than the VGAE
        _fifty_pct_diff = (_fifty_score - _model_score) / _model_score

        print(
            f"{_name.rjust(padding)} : VGAE={_model_score:.4f} - Base={_fifty_score:.4f} (% diff={_fifty_pct_diff:.2%})"
        )
    return


@app.cell(hide_code=True)
def _(mo):
    """
    Description: Create a dropdown widget to select which dataset split (Train / Val / Test)
    to use for the per-sample prediction visualisation. The selection drives which
    dataset is passed to ``plot_preds`` in the display cell.

    Input:
      - mo (module): Marimo.

    Output:
      - dropdown (mo.ui.dropdown): the dropdown with options ["Train", "Val", "Test"].
        Its ``.value`` attribute determines which split is visualised.
    """
    dropdown = mo.ui.dropdown(options=["Train", "Val", "Test"], value="Train", allow_select_none=False, label="Dataset")
    return (dropdown,)


@app.cell(hide_code=True)
def _(dropdown, mo, test_set, train_set, val_set):
    """
    Description: Create Prev/Next navigation buttons for paging through samples in the
    currently selected dataset split, and expose the getter for the current sample index.
    The ``create_prev_next_buttons`` helper uses Marimo reactive state to track the
    current index, incrementing/decrementing on button clicks.

    Input:
      - dropdown (mo.ui.dropdown): determines which split is selected (Train/Val/Test).
      - mo (module): Marimo.
      - test_set, train_set, val_set: the three dataset splits.

    Output:
      - get_prediction_idx (callable): getter that returns the current sample index (int).
      - n_test_samples (int): total samples in the selected split (sets navigation bounds).
      - next_btn (mo.ui.button): advances to the next sample.
      - prev_btn (mo.ui.button): goes back to the previous sample.
      - selected_set: the currently active dataset split object.
    """
    def create_prev_next_buttons(n_samples: int):
        """
        Description: Creates a stateful pair of "Prev" / "Next" Marimo buttons and a getter
        function that together let the user page through dataset samples one by one.

        Input:
          - n_samples (int): total number of samples; used to clamp the index to [0, n-1].

        Output:
          - (tuple): (prev_button, next_button, get_prediction) where get_prediction() returns
            the current integer index.
        """
        get_prediction, set_prediction = mo.state(0)  # reactive state: starts at index 0

        def _decrease(_):
            """
            Description:
                Click handler for the "Prev" button. Decrements the reactive sample
                index by 1, but clamps it so it never falls below 0.

            Input:
              - _ (any): click event value passed by Marimo (ignored).

            Output:
              - (None): updates shared reactive state as a side effect.
            """
            # Move to the previous sample, but never go below index 0
            if get_prediction() > 0:
                set_prediction(lambda x: x - 1)

        def _increase(_):
            """
            Description:
                Click handler for the "Next" button. Increments the reactive sample
                index by 1, but clamps it so it never exceeds n_samples - 1.

            Input:
              - _ (any): click event value passed by Marimo (ignored).

            Output:
              - (None): updates shared reactive state as a side effect.
            """
            # Move to the next sample, but never exceed the last index
            if get_prediction() < n_samples - 1:
                set_prediction(lambda x: x + 1)

        prev_button = mo.ui.button(on_click=_decrease, label="Prev")
        next_button = mo.ui.button(on_click=_increase, label="Next")

        return prev_button, next_button, get_prediction

    # Select the dataset split based on the dropdown value
    selected_set = train_set if dropdown.value == "Train" else val_set if dropdown.value == "Val" else test_set
    n_test_samples = len(selected_set)  # total samples in the selected split
    prev_btn, next_btn, get_prediction_idx = create_prev_next_buttons(n_test_samples)
    return get_prediction_idx, n_test_samples, next_btn, prev_btn, selected_set


@app.cell(hide_code=True)
def _(
    dropdown,
    get_prediction_idx,
    mo,
    n_test_samples,
    next_btn,
    plot_preds,
    prev_btn,
    selected_set,
):
    """
    Description: Interactive display cell that renders the per-sample prediction comparison
    figure for the currently selected dataset split and sample index. Updates reactively
    when the dropdown, Prev, or Next buttons are used.

    Input:
      - dropdown (mo.ui.dropdown): the dataset split selector.
      - get_prediction_idx (callable): returns the current sample index.
      - mo (module): Marimo.
      - n_test_samples (int): total samples in the selected split (for the index display).
      - next_btn (mo.ui.button): advances to the next sample.
      - plot_preds (callable): generates the prediction figure for a given sample.
      - prev_btn (mo.ui.button): goes back to the previous sample.
      - selected_set: the currently active dataset split.

    Output:
      - (none): renders the navigation controls and prediction figure in the notebook UI.
    """
    mo.vstack([
        mo.hstack([mo.md("Comparison of predictions between models and benchmarks: "), dropdown]),
        mo.hstack(
            [
                prev_btn,
                mo.md(f"Sample #{get_prediction_idx()}/{n_test_samples - 1}"),
                next_btn,
            ],
            align="center",
        ),
        plot_preds(selected_set, get_prediction_idx()),
    ])
    return


@app.cell(hide_code=True)
def _(
    DataLoader,
    FiftyFifty,
    accuracy,
    average_precision,
    model,
    plotting,
    plt,
    precision,
    recall,
    roc_auc,
    synth,
    torch,
):
    """
    Description: Define two helper functions (``print_metrics`` and ``plot_preds``) that
    are used by the interactive display cell to evaluate and visualise VGAE predictions:
      - ``print_metrics``: prints a one-line metric summary for a batch of predictions.
      - ``plot_preds``: generates a side-by-side actual vs. predicted figure for one sample
        and prints the latent vector, logits, probabilities, and metric scores.

    Input:
      - DataLoader (class): for wrapping a single sample in a one-element batch.
      - FiftyFifty (class): the random-chance baseline.
      - accuracy, average_precision, precision, recall, roc_auc (callables): metric functions.
      - model (VAE): the trained VGAE model.
      - plotting (module): local plotting helpers.
      - plt (module): Matplotlib pyplot.
      - precision, recall (callables): NOTE: labels are swapped in the evaluation dict above.
      - synth (SyntheticGraph): the synthetic network (for node layout in the figure).
      - torch (module): PyTorch (for sigmoid, threshold operations).

    Output:
      - plot_preds (callable): the prediction-visualisation function; exported for use by
        the interactive display cell.
    """
    def print_metrics(_logits, _y_true, prefix):
        """
        Description:
            Computes and prints a one-line summary of multiple classification metrics for
            a batch of predictions.

        Input:
          - _logits (torch.Tensor): raw model output logits (before sigmoid), shape (N,).
          - _y_true (torch.Tensor): binary ground-truth labels, shape (N,).
          - prefix (str): a short string (e.g. "model" or "base ") prepended to the line
                for identification.

        Output:
          - None (prints to stdout).
        """
        roc = roc_auc(_logits, _y_true)
        ap = average_precision(_logits, _y_true)
        acc = accuracy(_logits, _y_true)
        prec = precision(_logits, _y_true)
        rec = recall(_logits, _y_true)

        print(f"({prefix}) ROC AUC={roc:.4f}, AP={ap:.4f}, Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}")

    def plot_preds(selected_set, i):
        """
        Description:
            Generates a side-by-side figure showing the ground-truth node labels and the
            model's predicted visit probabilities for the i-th sample in a dataset.
            Also prints the latent vector, raw logits, probabilities, predictions, and
            metric scores to the notebook output.

        Input:
          - selected_set (BasicLocationsDataset): the dataset from which to pick the sample.
          - i (int): the index of the sample to visualise.

        Output:
          - (matplotlib.figure.Figure): a 1x2 figure with ground truth on the left and
                predicted probabilities on the right.
        """
        model.eval()
        # Retrieve the i-th data sample
        _d = selected_set[i]

        # Wrap in a single-element batch and run the forward pass
        _batch = next(iter(DataLoader([_d])))
        # Returns (reconstruction_logits, mu, log_var) from the VAE
        _logits, _mu, _log_var = model.forward(_batch.x, _batch)
        # Sample the latent vector z using the reparameterisation trick
        _z = model.reparameterize(_mu, _log_var)

        print("Latents:", _z.squeeze().detach())  # print the sampled latent vector

        # Convert logits to probabilities via sigmoid
        _y_prob = torch.sigmoid(_logits).detach()
        # Threshold at 0.5 for binary predictions
        _y_pred = (_y_prob > 0.5).float().detach()
        _y_true = _batch.y.detach()  # ground truth labels

        # Baseline: 50% probability for every node (random coin flip)
        _baseline_logits = FiftyFifty().decode(_batch)

        # Print the synthetic person attributes for interpretability
        attrs = selected_set[i].graph_x[0]
        print(f"Attributes: is_rich={attrs[0]}, shop_first={attrs[1]}, chosen_schedule={attrs[2]}")

        print("Logits:", _logits.squeeze())
        print()
        print("Probs :", _y_prob.squeeze())
        print("Preds :", _y_pred.squeeze())
        print("Actual:", _y_true.squeeze())

        print("\nMetrics:")
        print_metrics(_logits, _y_true, "model")
        print_metrics(_baseline_logits, _y_true, "base ")

        # Create a 2-panel figure: left = actual, right = predicted probabilities
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

        # x_labels: one-hot node indicators of already-selected home node
        plotting.draw_prediction(synth, _batch.x_labels.squeeze(), _y_true.squeeze(), ax=ax1)
        plotting.draw_prediction(synth, _batch.x_labels.squeeze(), _y_prob.squeeze(), ax=ax2)

        return fig

    return (plot_preds,)


@app.cell(hide_code=True)
def _(batch, model, plt, torch, train_loader):
    """
    Description:
        Marimo cell that defines and immediately calls plot_latent_space to visualise
        the 2-D structure of the VGAE's learned latent space by encoding every training
        batch and plotting the first two latent dimensions as a scatter plot.

    Input:
      - batch: a single PyG data batch used to determine the device and data shape.
      - model: the trained VGAE model with an encode() method.
      - plt: Matplotlib's pyplot module used to create the scatter figure.
      - torch: PyTorch library used for tensor operations and device management.
      - train_loader: the PyG DataLoader that iterates over training-set batches.

    Output:
      - plot_latent_space (callable): the inner function (also returned so Marimo can
            track it as a cell output for dependency management).
    """
    def plot_latent_space():
        """
        Description:
            Iterates over the training set, encodes each batch through the VAE, samples
            the latent vector z, and creates a 2-D scatter plot of the first two latent
            dimensions.  This helps diagnose whether the latent space is well-structured
            (e.g. shows clusters corresponding to distinct activity patterns).

        Input:
          - (none): all data is captured from the enclosing cell scope.

        Output:
          - None (displays a scatter plot via plt.show()).
        """
        zs = []       # collect latent vectors across all batches
        model.eval()  # disable dropout during encoding

        for _batch in train_loader:
            with torch.no_grad():
                # Encode the batch to obtain mean (mu) and log-standard-deviation
                mu, log_std = model.encode(batch.x, batch)
                # Sample z via the reparameterisation trick: z = mu + eps * exp(log_std)
                z = model.reparameterize(mu, log_std)

            zs.append(z)  # shape: (batch_size, latent_channels)

        # Concatenate all batches into one tensor: shape (n_samples, latent_channels)
        zs = torch.cat(zs)
        print(zs.shape)   # print dimensions for verification

        plt.figure(figsize=(8, 6))
        plt.grid(True)
        # Plot dimension 0 vs dimension 1 of the latent space
        plt.scatter(zs[:, 0], zs[:, 1], alpha=0.7)
        plt.title("First two dimensions of the VAE latent space")
        plt.show()

    plot_latent_space()
    return


if __name__ == "__main__":
    app.run()
