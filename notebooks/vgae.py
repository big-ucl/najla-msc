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
    import matplotlib.pyplot as plt
    return np, plt


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

    dataset = convert_to_pyg_dataset(schedules, label_reason_of_visit=False)
    dataset
    return (dataset,)


@app.cell
def _(dataset):
    from datasets import train_test_split

    train_val_set, test_set = train_test_split(dataset, random_state=42)
    train_set, val_set = train_test_split(train_val_set, random_state=42)
    train_set, val_set, test_set
    return test_set, train_set, val_set


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Defining the models""")
    return


@app.cell
def _():
    hidden_channels = 32
    latent_channels = 2
    return hidden_channels, latent_channels


@app.cell
def _(hidden_channels, latent_channels, train_set):
    from models import MLPEncoder, MLPDecoder, VAE

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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Train the model""")
    return


@app.cell
def _():
    import torch
    return (torch,)


@app.cell
def _():
    from models import EarlyStopping

    early_stop = EarlyStopping(verbose=True)
    return


@app.cell
def _(DataLoader, test_set, torch, train_set, vae, val_set):
    lr = 0.001
    n_epochs = 100
    batch_size = 32
    kl_weight = 0.00

    model = vae
    device = torch.device("cpu")
    train_loader = DataLoader(train_set, batch_size=batch_size)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)
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
    from losses import make_weighted_recon_loss, make_elbo_loss
    from experiment import train_vae_epoch, evaluate_model
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
    weighted_recon_loss = make_weighted_recon_loss(train_set)
    elbo_loss = make_elbo_loss(weighted_recon_loss, kl_weight=kl_weight)

    train_losses = []
    val_losses = []

    for epoch in range(n_epochs):
        model.train()

        train_loss = train_vae_epoch(model, device, train_loader, optimizer, elbo_loss)
        val_loss = evaluate_model(model, device, val_loader, weighted_recon_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # if early_stop.check(val_loss):
        #    pass # break

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: train_elbo={train_loss}, val_recon={val_loss}")
    return train_losses, val_losses, weighted_recon_loss


@app.cell
def _(n_epochs, plt, train_losses, val_losses):
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
    from models import FiftyFifty
    return (FiftyFifty,)


@app.cell
def _():
    from losses import roc_auc, average_precision, accuracy, recall, precision
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
    metrics = {
        "Weighted BCE": weighted_recon_loss,
        "ROC AUC": roc_auc,
        "Average Precision": average_precision,
        "Accuracy": accuracy,
        "Precision": recall,
        "Recall": precision,
    }

    padding = max(len(k) for k in metrics.keys())

    for _name, _metric in metrics.items():
        _model_score = evaluate_model(model, device, train_loader, _metric)
        _fifty_score = evaluate_model(FiftyFifty(), device, train_loader, _metric)
        _fifty_pct_diff = (_fifty_score - _model_score) / _model_score

        print(f"{_name.rjust(padding)} : VGAE={_model_score:.4f} - Base={_fifty_score:.4f} (% diff={_fifty_pct_diff:.2%})")
    return


@app.cell(hide_code=True)
def _(mo):
    dropdown = mo.ui.dropdown(options=["Train", "Val", "Test"], value="Train", allow_select_none=False, label="Dataset")
    return (dropdown,)


@app.cell(hide_code=True)
def _(dropdown, mo, test_set, train_set, val_set):
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


    _set = train_set if dropdown.value == "Train" else val_set if dropdown.value == "Val" else test_set
    n_test_samples = len(_set)
    prev_btn, next_btn, get_prediction_idx = create_prev_next_buttons(n_test_samples)
    return get_prediction_idx, n_test_samples, next_btn, prev_btn


@app.cell(hide_code=True)
def _(
    dropdown,
    get_prediction_idx,
    mo,
    n_test_samples,
    next_btn,
    plot_preds,
    prev_btn,
):
    mo.vstack(
        [
            mo.hstack([mo.md("Comparison of predictions between models and benchmarks: "), dropdown]),
            mo.hstack(
                [
                    prev_btn,
                    mo.md(f"Sample #{get_prediction_idx()}/{n_test_samples - 1}"),
                    next_btn,
                ],
                align="center",
            ),
            plot_preds(get_prediction_idx()),
        ]
    )
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
    test_set,
    torch,
):
    def print_metrics(_logits, _y_true, prefix):
        roc = roc_auc(_logits, _y_true)
        ap = average_precision(_logits, _y_true)
        acc = accuracy(_logits, _y_true)
        prec = precision(_logits, _y_true)
        rec = recall(_logits, _y_true)

        print(f"({prefix}) ROC AUC={roc:.4f}, AP={ap:.4f}, Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}")


    def plot_preds(i):
        model.eval()
        _d = test_set[i]

        _batch = next(iter(DataLoader([_d])))
        _logits, _mu, _log_var = model.forward(_batch.x, _batch)
        _z = model.reparameterize(_mu, _log_var)

        print("Latents:", _z.squeeze().detach())

        _y_prob = torch.sigmoid(_logits).detach()
        _y_pred = (_y_prob > 0.5).float().detach()
        _y_true = _batch.y.detach()

        _baseline_logits = FiftyFifty().decode(_batch)

        print("Logits:", _logits.squeeze())
        print()
        print("Probs :", _y_prob.squeeze())
        print("Preds :", _y_pred.squeeze())
        print("Actual:", _y_true.squeeze())

        print("\nMetrics:")
        print_metrics(_logits, _y_true, "model")
        print_metrics(_baseline_logits, _y_true, "base ")

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

        plotting.draw_prediction(synth, _batch.x_labels.squeeze(), _y_true.squeeze(), ax=ax1)
        plotting.draw_prediction(synth, _batch.x_labels.squeeze(), _y_prob.squeeze(), ax=ax2)

        return fig
    return (plot_preds,)


@app.cell(hide_code=True)
def _(batch, model, plt, torch, train_loader):
    def plot_latent_space():
        zs = []
        model.eval()

        for _batch in train_loader:
            with torch.no_grad():
                mu, log_std = model.encode(batch.x, batch)
                z = model.reparameterize(mu, log_std)

            zs.append(z)

        zs = torch.cat(zs)
        print(zs.shape)
        plt.figure(figsize=(8, 6))
        plt.grid(True)
        plt.scatter(zs[:, 0], zs[:, 1], alpha=0.7)
        plt.title("First two dimensions of the VAE latent space")
        plt.show()


    plot_latent_space()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
