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
def _(train_set):
    from models import GCNEncoder

    hidden_channels = 32
    latent_channels = 6

    gcn_encoder = GCNEncoder(
        in_node_channels=train_set.num_features,
        in_graph_channels=train_set.num_graph_features,
        hidden_channels=hidden_channels,
        num_layers=2,
        latent_channels=latent_channels,
    )

    gcn_encoder
    return hidden_channels, latent_channels


@app.cell
def _(hidden_channels, latent_channels, train_set):
    from models import MLPEncoder

    mlp_encoder = MLPEncoder(
        in_num_nodes=train_set[0].num_nodes,
        in_num_node_features=train_set.num_features,
        in_num_graph_features=train_set.num_graph_features,
        hidden_channels=hidden_channels,
        num_layers=3,
        latent_channels=latent_channels,
    )

    mlp_encoder
    return (mlp_encoder,)


@app.cell
def _(hidden_channels, latent_channels, train_set):
    from models import MLPDecoder

    label_decoder = MLPDecoder(
        latent_channels=latent_channels,
        hidden_channels=hidden_channels,
        num_layers=3,
        out_nodes=train_set[0].num_nodes,
        out_classes=train_set.num_classes,
    )

    label_decoder
    return (label_decoder,)


@app.cell
def _(label_decoder, mlp_encoder):
    from models import NodeLabelVGAE

    vgae = NodeLabelVGAE(mlp_encoder, label_decoder)
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
    return (DataLoader,)


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
def _(DataLoader, test_set, torch, train_set, val_set, vgae):
    lr = 0.0001
    n_epochs = 150
    batch_size = 32
    kl_weight = 0.000001

    model = vgae
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
        test_loader,
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
    from models import AlwaysZero, FiftyFifty
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
    test_loader,
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
        _model_score = evaluate_model(model, device, test_loader, _metric)
        _fifty_score = evaluate_model(FiftyFifty(), device, test_loader, _metric)
        _fifty_pct_diff = (_fifty_score - _model_score) / _model_score

        print(f"{_name.rjust(padding)} : VGAE={_model_score:.4f} - Base={_fifty_score:.4f} (% diff={_fifty_pct_diff:.2%})")
    return


@app.cell(hide_code=True)
def _():
    from torch_geometric.utils import to_dense_batch
    import torch.nn.functional as F
    return


@app.cell(hide_code=True)
def _(mo, test_set):
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


    n_test_samples = len(test_set)
    prev_btn, next_btn, get_prediction_idx = create_prev_next_buttons(n_test_samples)
    return get_prediction_idx, n_test_samples, next_btn, prev_btn


@app.cell(hide_code=True)
def _(get_prediction_idx, mo, n_test_samples, next_btn, plot_preds, prev_btn):
    mo.vstack(
        [
            mo.md("Comparison of predictions between models and benchmarks: "),
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
        _z = model.encode(_batch.x, _batch)

        print("Latents:", _z.squeeze().detach())

        _logits = model.decode(_z).detach()
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

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))

        plotting.draw_prediction(synth, _batch.x_labels.squeeze(), _y_true.squeeze(), ax=ax1)
        plotting.draw_prediction(synth, _batch.x_labels.squeeze(), _y_prob.squeeze(), ax=ax2)

        return fig
    return (plot_preds,)


if __name__ == "__main__":
    app.run()
