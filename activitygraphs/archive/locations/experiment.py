from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import BasicLocationsDataset, train_test_split
from ml.models import Benchmark
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch
from utils import check_schema


@dataclass(frozen=True)
class Experiment:
    """Represents an experiment on which to train & test ML models."""

    train_set: BasicLocationsDataset
    test_set: BasicLocationsDataset
    n_epochs: int
    val_size: int
    batch_size: int
    random_state: int

    @property
    def n_nodes(self) -> int:
        """Returns the number of nodes in the graph"""
        return self.train_set.num_classes


class Results:
    """Represents the results of training / testing an ML model on an `Experiment`."""

    LOSSES_SCHEMA = pl.Schema({
        "epoch": pl.Int64,
        "loss": pl.Float64,
        "type": pl.String,
        "name": pl.String,
    })

    def __init__(self, name: str, losses: pl.DataFrame):
        """
        Args:
            name (str): The name given to the results, typically the model name
            losses (pl.DataFrame): A DataFrame containing the train / val / test losses per epoch
        """
        self.name = name
        self.losses = check_schema(losses, self.LOSSES_SCHEMA)
        self.n_epochs = losses["epoch"].max()

    def has_training_history(self):
        """
        Description: Returns True if this Results object contains training and validation loss
        history (i.e. the model was actually trained, not just benchmarked). Used to determine
        how to plot or summarise results.

        Output:
          - (bool): True if both 'train' and 'val' loss types are present in the losses DataFrame.
        """
        return "train" in self.losses["type"] and "val" in self.losses["type"]

    def _losses(self, loss_type: str):
        """
        Description: Internal helper that filters the losses DataFrame to rows of a specific type
        (e.g. 'train', 'val', or 'test'). Raises an error if no entries of that type exist.

        Input:
          - loss_type (str): The type of loss to retrieve, e.g. 'train', 'val', or 'test'.

        Output:
          - (pl.DataFrame): A filtered DataFrame containing only losses of the requested type.
        """
        losses = self.losses.filter(pl.col("type") == loss_type)

        if len(losses) == 0:
            raise ValueError(f"No loss of type {loss_type}")

        return losses

    def _latest_loss(self, loss_type: str) -> float:
        """
        Description: Internal helper that retrieves the scalar loss value at the final epoch for a
        given loss type. This is used for reporting final train, val, or test performance.

        Input:
          - loss_type (str): The type of loss, e.g. 'train', 'val', or 'test'.

        Output:
          - (float): The loss value at the last recorded epoch for that loss type.
        """
        return self._losses(loss_type).filter(pl.col("epoch") == pl.col("epoch").max())["loss"][0]

    def train_losses(self):
        """
        Description: Returns the training loss at every epoch as a Series.
        Useful for plotting the learning curve.

        Output:
          - (pl.Series): A polars Series of float training loss values, one per epoch.
        """
        return self._losses("train")["loss"]

    def val_losses(self):
        """
        Description: Returns the validation loss at every epoch as a Series.
        Useful for plotting the learning curve and detecting overfitting.

        Output:
          - (pl.Series): A polars Series of float validation loss values, one per epoch.
        """
        return self._losses("val")["loss"]

    def test_losses(self):
        """
        Description: Returns the test loss values as a Series (usually one entry, the final
        evaluation on the held-out test set).

        Output:
          - (pl.Series): A polars Series of float test loss values.
        """
        return self._losses("test")["loss"]

    def test_loss(self):
        """
        Description: Returns the final (single) test loss as a scalar float — the model's
        performance on the held-out test set at the end of training.

        Output:
          - (float): The test loss value.
        """
        return self._latest_loss("test")

    def final_losses(self) -> tuple[float, float, float]:
        """
        Description: Returns the final train, validation, and test losses as a tuple. Useful for
        summarising model performance after training.

        Output:
          - (tuple[float, float, float]): A tuple of (final_train_loss, final_val_loss, test_loss).
        """
        return self._latest_loss("train"), self._latest_loss("val"), self.test_loss()

    def __repr__(self):
        """
        Description: Returns a human-readable string representation of the Results object,
        showing the run name and the final test loss rounded to 4 decimal places.

        Output:
          - (str): A short summary string, e.g. 'Results(MyModel | Test loss=0.3214)'.
        """
        return f"Results({self.name} | Test loss={self.test_loss():.4f})"


def create_loss(
    n_classes: int, with_logits=False, epsilon=0.0001
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Description: Creates and returns a loss function (closure) for multi-class node classification.
    The loss function reshapes model output to (batch_size, n_classes) before computing either
    cross-entropy (when the model outputs raw logits) or negative log-likelihood (when the model
    outputs probabilities). The epsilon prevents taking log(0) in NLL mode.

    Input:
      - n_classes (int): The number of output classes (graph nodes / location choices).
      - with_logits (bool): If True, uses cross_entropy (expects raw logits from model).
        If False, uses NLL loss (expects probabilities from model). Defaults to False.
      - epsilon (float): Small value added before taking log in NLL mode to avoid log(0).
        Defaults to 0.0001.

    Output:
      - (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): A loss function that takes
        (model_output, ground_truth_labels) and returns a scalar loss tensor.
    """
    def loss(out: torch.Tensor, y: torch.Tensor):
        """
        Description: Inner loss function created by create_loss. Reshapes the model output and
        computes the appropriate classification loss.

        Input:
          - out (torch.Tensor): Raw model output, will be reshaped to (-1, n_classes).
          - y (torch.Tensor): Ground-truth class labels.

        Output:
          - (torch.Tensor): Scalar loss value.
        """
        out = out.reshape((-1, n_classes))  # Flatten batch × nodes into a 2D (samples, classes) tensor
        if with_logits:
            return F.cross_entropy(out, y)  # Applies softmax internally; expects raw logits
        else:
            return F.nll_loss(torch.log(out + epsilon), y)  # Expects log-probabilities; epsilon avoids log(0)

    return loss


def run_experiment(
    experiment: Experiment,
    model: nn.Module,
    lr=0.01,
    name: str = None,
    verbose: int | None = 5,
) -> Results:
    """Trains a Model on the Experiment train set and evaluates the model on the test set.

    Args:
        experiment (Experiment): The experiment to train/test the model on
        model (nn.Module): The model to be trained
        lr (float, optional): The Adam learning rate. Defaults to 0.01.
        name (str, optional): The name given to the run. Defaults to the model name if left `None`.
        verbose (int | None, optional): How often to print intermediate results, does not print if None. Defaults to 5.

    Returns:
        Results: the results from the training & testing
    """
    exp = experiment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    name = model.__class__.__name__ if name is None else name
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = create_loss(n_classes=experiment.n_nodes, with_logits=True)

    train_set, val_set = train_test_split(exp.train_set, test_size=exp.val_size, random_state=exp.random_state)
    train_loader = DataLoader(dataset=train_set, batch_size=exp.batch_size)
    val_loader = DataLoader(dataset=val_set, batch_size=exp.batch_size)

    train_losses = []
    val_losses = []

    if verbose is not None:
        print(f"======= {name} (n_epochs={exp.n_epochs}) =======")

    # Training phase
    for epoch in range(exp.n_epochs):
        train_loss = train_epoch(model, device, train_loader, optimizer, criterion)
        val_loss = evaluate_model(model, device, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if verbose is not None and epoch % verbose == 0:
            print(f"Epoch {epoch + 1:3}, Training loss: {train_loss:.4f} | Validation loss: {val_loss:.4f}")

    # Evaluate on the test set
    test_loader = DataLoader(dataset=exp.test_set, batch_size=exp.batch_size)
    test_loss = evaluate_model(model, device, test_loader, criterion)

    # Build results DataFrame
    epochs = list(range(1, exp.n_epochs + 1))
    losses = pl.concat([
        pl.DataFrame({"epoch": epochs, "loss": train_losses}).with_columns(pl.lit("train").alias("type")),
        pl.DataFrame({"epoch": epochs, "loss": val_losses}).with_columns(pl.lit("val").alias("type")),
        pl.DataFrame({"epoch": exp.n_epochs, "loss": test_loss}).with_columns(pl.lit("test").alias("type")),
    ]).with_columns(pl.lit(name).alias("name"))

    return Results(name, losses)


def compute_benchmark(experiment: Experiment, benchmark_model: Benchmark, name: str = None) -> Results:
    """Evaluates a benchmark model on the experiment test set using CE loss

    Args:
        experiment (Experiment): the experiment to evaluate over
        benchmark_model (models.Benchmark): the benchmark model to be evaluated
        name (str, optional): The name given to the run. Defaults to the model name if left `None`.

    Returns:
        Results: the results from the testing
    """
    name = benchmark_model.__class__.__name__ if name is None else name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    benchmark_model = benchmark_model.to(device)

    loader = DataLoader(dataset=experiment.test_set, batch_size=experiment.batch_size)
    loss = evaluate_model(benchmark_model, device, loader, create_loss(n_classes=experiment.n_nodes, with_logits=False))

    return Results(name, pl.DataFrame({"name": name, "epoch": 0, "loss": loss, "type": "test"}))


def train_epoch(
    model: nn.Module,
    device: torch.device,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> float:
    """Runs a single epoch of training over a model.

    Args:
        model (nn.Module): The model to be trained
        device (torch.device): The device on which to run computations
        loader (DataLoader): The DataLoader containing the training data
        optimizer (torch.optim.Optimizer): the optimizer
        criterion (torch.nn.Module): The loss function

    Returns:
        float: the average training loss (normalized, per sample)
    """
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


class VAE(Protocol):
    """
    Description: A structural Protocol (interface) that describes the expected API of a
    Variational Autoencoder (VAE) model used in this project. Any object implementing
    these methods can be used wherever a VAE is expected (e.g. in train_vae_epoch and
    evaluate_model). A VAE encodes inputs into a latent distribution (mu, log_std),
    samples a latent vector z, and decodes z back to a reconstruction.
    """
    def train(self):
        """
        Description: Puts the model into training mode (enables dropout, batch norm, etc.).
        This is the standard PyTorch nn.Module interface method.
        """
        pass

    def forward(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Description: Runs a full forward pass through the VAE: encode → reparameterise → decode.

        Output:
          - (tuple[torch.Tensor, torch.Tensor, torch.Tensor]): A tuple of
              (reconstruction_y, latent_mean_mu, latent_log_std).
        """
        pass

    def reparametrize(self, mu: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        """
        Description: Applies the VAE reparameterisation trick: samples z = mu + eps * exp(log_std),
        where eps ~ N(0, 1). This makes the sampling step differentiable for backpropagation.

        Input:
          - mu (torch.Tensor): The mean of the latent distribution, shape=(batch, latent_dim).
          - log_std (torch.Tensor): The log standard deviation, shape=(batch, latent_dim).

        Output:
          - (torch.Tensor): Sampled latent vector z, shape=(batch, latent_dim).
        """
        pass

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Description: Decodes a latent vector z back into the output space (e.g. node label probabilities).

        Input:
          - z (torch.Tensor): Latent vector sampled from the posterior, shape=(batch, latent_dim).

        Output:
          - (torch.Tensor): Reconstructed output (node label predictions).
        """
        pass


def train_vae_epoch(
    model: VAE,
    device: torch.device,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: Callable[..., torch.Tensor],
) -> float:
    """Runs a single epoch of training over a model.

    Args:
        model (nn.Module): The model to be trained
        device (torch.device): The device on which to run computations
        loader (DataLoader): The DataLoader containing the training data
        optimizer (torch.optim.Optimizer): the optimizer
        criterion (torch.nn.Module): The loss function

    Returns:
        float: the average training loss (normalized, per sample)
    """
    model.train()
    epoch_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        y, mu, log_std = model.forward(batch.x, batch)
        y_true, y_true_idx = to_dense_batch(batch.y, batch.batch)
        loss = criterion(mu, log_std, y, y_true)
        loss.backward()

        epoch_loss += loss.item()
        optimizer.step()

    return epoch_loss / len(loader)


T_Output = TypeVar("T_Output")
MetricFn = Callable[[torch.Tensor, torch.Tensor], Generic[T_Output]]


def evaluate_model(
    model: VAE, device: torch.device, loader: DataLoader, metric: MetricFn, average=True, infer=False
) -> T_Output:
    """Evaluate a model over the test data given a loss function.

    Args:
        model (nn.Module): The model to be trained
        device (torch.device): The device on which to run computations
        loader (DataLoader): The DataLoader containing the test data
        metric (MetricFn[T_Output]): The metric to evaluate the model against
        average (bool, optional): Averages over batches if True, sums otherwise. Defaults to False.

    Returns:
        T_Output: the result of the metric
    """
    model.train()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)

        with torch.no_grad():
            x = batch.x_infer if infer else batch.x
            out, _, _ = model.forward(x, batch)

            y_true, y_true_idx = to_dense_batch(batch.y, batch.batch)
            loss = metric(out, y_true)

        total_loss += loss

    if average:
        total_loss = total_loss / len(loader)

    return total_loss
