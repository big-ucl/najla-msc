from dataclasses import dataclass

import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import train_test_split
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from utils import check_schema

from models import Benchmark


@dataclass(frozen=True)
class Experiment:
    """Represents an experiement on which to train & test ML models."""

    train_set: Dataset
    test_set: Dataset
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
        return "train" in self.losses["type"] and "val" in self.losses["type"]

    def _losses(self, loss_type: str):
        losses = self.losses.filter(pl.col("type") == loss_type)

        if len(losses) == 0:
            raise ValueError(f"No loss of type {loss_type}")

        return losses

    def _latest_loss(self, loss_type: str) -> float:
        return self._losses(loss_type).filter(pl.col("epoch") == pl.col("epoch").max())["loss"][0]

    def train_losses(self):
        return self._losses("train")["loss"]

    def val_losses(self):
        return self._losses("val")["loss"]

    def test_losses(self):
        return self._losses("test")["loss"]

    def test_loss(self):
        return self._latest_loss("test")

    def final_losses(self) -> tuple[float, float, float]:
        return self._latest_loss("train"), self._latest_loss("val"), self.test_loss()

    def __repr__(self):
        return f"Results({self.name} | Test loss={self.test_loss():.4f})"


def create_loss(n_classes: int, with_logits=False, epsilon=0.0001) -> nn.Module:
    def loss(out: torch.Tensor, y: torch.Tensor):
        out = out.reshape((-1, n_classes))
        if with_logits:
            return F.cross_entropy(out, y)
        else:
            return F.nll_loss(torch.log(out + epsilon), y)

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
    criterion: torch.nn.Module,
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


def evaluate_model(model: nn.Module, device: torch.device, loader: DataLoader, criterion: torch.nn.Module) -> float:
    """Evaluate a model over the test data given a loss function.

    Args:
        model (nn.Module): The model to be trained
        device (torch.device): The device on which to run computations
        loader (DataLoader): The DataLoader containing the test data
        criterion (torch.nn.Module): The loss function

    Returns:
        float: the average test loss (normalized, per sample)
    """
    model.eval()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)

        with torch.no_grad():
            out = model(batch)
            loss = criterion(out, batch.y)

        total_loss += loss

    return total_loss / len(loader)
