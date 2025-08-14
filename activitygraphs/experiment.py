from dataclasses import dataclass

import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import train_test_split
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader


@dataclass(frozen=True)
class Experiment:
    train_set: Dataset
    test_set: Dataset
    n_epochs: int
    val_size: int
    batch_size: int
    random_state: int


class Results:
    def __init__(self, name: str, losses: pl.DataFrame):
        self.name = name
        self.losses = losses
        self.n_epochs = losses["epoch"].max()

    def _losses(self, loss_type: str):
        return self.losses.filter(pl.col("type") == loss_type)

    def _latest_loss(self, loss_type: str) -> float:
        return self._losses(loss_type).filter(pl.col("epoch") == pl.col("epoch").max())["loss"][0]

    def train_losses(self):
        return self._losses("train")["loss"]

    def val_losses(self):
        return self._losses("val")["loss"]

    def test_losses(self):
        return self._losses("test")["loss"]

    def final_losses(self) -> tuple[float, float, float]:
        return self._latest_loss("train"), self._latest_loss("val"), self._latest_loss("test")

    def __repr__(self):
        train, val, test = self.final_losses()
        return f"Results({self.name} | {train=:.4f} , {val=:.4f}, {test=:.4f})"


def run_experiment(
    experiment: Experiment,
    model: nn.Module,
    lr=0.01,
    name: str = None,
    verbose=True,
) -> pl.DataFrame:
    exp = experiment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = F.binary_cross_entropy_with_logits

    train_set, val_set = train_test_split(exp.train_set, test_size=exp.val_size, random_state=exp.random_state)

    train_loader = DataLoader(dataset=train_set, batch_size=exp.batch_size)
    val_loader = DataLoader(dataset=val_set, batch_size=exp.batch_size)

    train_losses = []
    val_losses = []

    if verbose:
        print(f"======= {name} (n_epochs={exp.n_epochs}) =======")

    # Training phase
    for epoch in range(exp.n_epochs):
        train_loss = train_epoch(model, device, train_loader, optimizer, criterion)
        val_loss = evaluate_model(model, device, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % 10 == 0 and verbose:
            print(f"Epoch {epoch + 1:3}, Training loss: {train_loss:.4f} | Validation loss: {val_loss:.4f}")

    # Evaluate on the test set
    test_loader = DataLoader(dataset=exp.test_set, batch_size=exp.batch_size)
    test_loss = evaluate_model(model, device, test_loader, criterion)

    epochs = list(range(1, exp.n_epochs + 1))

    losses = pl.concat([
        pl.DataFrame({"epoch": epochs, "loss": train_losses}).with_columns(pl.lit("train").alias("type")),
        pl.DataFrame({"epoch": epochs, "loss": val_losses}).with_columns(pl.lit("val").alias("type")),
        pl.DataFrame({"epoch": exp.n_epochs, "loss": test_loss}).with_columns(pl.lit("test").alias("type")),
    ]).with_columns(pl.lit(name).alias("name"))

    return Results(name, losses)


def train_epoch(
    model: nn.Module,
    device: torch.device,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
):
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


def evaluate_model(model: nn.Module, device: torch.device, loader: DataLoader, criterion: torch.nn.Module):
    model.eval()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)

        with torch.no_grad():
            out = model(batch)
            loss = criterion(out, batch.y)

        total_loss += loss

    return total_loss / len(loader)
