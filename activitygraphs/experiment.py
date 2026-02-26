import torch
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.logging import log


def compute_training_weights(loader: pyg.loader.DataLoader):
    num_neg = 0
    num_pos = 0

    for batch in loader:
        num_neg += (batch.y == 0).sum()
        num_pos += batch.y.sum()
    return num_neg / num_pos


def extract_features(batch: pyg.data.Data, full_info: bool):
    if not full_info:
        return batch.x

    if batch.batch is not None:
        home_feature = batch.home_feature[batch.batch].unsqueeze(1)
    else:
        home_feature = torch.full((batch.x.shape[0], 1), batch.home_feature.item())

    return torch.cat([batch.x, home_feature], dim=1)


def train(
    device: torch.device,
    model: torch.nn.Module,
    loader: pyg.loader.DataLoader,
    optimizer: torch.optim.Optimizer,
    full_info: bool,
):
    model.train()

    epoch_loss = 0.0
    num_nodes = 0

    for batch in loader:
        batch = batch.to(device)
        x = extract_features(batch, full_info)

        optimizer.zero_grad()
        out = model(x, batch.edge_index)
        loss = F.binary_cross_entropy_with_logits(out, batch.y.float())
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * batch.num_nodes
        num_nodes += batch.num_nodes

    return epoch_loss / num_nodes


@torch.no_grad()
def test(device: torch.device, model: torch.nn.Module, loader: pyg.loader.DataLoader, full_info: bool):
    model.eval()

    epoch_loss = 0.0
    num_nodes = 0

    for batch in loader:
        batch = batch.to(device)
        x = extract_features(batch, full_info)

        out = model(x, batch.edge_index)
        loss = F.binary_cross_entropy_with_logits(out, batch.y.float())

        epoch_loss += loss.item() * batch.num_nodes
        num_nodes += batch.num_nodes

    return epoch_loss / num_nodes


def run_experiment(
    model: torch.nn.Module,
    train_loader: pyg.loader.DataLoader,
    test_loader: pyg.loader.DataLoader,
    num_epochs: int = 10,
    verbose: int = 1,
    name: str | None = None,
    lr: float = 0.01,
    full_info: bool = False,
):
    name = name or model.__class__.__name__

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epoch = train_loss = train_eval_loss = test_loss = None
    train_losses = []
    test_losses = []

    log(Model=name)

    for epoch in range(1, num_epochs + 1):
        train_loss = train(device, model, train_loader, optimizer, full_info)
        train_eval_loss = test(device, model, train_loader, full_info)
        test_loss = test(device, model, test_loader, full_info)

        train_losses.append(train_eval_loss)
        test_losses.append(test_loss)

        if verbose and (epoch - 1) % verbose == 0:
            log(Epoch=epoch, train_loss=train_loss, train_eval_loss=train_eval_loss, test_loss=test_loss)

    log(Epoch=epoch, train_loss=train_loss, train_eval_loss=train_eval_loss, test_loss=test_loss)

    return {"name": name, "epoch": range(1, num_epochs + 1), "train": train_losses, "test": test_losses}
