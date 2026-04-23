from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, recall_score, roc_auc_score

MAX_LOG_VAR = 10


class WeightedDataset(Protocol):
    def class_weights(self) -> torch.Tensor:
        pass


def make_weighted_recon_loss(train_set: WeightedDataset):
    weights = train_set.class_weights()

    def weighted_recon_loss(y_pred: torch.Tensor, y_true: torch.Tensor):
        return F.binary_cross_entropy_with_logits(y_pred, y_true, pos_weight=weights)

    return weighted_recon_loss


def recon_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(y_pred, y_true)


def kl_loss(mu: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    log_std = torch.clamp(log_std, max=MAX_LOG_VAR)
    distances = torch.sum(1 + 2 * log_std - mu**2 - log_std.exp() ** 2, dim=1)
    return -0.5 * torch.mean(distances)


def make_elbo_loss(recon_loss_fn, kl_weight=0.001):
    def elbo_loss(mu: torch.Tensor, log_std: torch.Tensor, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return recon_loss_fn(y_pred, y_true) + kl_weight * kl_loss(mu, log_std)

    return elbo_loss


def _prepare(y: torch.Tensor, sig=False, threshold_proba: float | None = None) -> np.ndarray:
    y = torch.flatten(y, end_dim=1)
    y = torch.sigmoid(y) if sig else y

    if threshold_proba is not None:
        y = (y > threshold_proba).float()

    return y.detach().squeeze().cpu().numpy()


def roc_auc(logits: torch.Tensor, y_true: torch.Tensor) -> float:
    return roc_auc_score(_prepare(y_true), _prepare(logits, sig=True))


def average_precision(logits: torch.Tensor, y_true: torch.Tensor) -> float:
    return average_precision_score(_prepare(y_true), _prepare(logits, sig=True))


def accuracy(logits: torch.Tensor, y_true: torch.Tensor, threshold_proba: float = 0.5) -> float:
    return accuracy_score(_prepare(y_true), _prepare(logits, sig=True, threshold_proba=threshold_proba))


def precision(logits: torch.Tensor, y_true: torch.Tensor, threshold_proba: float = 0.5) -> float:
    return precision_score(
        _prepare(y_true), _prepare(logits, sig=True, threshold_proba=threshold_proba), zero_division=0
    )


def recall(logits: torch.Tensor, y_true: torch.Tensor, threshold_proba: float = 0.5) -> float:
    return recall_score(_prepare(y_true), _prepare(logits, sig=True, threshold_proba=threshold_proba))
