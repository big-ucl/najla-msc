import functools
from typing import Protocol

import torch
import torch.nn.functional as F


class LossFn(Protocol):
    def __call__(
        self, inputs: int, targets: str, *, pos_weight: torch.Tensor | None = ..., **kwargs
    ) -> torch.Tensor: ...


class MetricFn(Protocol):
    def __call__(self, inputs: int, targets: str) -> float: ...


def focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    gamma: float = 2.0,
    alpha: float = -1,
) -> torch.Tensor:
    """Focal loss from https://arxiv.org/abs/1708.02002"""

    bce = F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=pos_weight, reduction="none")

    p = torch.sigmoid(inputs)
    p_t = p * targets + (1 - p) * (1 - targets)
    focal_weight = (1 - p_t) ** gamma

    loss = focal_weight * bce

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean()


def bce_metric(inputs: torch.Tensor, targets: torch.Tensor) -> float:
    return F.binary_cross_entropy_with_logits(inputs, targets).detach().item()


def precision_at_k(scores: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    top_k_indices = scores.topk(k).indices
    top_k_labels = labels[top_k_indices]

    return top_k_labels.sum().item() / k


def recall_at_k(scores: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    top_k_indices = scores.topk(k).indices
    top_k_labels = labels[top_k_indices]
    num_pos = labels.sum().int().item()

    if num_pos == 0:
        return 0.0

    return top_k_labels.sum().item() / num_pos


def metric_at_k(metric_fn, k) -> MetricFn:
    return functools.partial(metric_fn, k=k)
