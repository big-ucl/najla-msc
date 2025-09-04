import torch
import torch.nn.functional as F

MAX_LOG_VAR = 10


def elbo_loss(mu: torch.Tensor, log_std: torch.Tensor, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return recon_loss(y_pred, y_true) + kl_loss(mu, log_std)


def recon_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(y_pred, y_true)


def kl_loss(mu: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    log_std = torch.clamp(log_std, max=MAX_LOG_VAR)
    distances = torch.sum(1 + 2 * log_std - mu ** 2 - log_std.exp() ** 2, dim=1)
    return -0.5 * torch.mean(distances)
