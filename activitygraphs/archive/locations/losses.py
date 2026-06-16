from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, recall_score, roc_auc_score

# Upper bound on log variance to prevent numerical overflow (exp(10) is already very large)
MAX_LOG_VAR = 10


class WeightedDataset(Protocol):
    """
    Description: A structural Protocol (interface) that specifies any dataset object exposing
    a `class_weights()` method. Used as a type hint for loss functions that need to apply
    per-class positive weights to handle class imbalance (e.g. rare visited locations).
    """
    def class_weights(self) -> torch.Tensor:
        """
        Description: Returns a tensor of per-class positive weights for use in weighted
        binary cross-entropy. Higher weights upweight rare (positive) classes.

        Output:
          - (torch.Tensor): A 1D tensor of shape (num_classes,) with per-class weights.
        """
        pass


def make_weighted_recon_loss(train_set: WeightedDataset):
    """
    Description: Creates a weighted binary cross-entropy reconstruction loss function that
    addresses class imbalance by giving higher weight to positive (visited) examples.
    The weights are computed from the training set class distribution and frozen into the
    returned closure — they do not change during training.

    Input:
      - train_set (WeightedDataset): The training dataset, used to compute per-class positive
        weights via `class_weights()`.

    Output:
      - (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): A loss function that takes
        (y_pred_logits, y_true_binary) and returns a scalar weighted BCE loss.
    """
    weights = train_set.class_weights()  # Compute per-class weights from the training set distribution

    def weighted_recon_loss(y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        Description: Computes weighted binary cross-entropy loss between predicted logits and
        binary ground-truth labels. Positive examples are upweighted to offset class imbalance.

        Input:
          - y_pred (torch.Tensor): Raw logit predictions from the model (before sigmoid).
          - y_true (torch.Tensor): Ground-truth binary labels (0 or 1).

        Output:
          - (torch.Tensor): Scalar weighted binary cross-entropy loss.
        """
        return F.binary_cross_entropy_with_logits(y_pred, y_true, pos_weight=weights)

    return weighted_recon_loss


def recon_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Description: Computes unweighted binary cross-entropy reconstruction loss between predicted
    logits and binary ground-truth labels. Does NOT apply class imbalance correction — use
    `make_weighted_recon_loss` if the dataset is heavily imbalanced.

    Input:
      - y_pred (torch.Tensor): Raw logit predictions from the model (before sigmoid).
      - y_true (torch.Tensor): Ground-truth binary labels (0 or 1), same shape as y_pred.

    Output:
      - (torch.Tensor): Scalar binary cross-entropy loss.
    """
    return F.binary_cross_entropy_with_logits(y_pred, y_true)


def kl_loss(mu: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    """
    Description: Computes the KL divergence between the approximate posterior q(z|x) = N(mu, std^2)
    and the standard normal prior p(z) = N(0, 1). This is the regularisation term in the VAE ELBO
    (Evidence Lower BOund) loss. Minimising this term encourages the latent space to stay close to
    a standard normal, preventing the encoder from encoding arbitrary information.

    The closed-form KL between two Gaussians is:
        KL = -0.5 * sum(1 + 2*log_std - mu^2 - exp(2*log_std))

    Input:
      - mu (torch.Tensor): Mean of the approximate posterior, shape=(batch, latent_dim).
      - log_std (torch.Tensor): Log standard deviation of the posterior, shape=(batch, latent_dim).

    Output:
      - (torch.Tensor): Scalar KL divergence loss, averaged over the batch.
    """
    # Clamp log_std to avoid numerical overflow when computing exp(log_std)^2
    log_std = torch.clamp(log_std, max=MAX_LOG_VAR)
    # Closed-form KL divergence summed over the latent dimensions
    distances = torch.sum(1 + 2 * log_std - mu**2 - log_std.exp() ** 2, dim=1)
    return -0.5 * torch.mean(distances)  # Average KL over the batch


def make_elbo_loss(recon_loss_fn, kl_weight=0.001):
    """
    Description: Creates an ELBO (Evidence Lower BOund) loss function for training a VAE.
    The ELBO loss combines:
      - A reconstruction loss: how well the VAE reproduces the original input (node labels).
      - A KL divergence term: regularisation that keeps the latent space close to N(0,1).
    The kl_weight controls the balance between reconstruction quality and latent space regularity.
    A small kl_weight (e.g. 0.001) focuses training on reconstruction quality.

    Input:
      - recon_loss_fn (Callable): A reconstruction loss function, e.g. `recon_loss` or the result
        of `make_weighted_recon_loss`. Signature: (y_pred, y_true) -> scalar.
      - kl_weight (float): Weight applied to the KL divergence term. Defaults to 0.001.

    Output:
      - (Callable): An ELBO loss function with signature:
          elbo_loss(mu, log_std, y_pred, y_true) -> scalar torch.Tensor.
    """
    def elbo_loss(mu: torch.Tensor, log_std: torch.Tensor, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Description: Computes the combined ELBO loss = reconstruction loss + weighted KL divergence.

        Input:
          - mu (torch.Tensor): Latent mean from the encoder, shape=(batch, latent_dim).
          - log_std (torch.Tensor): Latent log std from the encoder, shape=(batch, latent_dim).
          - y_pred (torch.Tensor): Reconstructed output (model predictions / logits).
          - y_true (torch.Tensor): Ground-truth binary labels.

        Output:
          - (torch.Tensor): Scalar ELBO loss (reconstruction + kl_weight * KL divergence).
        """
        return recon_loss_fn(y_pred, y_true) + kl_weight * kl_loss(mu, log_std)

    return elbo_loss


def _prepare(y: torch.Tensor, sig=False, threshold_proba: float | None = None) -> np.ndarray:
    """
    Description: Internal utility that converts a model output or ground-truth tensor into a flat
    numpy array suitable for scikit-learn metrics. Optionally applies sigmoid activation (to
    convert logits to probabilities) and/or applies a threshold to produce binary predictions.

    Input:
      - y (torch.Tensor): Input tensor, typically model logits or ground-truth labels.
        May have a batch dimension that will be flattened.
      - sig (bool): If True, applies sigmoid activation to convert logits to probabilities [0,1].
        Set to True for model predictions. Defaults to False.
      - threshold_proba (float | None): If provided, converts probabilities to binary predictions
        (1 if probability > threshold, else 0). If None, probabilities are returned as-is.
        Defaults to None.

    Output:
      - (np.ndarray): A flat numpy array of predictions or labels, ready for sklearn metrics.
    """
    y = torch.flatten(y, end_dim=1)  # Flatten (batch, nodes) into a 1D vector
    y = torch.sigmoid(y) if sig else y  # Convert logits to probabilities if requested

    if threshold_proba is not None:
        y = (y > threshold_proba).float()  # Convert to binary 0/1 using the given probability threshold

    return y.detach().squeeze().cpu().numpy()  # Detach from computation graph and convert to numpy


def roc_auc(logits: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Description: Computes the ROC-AUC (Receiver Operating Characteristic - Area Under Curve) score
    between model logit predictions and ground-truth binary labels. ROC-AUC measures the model's
    ability to rank positive examples (visited locations) above negative ones. A value of 0.5 means
    random chance; 1.0 means perfect discrimination.

    Input:
      - logits (torch.Tensor): Raw model output logits (before sigmoid), any shape.
      - y_true (torch.Tensor): Ground-truth binary labels (0 or 1), same shape as logits.

    Output:
      - (float): ROC-AUC score between 0.0 and 1.0.
    """
    return roc_auc_score(_prepare(y_true), _prepare(logits, sig=True))


def average_precision(logits: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Description: Computes the Average Precision (AP) score, which summarises the precision-recall
    curve as a weighted average of precisions at each threshold. AP is particularly useful for
    imbalanced datasets (e.g. when only a few locations are visited out of many possibilities).
    A value of 1.0 means perfect precision at all recall levels.

    Input:
      - logits (torch.Tensor): Raw model output logits (before sigmoid), any shape.
      - y_true (torch.Tensor): Ground-truth binary labels (0 or 1), same shape as logits.

    Output:
      - (float): Average precision score between 0.0 and 1.0.
    """
    return average_precision_score(_prepare(y_true), _prepare(logits, sig=True))


def accuracy(logits: torch.Tensor, y_true: torch.Tensor, threshold_proba: float = 0.5) -> float:
    """
    Description: Computes the classification accuracy between model predictions and ground-truth
    binary labels. Probabilities are thresholded at `threshold_proba` to produce binary predictions.
    Accuracy = (correct predictions) / (total predictions).

    Input:
      - logits (torch.Tensor): Raw model output logits (before sigmoid), any shape.
      - y_true (torch.Tensor): Ground-truth binary labels (0 or 1), same shape as logits.
      - threshold_proba (float): Probability threshold above which a node is predicted as visited (1).
        Defaults to 0.5.

    Output:
      - (float): Accuracy score between 0.0 and 1.0.
    """
    return accuracy_score(_prepare(y_true), _prepare(logits, sig=True, threshold_proba=threshold_proba))


def precision(logits: torch.Tensor, y_true: torch.Tensor, threshold_proba: float = 0.5) -> float:
    """
    Description: Computes the precision score: among all nodes predicted as visited, what fraction
    were truly visited? Precision = TP / (TP + FP). Using zero_division=0 returns 0 if there
    are no positive predictions (avoids division by zero).

    Input:
      - logits (torch.Tensor): Raw model output logits (before sigmoid), any shape.
      - y_true (torch.Tensor): Ground-truth binary labels (0 or 1), same shape as logits.
      - threshold_proba (float): Probability threshold above which a node is predicted as visited (1).
        Defaults to 0.5.

    Output:
      - (float): Precision score between 0.0 and 1.0.
    """
    return precision_score(
        _prepare(y_true), _prepare(logits, sig=True, threshold_proba=threshold_proba), zero_division=0
    )


def recall(logits: torch.Tensor, y_true: torch.Tensor, threshold_proba: float = 0.5) -> float:
    """
    Description: Computes the recall (sensitivity) score: among all truly visited nodes, what
    fraction did the model correctly predict as visited? Recall = TP / (TP + FN).

    Input:
      - logits (torch.Tensor): Raw model output logits (before sigmoid), any shape.
      - y_true (torch.Tensor): Ground-truth binary labels (0 or 1), same shape as logits.
      - threshold_proba (float): Probability threshold above which a node is predicted as visited (1).
        Defaults to 0.5.

    Output:
      - (float): Recall score between 0.0 and 1.0.
    """
    return recall_score(_prepare(y_true), _prepare(logits, sig=True, threshold_proba=threshold_proba))
