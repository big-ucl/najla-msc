"""
Stochastic location-sampling strategies (Poisson, PPS) over model logits.

These functions turn continuous model output (logits) into discrete binary
location sets — simulating the actual location-choice decision of a traveller.

Two strategies are implemented:

``poisson_sampling``
  Each node is independently included with probability ``sigmoid(logit_n)``.
  The expected number of selected nodes equals the sum of all probabilities.
  Because each node is drawn independently (Bernoulli), the total selected
  set size follows a Poisson-like distribution, hence the name.

``pps_sampling``
  Probability-proportional-to-size (PPS) sampling without replacement:
  selects *exactly* ``num_samples`` nodes per graph, where each node's
  selection probability is proportional to ``sigmoid(logit_n)``.

Both functions return binary tensors (1 = selected, 0 = not selected) and
detach gradients so they can be used for evaluation without affecting
backpropagation.
"""

import torch
import torch_geometric as pyg


def unweight_probs(probs: torch.Tensor, pos_weight: float) -> torch.Tensor:
    """
    Description: Undo the distortion introduced by positive-class weighting
    in the BCE loss to recover calibrated probabilities.

    When a model is trained with ``pos_weight > 1`` in
    ``F.binary_cross_entropy_with_logits``, the learned sigmoid output is
    biased upward (the model is trained as if positives were ``pos_weight``
    times more common than they actually are).  This function reverses that
    bias so that the output is interpretable as a well-calibrated probability.

    The formula is derived from the relationship between the weighted and
    unweighted probabilities:
      ``p_calibrated = p_weighted / (pos_weight - (pos_weight - 1) * p_weighted)``

    Input:
      - probs (torch.Tensor): Sigmoid probabilities from the model, possibly
        biased by positive-class weighting.  Values in (0, 1).
      - pos_weight (float): The positive-class weight used during training.
        Pass ``1.0`` for unweighted training (no-op).

    Output:
      - (torch.Tensor): Calibrated probabilities with the same shape as
        ``probs``.
    """
    w = pos_weight  # shorthand for the weight value
    # Inversion formula: recovers the pre-weighting probability
    return probs / (w - (w - 1) * probs)


def poisson_sampling(
    logits: torch.Tensor, generator: torch.Generator | None, pos_weight: float | None = None
) -> torch.Tensor:
    """
    Description: Draw an independent Bernoulli sample for each node using
    the model's predicted probabilities.

    Each node ``n`` is independently assigned to the sampled set with
    probability ``p_n = sigmoid(logit_n)`` (with optional calibration
    correction if the model was trained with a positive-class weight).

    The resulting binary vector has an *expected* sum equal to
    ``sum(sigmoid(logits))``, which approximates the model's prediction of
    the user's set size.  Because each node is drawn independently, the
    selected set size is random (not fixed).

    Gradients are detached because this function is used for evaluation only.

    Input:
      - logits (torch.Tensor): Raw model output logits, shape [num_nodes].
        Sigmoid is applied internally.
      - generator (torch.Generator | None): Optional random-number generator
        for reproducibility.  Pass ``None`` for default randomness.
      - pos_weight (float | None): If the model was trained with a
        positive-class BCE weight, pass that weight here to calibrate
        probabilities before sampling.  Pass ``None`` (default) for
        unweighted training.

    Output:
      - (torch.Tensor): Binary sample tensor, shape [num_nodes].
        1 = node selected, 0 = node not selected.  Gradients detached.
    """
    # Convert raw logits to probabilities in (0, 1)
    probs = torch.sigmoid(logits)

    # Optionally reverse the pos_weight distortion to get calibrated probabilities
    if pos_weight is not None:
        probs = unweight_probs(probs, pos_weight)

    # Independent Bernoulli draws (one per node); detach to stop gradient flow
    return torch.bernoulli(probs, generator=generator).detach()


def pps_sampling(
    num_samples: int, logits: torch.Tensor, batch: torch.Tensor, generator: torch.Generator | None = None
) -> torch.Tensor:
    """
    Description: Sample exactly ``num_samples`` nodes per graph using
    probability-proportional-to-size (PPS) sampling without replacement.

    Unlike Poisson sampling (where the total selected count is random),
    this function selects a *fixed* number of nodes per graph.  Each node's
    inclusion probability is proportional to its predicted probability:
      ``selection_prob[n] ∝ sigmoid(logit[n])``

    This is useful when you want the model to predict a fixed-size itinerary
    rather than a variable-size set.

    Steps:
      1. Convert logits to a dense [num_graphs, max_nodes] tensor (pad shorter
         graphs with zeros and track a mask).
      2. Compute normalised selection weights per graph.
      3. Draw ``num_samples`` indices per graph via ``torch.multinomial`` (no
         replacement).
      4. Convert the index selections back to a binary mask.
      5. Return the flat [num_nodes, 1] binary mask (excl. padding).

    Input:
      - num_samples (int): Fixed number of nodes to select per graph.
      - logits (torch.Tensor): Raw model output logits, shape [num_nodes, 1]
        or [num_nodes].
      - batch (torch.Tensor): Batch assignment vector, shape [num_nodes].
        Entry ``r`` = which graph node ``r`` belongs to.
      - generator (torch.Generator | None): Optional RNG for reproducibility.

    Output:
      - (torch.Tensor): Binary selection tensor, shape [num_nodes, 1].
        1 = selected, 0 = not selected.  No gradients.
    """
    # Convert the flat (possibly padded) logits to a dense batch tensor
    # logits shape after to_dense_batch: [num_graphs, max_nodes]
    # mask: True for real nodes, False for padding
    logits, mask = pyg.utils.to_dense_batch(logits, batch, fill_value=0.0)

    # Convert logits to probabilities; squeeze if shape is [num_graphs, max_nodes, 1]
    # Zero out padding positions so they cannot be selected
    probs = torch.sigmoid(logits).squeeze(-1) * mask

    # Normalise probabilities per graph so they sum to 1 (required by multinomial)
    weights = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-8)

    # Draw exactly num_samples indices per graph (without replacement)
    indices = torch.multinomial(weights, num_samples, replacement=False, generator=generator)

    # Convert selected indices back to a binary [num_graphs, max_nodes] mask
    samples = torch.zeros_like(probs)
    samples.scatter_(dim=1, index=indices, value=1.0)  # set selected positions to 1

    # Remove padding by indexing with the original mask; add trailing dim for compatibility
    return samples[mask].unsqueeze(-1)
