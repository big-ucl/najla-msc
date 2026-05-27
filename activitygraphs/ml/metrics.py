"""Ranking metrics: precision@k, recall@k, MRR, NDCG@k."""

import torch


def precision_at_k(scores: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    """Return the fraction of the top-k scoring nodes that are true positives."""
    top_k_indices = scores.topk(k).indices
    return labels[top_k_indices].sum().item() / k


def recall_at_k(scores: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    """Return the fraction of all positives that fall within the top-k scoring nodes."""
    top_k_indices = scores.topk(k).indices
    num_pos = labels.sum().int().item()
    if num_pos == 0:
        return 0.0
    return labels[top_k_indices].sum().item() / num_pos


def mean_reciprocal_rank(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Return the mean reciprocal rank of positive labels given ``scores``."""
    ranked_indices = scores.argsort(descending=True)
    ranked_labels = labels[ranked_indices]
    positive_ranks = (ranked_labels == 1).nonzero().squeeze(1) + 1  # 1-indexed
    if len(positive_ranks) == 0:
        return 0.0
    return (1.0 / positive_ranks.float()).mean().item()


def ndcg_at_k(scores: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    """Return normalised discounted cumulative gain at rank k."""
    ranked_indices = scores.argsort(descending=True)[:k]
    ranked_labels = labels[ranked_indices].float()

    # DCG
    device = scores.device
    positions = torch.arange(1, k + 1, dtype=torch.float, device=device)
    discounts = 1.0 / torch.log2(positions + 1)
    dcg = (ranked_labels * discounts).sum().item()

    # Ideal DCG — best possible ranking
    num_pos = int(labels.sum().item())
    ideal_labels = torch.zeros(k, device=device)
    ideal_labels[: min(num_pos, k)] = 1.0
    idcg = (ideal_labels * discounts).sum().item()

    if idcg == 0:
        return 0.0
    return dcg / idcg
