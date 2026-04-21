import torch
import torch_geometric as pyg


def poisson_sampling(logits: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    return torch.bernoulli(probs, generator=generator).detach()


def pps_sampling(num_samples: int, logits: torch.Tensor, batch: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    logits, mask = pyg.utils.to_dense_batch(logits, batch, fill_value=0.0)
    probs = torch.sigmoid(logits).squeeze(-1) * mask

    weights = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-8)
    indices = torch.multinomial(weights, num_samples, replacement=False, generator=generator)

    samples = torch.zeros_like(probs)
    samples.scatter_(dim=1, index=indices, value=1.0)

    return samples[mask].unsqueeze(-1)
