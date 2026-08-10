import torch
from typing import Any, Callable, Literal


def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[torch.Tensor, dict[str, float]]:
    
    grouped_rewards = raw_rewards.reshape(-1, group_size)
    
    if baseline != "mean":
        raise NotImplementedError
    
    
    group_mean = grouped_rewards.mean(dim= 1, keepdim= True)


    if advantage_normalizer != "std":
        raise NotImplementedError
    
    group_std = grouped_rewards.std(dim = 1, keepdim=True)

    advantages_2d = (
        grouped_rewards - group_mean
    ) / (
        group_std + advantage_eps
    )

    advantages = advantages_2d.reshape(-1)
    
    metadata = {
        "mean":float(raw_rewards.mean())
    }
    
    return  advantages,metadata