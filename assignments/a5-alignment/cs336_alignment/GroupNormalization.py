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
    
    if baseline == "none":
        centered_rewards = grouped_rewards
    
    if baseline == "mean":
        centered_rewards = grouped_rewards - grouped_rewards.mean(dim= 1, keepdim= True)

    # Normalize phase
    # ================================================================
    if advantage_normalizer == "none":
        
        advantages_2d = (
            centered_rewards
        ) 
        
        
    if advantage_normalizer == "std":
        group_std = grouped_rewards.std(dim = 1, keepdim=True)

        advantages_2d = (
            centered_rewards
        ) / (
            group_std + advantage_eps
        )
        
    if advantage_normalizer == "mean":
        group_mean = grouped_rewards.mean(dim=1, keepdim=True)
        
        advantages_2d = (
            centered_rewards
        ) / (
            group_mean + advantage_eps
        )
        
    # ================================================================

    advantages = advantages_2d.reshape(-1)
    
    metadata = {
        "mean":float(raw_rewards.mean())
    }
    
    return  advantages,metadata