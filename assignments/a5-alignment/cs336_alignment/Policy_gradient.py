from typing import Literal

import torch

def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    
    if importance_reweighting_method != "none":
        raise NotImplementedError
    
    if raw_rewards_or_advantages.ndim == 1:
        # (B,) -> (B,1)
        raw_rewards_or_advantages = raw_rewards_or_advantages.unsqueeze(-1)
    
    per_token_loss = - raw_rewards_or_advantages * policy_log_probs
    
    metadata = {}
    
    return per_token_loss,metadata


def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence","constant"] = "sequence",
    normalization_constant: int | None = None,
)-> torch.Tensor:
    
    if loss_normalization != "sequence":
        raise NotImplementedError
    
    masked_loss = per_token_policy_gradient_loss * mask
    
    loss_sum = masked_loss.sum(dim= 1)
    
    token_count = mask.sum(dim=1)
    
    sequence_loss = loss_sum / token_count
    
    final_loss = sequence_loss.mean(dim=0)
    
    return final_loss