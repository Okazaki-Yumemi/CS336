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