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
    
    
    # === shaping ===
    if raw_rewards_or_advantages.ndim == 1:
        # (B,) -> (B,1)
        advantages = raw_rewards_or_advantages.unsqueeze(-1)
    else:
        advantages = raw_rewards_or_advantages
    
    
    
    if importance_reweighting_method == "none":
        per_token_loss = - advantages * policy_log_probs
    
    elif importance_reweighting_method == "noclip":
        
        if old_log_probs == None:
            raise ValueError
        
        ratio = torch.exp(policy_log_probs - old_log_probs)

        objective = advantages * ratio

        per_token_loss = - objective
    
    elif importance_reweighting_method == "grpo":
    
        if old_log_probs == None:
            raise ValueError
                
        ratio = torch.exp(policy_log_probs - old_log_probs)

        unclipped_objective = advantages * ratio
        
        if cliprange == None:
            raise ValueError
        
        clipped_ratio = torch.clamp(ratio, 1-cliprange, 1+cliprange)

        clipped_objective = advantages * clipped_ratio
        
        objective = torch.minimum(
            unclipped_objective,
            clipped_objective,
        )
    
        per_token_loss = - objective
        
    elif importance_reweighting_method == "gspo":
        
        if old_log_probs == None:
            raise ValueError
        if cliprange == None:
            raise ValueError
        if response_mask == None:
            raise ValueError
        
        log_ratio = policy_log_probs - old_log_probs
        
        masked_log_ratio = log_ratio * response_mask
        
        response_length = response_mask.sum(dim = 1, keepdim= True)
        
        mean_log_ratio = masked_log_ratio.sum(dim=1, keepdim=True) / response_length
        
        sequence_ratio = torch.exp(mean_log_ratio)
        
        unclipped_objective = advantages * sequence_ratio
        
        clipped_ratio = torch.clamp(sequence_ratio,1-cliprange,1+cliprange)
        
        clipped_objective = advantages * clipped_ratio
        
        objective = torch.minimum(
            unclipped_objective,
            clipped_objective,
        )
        
        per_token_loss = - objective.expand_as(policy_log_probs)
        
    
    metadata = {}
    
    return per_token_loss,metadata


def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence","constant"] = "sequence",
    normalization_constant: int | None = None,
)-> torch.Tensor:
    
    masked_loss = per_token_policy_gradient_loss * mask
    loss_sum = masked_loss.sum(dim= 1)
    
    if loss_normalization == "constant":
        
        if normalization_constant is None:
            raise ValueError
        
        final_loss = loss_sum.div(normalization_constant).sum(dim=0)
    
    if loss_normalization == "sequence":
  
        token_count = mask.sum(dim=1)
        
        sequence_loss = loss_sum / token_count
        
        final_loss = sequence_loss.mean(dim=0)
    
    return final_loss