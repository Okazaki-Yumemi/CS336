from cs336_alignment.PromptAndOutput import tokenize_prompt_and_output
from cs336_alignment.ResponseLogProbs import get_response_log_probs
from cs336_alignment.Compute_rollout_reward import compute_rollout_rewards
from cs336_alignment.GroupNormalization import compute_group_normalized_rewards
from cs336_alignment.Policy_gradient import compute_policy_gradient_loss, aggregate_loss_across_microbatch


import torch
from typing import Literal,Callable
from transformers import PreTrainedTokenizerBase



def grpo_train_step(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    
    full_batch_size = len(rollout_responses)
    microbatch_size = full_batch_size// gradient_accumulation_steps
    
    device = next(model.parameters()).device
    
    raw_rewards,reward_metadata = compute_rollout_rewards(
        reward_fn,
        rollout_responses,
        repeated_ground_truths,
    )
    
    advantages,_ = compute_group_normalized_rewards(
        raw_rewards,
        group_size,
        baseline,
        advantage_eps,
        advantage_normalizer,
    )
    
    tokenized = tokenize_prompt_and_output(
        repeated_prompts,
        rollout_responses,
        tokenizer
    )
    
    total_loss = torch.zeros((), device=device)

    
    entropy_sum = torch.zeros((), device=device)
    entropy_count = torch.zeros((), device=device)
    
    input_ids = tokenized["input_ids"]
    labels = tokenized["labels"]
    response_mask = tokenized["response_mask"]
    
    for i in range(0, full_batch_size, microbatch_size):
        input_id_sliced = input_ids[i:i+microbatch_size]
        labels_sliced   = labels[i:i+microbatch_size]
        response_mask_sliced = response_mask[i:i+microbatch_size]
        advantages_sliced = advantages[i:i+microbatch_size]
        
        actual_microbatch_size = input_id_sliced.shape[0]
        
        input_id_sliced = input_id_sliced.to(device)
        labels_sliced = labels_sliced.to(device)
        response_mask_sliced = response_mask_sliced.to(device)
        advantages_sliced = advantages_sliced.to(device)
        
        log_dict = get_response_log_probs(
            model,
            input_id_sliced,
            labels_sliced,
            True
        )
        
        log_probs = log_dict["log_probs"]
        token_entropy = log_dict["token_entropy"]

        per_token_loss,_ = compute_policy_gradient_loss(
            advantages_sliced,
            log_probs,
            importance_reweighting_method,
        )
        
        microbatch_loss = aggregate_loss_across_microbatch(
            per_token_loss,
            response_mask_sliced,
            loss_normalization,
        )
        
        scaled_loss = microbatch_loss * actual_microbatch_size / full_batch_size
        
        scaled_loss.backward()
        
        total_loss += scaled_loss.detach()
        
        entropy_sum += (token_entropy.detach() * response_mask_sliced).sum()
        entropy_count += response_mask_sliced.sum()
    
    if max_grad_norm != None:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_grad_norm
        )
    
    optimizer.step()
    optimizer.zero_grad()
    
    entropy = entropy_sum/entropy_count
    
    metadata: dict[str, torch.Tensor | float] = {
        "total_loss": total_loss,
        "entropy": entropy,
    }
    
    return total_loss, metadata