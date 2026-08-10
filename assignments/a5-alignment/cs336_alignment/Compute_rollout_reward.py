import torch
from typing import Any, Callable, Literal

def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    
    raw_rewards = []
    total_reward_sum :float = 0
    format_reward_sum: float = 0
    answer_reward_sum: float = 0
    
    for response,ground_truth in zip(rollout_responses,repeated_ground_truths):
        
        result = reward_fn(response,ground_truth)
        
        raw_rewards.append(result["reward"])
        
        total_reward_sum += result["reward"]
        format_reward_sum += result["format_reward"]
        answer_reward_sum += result["answer_reward"]
        
    raw_rewards = torch.tensor(raw_rewards,dtype = torch.float32)
    
    n = len(rollout_responses)
    
    metadata = {
        "mean_reward": total_reward_sum/n,
        "mean_format_reward": format_reward_sum/n,
        "mean_answer_reward": answer_reward_sum/n,
    }
    return raw_rewards,metadata
    