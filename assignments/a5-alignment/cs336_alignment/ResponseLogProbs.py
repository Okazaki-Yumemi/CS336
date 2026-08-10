import torch
import torch.nn.functional as F

def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    
    output = model(input_ids)
    
    logits = output.logits
    
    all_log_probs = F.log_softmax(logits,dim= -1)
    
    token_log_probs = all_log_probs.gather(
        dim= -1,
        index=labels.unsqueeze(-1),
    ).squeeze(-1)
    
    if not return_token_entropy:
        return {
            "log_probs": token_log_probs
        }
    else:
        probs = all_log_probs.exp()
        
        entropy = -(probs*all_log_probs).sum(dim=-1)
        
        return {
            "log_probs": token_log_probs,
            "token_entropy": entropy
        }