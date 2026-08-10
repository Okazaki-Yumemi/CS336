import torch
from transformers import PreTrainedTokenizerBase


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:

    all_full_ids = []
    all_full_masks = []
    
    max_len = 0
    
    
    for prompt,output in zip(prompt_strs,output_strs):
        prompt_ids = tokenizer(prompt,add_special_tokens=False)["input_ids"]
        output_ids = tokenizer(output,add_special_tokens=False)["input_ids"]

        full_ids = prompt_ids + output_ids
        
        mask = [0]* len(prompt_ids) + [1] * len(output_ids)
        
        all_full_ids.append(full_ids)
        
        all_full_masks.append(mask)
        
        max_len = max(max_len, len(full_ids))
    
    for full_ids, full_masks in zip(all_full_ids,all_full_masks):
        
        len_for_pad = max_len - len(full_ids)
        
        full_ids += [tokenizer.pad_token_id]*len_for_pad
        full_masks += [0]*len_for_pad
    
    full_ids_tensor = torch.tensor(all_full_ids)
    
    full_masks_tensor = torch.tensor(all_full_masks)
    
    input_ids = full_ids_tensor[:,:-1]
    labels = full_ids_tensor[:,1:]
    response_mask = full_masks_tensor[:,1:]
    
    return{
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }
    
    
        
        
        
        
        
    