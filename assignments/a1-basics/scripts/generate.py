from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from cs336_basics.bpe import train_bpe
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.model import Linear
from cs336_basics.model import Embedding
from cs336_basics.model import RMSNorm
from cs336_basics.model import SwiGLU
from cs336_basics.model import RoPE
from cs336_basics.model import softmax
from cs336_basics.model import scaled_dot_product_attention
from cs336_basics.model import MultiHeadSelfAttention
from cs336_basics.model import TransformerBlock
from cs336_basics.model import TransformerLM
from cs336_basics.model import cross_entropy_loss
from cs336_basics.model import AdamW
from cs336_basics.model import cosine_lr_schedule
from cs336_basics.model import gradient_clipping
from cs336_basics.model import data_loader
from cs336_basics.model import save_checkpoint
from cs336_basics.model import load_checkpoint

def generate(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    context_length: int,
    temperature: float = 1.0,
    top_p : float = 1.0,
    eos_token_id: int | None = None,
    device: str = "cuda",
) -> str:
    """根据给定的 prompt 生成文本"""
    
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")

    if not 0 < top_p <= 1.0:
        raise ValueError("top_p must be in the range (0, 1]")
    
    
    
    model.eval()
    
    # prompt 编码
    input_ids = tokenizer.encode(prompt)
    # 空值拦截
    if len(input_ids) == 0:
        raise ValueError("prompt is empty after encoding")
    
    # 转为 tensor 并移动到指定设备
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)  # shape: (1, seq_len)
    # 生成的 token id 列表
    generated_ids = input_ids.tolist()[0]
    
    #循环最多生成 max_new_tokens 个 token
    for _ in range(max_new_tokens):
        # 如果当前输入长度超过 context_length，则截断
        if input_ids.size(1) > context_length:
            input_ids = input_ids[:, -context_length:]
            
        # 前向传播得到 logits
        with torch.no_grad():
            model_output = model(input_ids)
            logits = model_output[0, -1, :]  # 取最后一个 token 的 logits
            
        #应用温度缩放
        logits = logits / temperature
        #softmax 得到概率分布
        probs = softmax(logits,-1)
        
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(
                probs,
                descending = True,
            )
            
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            remove_mask = cumulative_probs > top_p
            
            # 将 mask 右移一位, 保留第一个跨过 top_p 的token
            remove_mask[1:] = remove_mask[:-1].clone()
            remove_mask[0] = False
            
            sorted_probs = sorted_probs.masked_fill(
                remove_mask,
                0.0,
            )
            
            sorted_probs = sorted_probs / sorted_probs.sum()
            
            #这里抽到的是排序后列表种的位置
            sample_rank = torch.multinomial(
                sorted_probs,
                num_samples=1,
            )
            
            # 映射为回原始词表 ID
            next_token_id = sorted_indices[sample_rank]
        
        else:
            
            # 不做top-p 截断, 直接在原词表分布上采样
            next_token_id = torch.multinomial(probs, num_samples=1)
        
        token_id = next_token_id.item()
        
        
        if eos_token_id is not None and token_id == eos_token_id:
            break
    
        # 将采样得到的 token id 添加到生成的 token id 列表中
        generated_ids.append(token_id)
        # 将采样得到的 token id 添加到输入中，以便下一次迭代
        input_ids = torch.cat([input_ids, next_token_id.view(1,1)], dim=1)
    # 从生成的 token id 列表中解码为字符串
    return tokenizer.decode(generated_ids)