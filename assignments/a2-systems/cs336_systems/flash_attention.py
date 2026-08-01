import torch

def _flash_forward_tiled_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_tile_size : int = 16,
    k_tile_size : int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回output和每行的logsumexp。注意：这是一个慢速的参考实现，使用纯PyTorch（没有Triton），按FlashAttention-2算法分块。"""
    
    batch_size, n_queries, d = q.shape
    batch_size_k , n_keys, d_k = k.shape
    
    assert batch_size == batch_size_k, "Batch sizes of q and k must match"
    assert d == d_k, "Feature dimensions of q and k must match"
    assert k.shape == v.shape, "Shapes of k and v must match"
    
    scale = d**-0.5
    
    output = torch.empty_like(q)
    
    logsumexp = torch.empty(
        batch_size,
        n_queries,
        device = q.device,
        dtype = torch.float32
    )
    
    for q_start in range(0, n_queries, q_tile_size):
        # 拦截，防止超出范围
        q_end = min(q_start + q_tile_size, n_queries)
        q_tile = q[:, q_start:q_end, :]  # [batch_size, q_tile_size, d]
        
        current_q_size = q_end - q_start
        
        # 为这个query tile 初始化 online softmax 状态
        m = torch.full(
            (batch_size, current_q_size),
            -torch.inf,
            device = q.device,
            dtype = torch.float32
        )
        
        l = torch.zeros(
            batch_size,
            current_q_size,
            device = q.device,
            dtype = torch.float32
        )
        
        acc = torch.zeros(
            batch_size,
            current_q_size,
            d,
            device = q.device,
            dtype = torch.float32
        )
        
        # m = running maximum, l = running exponential sum, acc = running output numerator
        
        q_float = q_tile.float() # 确保q_tile是float32
        
        for k_start in range(0, n_keys, k_tile_size):
            k_end = min(k_start + k_tile_size, n_keys)
            k_tile = k[:, k_start:k_end, :] # [batch_size, k_tile_size, d]
            v_tile = v[:, k_start:k_end, :] # [batch_size, k_tile_size, d]
            
            # 计算当前tile的注意力分数
            
            k_float = k_tile.float() # 确保k_tile是float32
            v_float = v_tile.float() # 确保v_tile是float32
            
            scores = q_float @ k_float.transpose(-2, -1) * scale  # [batch_size, q_tile_size, k_tile_size]
            
            #更新行最大值
            tile_max = torch.max(scores, dim=-1).values  # [batch_size, q_tile_size]
            m_new = torch.maximum(m, tile_max)  # [batch_size, q_tile_size]

            #重标定旧状态
            alpha = torch.exp(m - m_new)  # [batch_size, q_tile_size]
            
            # 当前tile的未归一化概率
            p = torch.exp(scores - m_new[..., None])  # [batch_size, q_tile_size, k_tile_size]

            # 更新分母
            l_new = alpha * l + torch.sum(p, dim=-1)  # [batch_size, q_tile_size]
            
            # 更新输出分子
            acc_new = alpha[..., None] * acc + p @ v_float  # [batch_size, q_tile_size, d]
            
            # 更新状态
            m = m_new
            l = l_new
            acc = acc_new
            
        # 完成当前query tile
        output_tile = acc / l[..., None]  # [batch_size, q_tile_size, d]
        logsumexp_tile = m + torch.log(l)  # [batch_size, q_tile_size]
        
        output[:, q_start:q_end, :] = output_tile.to(q.dtype)
        logsumexp[:, q_start:q_end] = logsumexp_tile

    return output, logsumexp

class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        
        if is_causal:
            raise NotImplementedError("Causal mode is not implemented in this reference implementation.")
        
        
        output, logsumexp = _flash_forward_tiled_pytorch(q, k, v)
        
        ctx.save_for_backward(logsumexp, q, k, v, output)
        ctx.is_causal = is_causal
        
        return output
    
    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ):
        
        raise NotImplementedError
    
def flash_attention_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    A convenience wrapper around the FlashAttentionPytorch autograd function.
    """
    return FlashAttentionPytorch.apply(q, k, v, is_causal)