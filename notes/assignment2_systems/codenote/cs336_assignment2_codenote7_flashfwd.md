# (a) Write a pure PyTorch (no Triton) autograd.Function that implements the FlashAttention-2 forward pass

先写一个“按 FlashAttention-2 算法分块，但全部使用 PyTorch”的慢速参考实现。它之后会成为 Triton 版本的逐步对照基准

```py
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
        
        for k_start in range(0, n_keys, k_tile_size):
            k_end = min(k_start + k_tile_size, n_keys)
            k_tile = k[:, k_start:k_end, :] # [batch_size, k_tile_size, d]
            v_tile = v[:, k_start:k_end, :] # [batch_size, k_tile_size, d]
            
            # 计算当前tile的注意力分数
            q_float = q_tile.float() # 确保q_tile是float32
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
        
        output, logsumexp = _flash_forward_tiled_pytorch(q, k, v)
        
        ctx.save_for_backward(q, k, v, logsumexp)
        ctx.is_causal = is_causal
        
        return output
    
    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ):
        
        raise NotImplementedError
```

测试

```bash
=== test session starts ===
platform linux -- Python 3.13.12, pytest-9.0.3, pluggy-1.6.0
rootdir: /home/soyo/projects/CS336-2026/assignments/a2-systems
configfile: pyproject.toml
plugins: timeout-2.4.0, jaxtyping-0.3.9
collected 14 items / 13 deselected / 1 selected                                                                                                                                                                                               

tests/test_attention.py::test_flash_forward_pass_pytorch PASSED

=== 1 passed, 13 deselected in 0.12s ===
```

笔记部分:

这个地方代码本质是在维护三个不变量

每处理完一个key block ，都可以认为下面三个等式成立

m = max(当前所有处理过的key block的scores)

l = sum(exp(当前所有处理过的key block的scores - m))

acc = sum(exp(当前所有处理过的key block的scores - m) * v)

只要每一轮更新后，让这三个不变量仍然成立，那么处理完全部key后， O = acc/l 就必定是正确的输出

上面的代码里面，我们要做的就是一开始维护m, l, acc的初始值，然后每处理完一个key block，就更新这三个不变量，最后输出的时候用acc/l就可以得到正确的结果。

分块的处理不多谈了。