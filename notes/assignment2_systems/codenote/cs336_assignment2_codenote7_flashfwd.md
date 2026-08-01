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


# (b) Write a Triton kernel for the forward pass of FlashAttention-2 following Algorithm 1. Then, write another subclass of torch.autograd.Function calls this fused kernel in the forward pass.


```py

from __future__ import annotations

import torch 
import triton
import triton.language as tl


# 先搭Kernel 签名
# 题目有写，直接照抄
@triton.jit
def flash_fwd_kernel(
    Q_ptr,K_ptr,V_ptr,O_ptr,L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    # 确定当前program负责哪个tile
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    
    # Q_block_ptr
    Q_block_ptr = tl.make_block_ptr(
        base=Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order = (1,0),
    ) # Q tile shape = (Q_TILE_SIZE, D)
    
    # K_block_ptr
    K_block_ptr = tl.make_block_ptr(
        base=K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order = (1,0),
    ) # K tile shape = (K_TILE_SIZE, D)
    
    # V_block_ptr
    V_block_ptr = tl.make_block_ptr(
        base=V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order = (1,0),
    ) # V tile shape = (K_TILE_SIZE, D)
    
    # O_block_ptr
    O_block_ptr = tl.make_block_ptr(
        base=O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order = (1,0),
    )    # O tile shape = (Q_TILE_SIZE, D)
    
    # L_block_ptr
    L_block_ptr = tl.make_block_ptr(
        base=L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order = (0,),
    )   # L tile shape = (Q_TILE_SIZE,)
    
    # q 在整个 key 循环中保持不变 ， shape = (Q_TILE_SIZE, D)
    # boundary_check = (0, 1) 表示检查第 0 维和第 1 维
    q = tl.load(
        Q_block_ptr,
        boundary_check=(0, 1),
        padding_option="zero",
    )
    
    # 初始化online softmax状态
    m = tl.full((Q_TILE_SIZE,), -float("inf"), dtype=tl.float32) # running maximum
    
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32) # running exponential sum
    
    acc = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32) # running output numerator
    
    query_offsets = (
        query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    )
    
    for k_start in tl.range(0, N_KEYS, K_TILE_SIZE):
        # 读取当前key tile 和 value tile
        k = tl.load(
            K_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero",
        )
        
        v = tl.load(
            V_block_ptr,
            boundary_check=(0, 1),
            padding_option="zero",
        )
        
        key_offsets = k_start + tl.arange(0, K_TILE_SIZE)
        valid_keys = key_offsets < N_KEYS # shape = (K_TILE_SIZE,)
        
        if IS_CAUSAL:
            
            causal_mask = (
                key_offsets[None, :] <= query_offsets[:, None]
            )
            score_mask = valid_keys[None, :] & causal_mask
        
        else:
            score_mask = valid_keys[None, :]
        
        # shape = (K_TILE_SIZE, D)
        # 计算当前tile的注意力分数
        scores = tl.dot(q, tl.trans(k),) * scale # shape = (Q_TILE_SIZE, K_TILE_SIZE)
        
        scores = tl.where(
            score_mask,
            scores,
            -float("inf"),
        )
        # [None,:] 把 valid_keys 扩展成 (1, K_TILE_SIZE) 形状，方便广播
        
        tile_max = tl.max(scores, axis =1) # shape = (Q_TILE_SIZE,)
        m_new = tl.maximum(m, tile_max) # shape = (Q_TILE_SIZE,)

        # 重标定旧状态
        alpha = tl.exp(m - m_new) # shape = (Q_TILE_SIZE,)
        
        # 当前tile的未归一化概率
        p = tl.exp(scores - m_new[:, None]) # shape = (Q_TILE_SIZE, K_TILE_SIZE)
        
        # 更新分母
        l_new = alpha * l + tl.sum(p, axis=1) # shape = (Q_TILE_SIZE,)
        
        # 更新输出分子
        acc_new = alpha[:, None] * acc + tl.dot(p.to(v.dtype), v) # shape
        
        # 更新状态
        m = m_new
        l = l_new
        acc = acc_new
        
        # 更新block指针
        K_block_ptr = tl.advance(
            K_block_ptr,
            (K_TILE_SIZE, 0),
        )
        
        V_block_ptr = tl.advance(
            V_block_ptr,
            (K_TILE_SIZE, 0),
        )
    
    output = acc / l[:, None] # shape = (Q_TILE_SIZE, D)
    logsumexp = m + tl.log(l) # shape = (Q_TILE_SIZE,)
    
    tl.store(
        O_block_ptr,
        output,
        boundary_check = (0,1),
    )
    
    tl.store(
        L_block_ptr,
        logsumexp,
        boundary_check = (0,),
    )
    
        

class FlashAttentionTriton(torch.autograd.Function):
    
    @ staticmethod
    def forward(
        ctx, 
        q : torch.Tensor, 
        k : torch.Tensor, 
        v :torch.Tensor,
        is_causal : bool = False,
    ) -> torch.Tensor:
        
        batch_size, n_queries, d = q.shape
        
        _, n_keys, _ = k.shape
        
        output = torch.empty_like(q)
        
        logsumexp = torch.empty(
            (batch_size, n_queries), 
            dtype=torch.float32, 
            device=q.device
        )
        
        q_tile_size = 16
        k_tile_size = 16
        
        # q的循环在这个地方由 grid并行展开
        grid = (
            triton.cdiv(n_queries, q_tile_size), # query tile 数量
            batch_size, # batch size
        )
        
        flash_fwd_kernel[grid](
            q,
            k,
            v,
            output,
            logsumexp,
            
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            logsumexp.stride(0), logsumexp.stride(1),
            
            N_QUERIES=n_queries,
            N_KEYS=n_keys,
            scale=d**-0.5,
            
            D = d, # type: ignore
            Q_TILE_SIZE = q_tile_size, # type: ignore
            K_TILE_SIZE = k_tile_size, # type: ignore
            IS_CAUSAL = is_causal, # type: ignore
        )
        
        ctx.save_for_backward(
            logsumexp,
            q,
            k,
            v,
            output,
        )
        
        ctx.is_causal = is_causal
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        raise NotImplementedError("Backward pass is not implemented in this assignment.")
    
def flash_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    FlashAttention2 implementation using Triton kernels.

    Args:
        q: torch.Tensor
            Query tensor of shape [batch_size, n_queries, d]
        k: torch.Tensor
            Key tensor of shape [batch_size, n_keys, d]
        v: torch.Tensor
            Value tensor of shape [batch_size, n_keys, d]
        is_causal: bool
            Whether to use causal attention (not implemented in this assignment)

    Returns:
        output: torch.Tensor
            Output tensor of shape [batch_size, n_queries, d]
    """
    return FlashAttentionTriton.apply(q, k, v, is_causal)
```

核心代码架构层次讲解:

是这样的 def flash_attention_triton() 是对外接口，然后 class FlashAttentionTriton是负责调用内核的转接

传入q,k,v,is_causal之后，会从里面提取出来batch_size, n_queries, d = q.shape 和 n_keys = k.shape[1]，然后创建一个空的output和logsumexp张量

再设置tile size， 然后送去grid, triton.cdiv(n_queries, q_tile_size)表示query tile的数量，batch_size表示batch size

然后调用 falsh_fwd_kernel[grid]()，传入所有参数，包括q,k,v,output,logsumexp的指针，stride信息，N_QUERIES, N_KEYS, scale, D, Q_TILE_SIZE, K_TILE_SIZE, IS_CAUSAL

在 kernel里面，因为q的循环已经被我们用grid并行展开了，所以每个program只负责一个query tile，batch_index也是由grid并行展开的

先用query_tile_index = tl.program_id(0) 和 batch_index = tl.program_id(1) 来确定当前program负责哪个tile和batch,然后搭建Q_block_ptr (这个地方offset是固定的，因为不用循环)

stride_qq 是指每个query tile的行跨度，stride_qd是每个query tile的列跨度，类似的K_block_ptr和V_block_ptr也是一样的

K,V 的offset从 0,0开始， 然后order = (1,0)表示按行优先存储

初始化 m, l, acc用于online softmax

然后和pytorch里面实现一样了，

把query_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE) 计算出来， 然后循环k_start in tl.range(0, N_KEYS, K_TILE_SIZE) 来处理每个key tile


如果是 CAUSAL的，就要计算causal_mask = (key_offsets[None, :] <= query_offsets[:, None])，然后score_mask = valid_keys[None, :] & causal_mask

最后output 和 logsumexp得到之后，用tl.store把结果存回O_block_ptr和L_block_ptr

```bash
=== test session starts ===
platform linux -- Python 3.13.12, pytest-9.0.3, pluggy-1.6.0
rootdir: /home/soyo/projects/CS336-2026/assignments/a2-systems
configfile: pyproject.toml
plugins: timeout-2.4.0, jaxtyping-0.3.9
collected 14 items / 12 deselected / 2 selected                                                                                                                                                                                               

tests/test_attention.py::test_flash_forward_pass_triton[False] PASSED
tests/test_attention.py::test_flash_forward_pass_triton[True] PASSED
=== 2 passed, 12 deselected in 1.47s ===
```

突然发现我们把(c) 也做了，那就结束了