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