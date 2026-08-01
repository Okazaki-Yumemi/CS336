from __future__ import annotations

import torch 
import triton
import triton.language as tl
import math

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
    
    output = (acc / l[:, None]).to(q.dtype) # shape = (Q_TILE_SIZE, D)
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

@triton.jit
def flash_bwd_kernel(
    Q_ptr,K_ptr,V_ptr,O_ptr,L_ptr,
    dO_ptr,dQ_ptr,dK_ptr,dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    stride_dob, stride_doq, stride_dod,
    stride_dqb, stride_dqq, stride_dqd,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,
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
    
    # dO_block_ptr
    dO_block_ptr = tl.make_block_ptr(
        base=dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order = (1,0),
    )    # dO tile shape = (Q_TILE_SIZE, D)
    
    # dQ_block_ptr
    dQ_block_ptr = tl.make_block_ptr(
        base=dQ_ptr + batch_index * stride_dqb,
        shape=(N_QUERIES, D),
        strides=(stride_dqq, stride_dqd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order = (1,0),
    )    # dQ tile shape = (Q_TILE_SIZE, D)
    
    # dK_block_ptr
    dK_block_ptr = tl.make_block_ptr(
        base=dK_ptr + batch_index * stride_dkb,
        shape=(N_KEYS, D),
        strides=(stride_dkk, stride_dkd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order = (1,0),
    ) # dK tile shape = (K_TILE_SIZE, D)
    
    # dV_block_ptr
    dV_block_ptr = tl.make_block_ptr(
        base=dV_ptr + batch_index * stride_dvb,
        shape=(N_KEYS, D),
        strides=(stride_dvk, stride_dvd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order = (1,0),
    ) # dV tile shape = (K_TILE_SIZE, D)
    
    query_positions = (
        query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    )
    
    valid_queries = query_positions < N_QUERIES # shape = (Q_TILE_SIZE,)
    
    d_offsets = tl.arange(0, D) # shape = (K_TILE_SIZE,)
    
    q = tl.load(
        Q_block_ptr,
        boundary_check=(0, 1),
        padding_option="zero",
    )
    
    o = tl.load(
        O_block_ptr,
        boundary_check=(0, 1),
        padding_option="zero",
    )
    
    logsumexp = tl.load(
        L_block_ptr,
        boundary_check=(0,),
        padding_option="zero",
    )
    do = tl.load(
        dO_block_ptr,
        boundary_check=(0, 1),
        padding_option="zero",
    )
    
    # D_i = sum_j o_ij * do_ij
    delta = tl.sum(o.to(tl.float32) * do, axis=1) # shape = (Q_TILE_SIZE,)
    
    dq_acc = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    
    for k_start in tl.range(0, N_KEYS, K_TILE_SIZE):
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
        
        key_positions = k_start + tl.arange(0, K_TILE_SIZE)
        valid_keys = key_positions < N_KEYS # shape = (K_TILE_SIZE,)
        
        scores = tl.dot(
            q,
            tl.trans(k),
            input_precision = "ieee",
        )*scale
        
        score_mask = (
            valid_queries[:, None] & valid_keys[None, :]
        )
        
        if IS_CAUSAL:
            score_mask = (
                score_mask
                & (key_positions[None, :] <= query_positions[:, None])
            )
        
        scores = tl.where(
            score_mask,
            scores,
            -float("inf"),
        )
        
        # P = exp(S - L)
        p = tl.exp(
            scores - logsumexp[:, None],
        )
        
        # dP = dO V^T
        dp = tl.dot(
            do,
            tl.trans(v),
            input_precision = "ieee",
        ) # shape = (Q_TILE_SIZE, K_TILE_SIZE)
        
        # dS = P *(dP - D)
        ds = p * (
            dp - delta[:, None] 
        )
        
        ds_dot = ds.to(q.dtype) 
        p_dot = p.to(q.dtype)
        
        
        # dq_acc 当前tile的dQ累加
        dq_acc += tl.dot(
            ds_dot,
            k,
            input_precision = "ieee",
        ) * scale
        
        # 当前 qk tile 对 dK,dV 的贡献
        dk_partial = tl.dot(
            tl.trans(ds_dot),
            q,
            input_precision = "ieee",
        ) * scale
        
        dv_partial = tl.dot(
            tl.trans(p_dot),
            do,
            input_precision = "ieee",
        )
        # 普通 pointer,供atomic add使用
        dk_ptrs =(
            dK_ptr
            + batch_index * stride_dkb
            + key_positions[:, None] * stride_dkk
            + d_offsets[None, :] * stride_dkd
        )
        
        dv_ptrs = (
            dV_ptr
            + batch_index * stride_dvb
            + key_positions[:, None] * stride_dvk
            + d_offsets[None, :] * stride_dvd
        )
        
        kv_mask = valid_keys[:, None]
        
        tl.atomic_add(
            dk_ptrs,
            dk_partial.to(k.dtype),
            mask = kv_mask,
        )
        
        tl.atomic_add(
            dv_ptrs,
            dv_partial.to(v.dtype),
            mask = kv_mask,
        )
        
        K_block_ptr = tl.advance(
            K_block_ptr,
            (K_TILE_SIZE, 0),
        )
        
        V_block_ptr = tl.advance(
            V_block_ptr,
            (K_TILE_SIZE, 0),
        )
    
    tl.store(
        dQ_block_ptr,
        dq_acc.to(q.dtype),
        boundary_check = (0,1),
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
        logsumexp, q, k, v, output = ctx.saved_tensors
        
        q_tile_size = 16
        k_tile_size = 16
        
        grid = (
            triton.cdiv(q.shape[1], q_tile_size), # query tile 数量
            q.shape[0], # batch size
        )
        
        dQ = torch.empty_like(q)
        dK = torch.zeros_like(k)
        dV = torch.zeros_like(v)
        
        
        flash_bwd_kernel[grid](
            q,
            k,
            v,
            output,
            logsumexp,
            
            grad_output,
            dQ,
            dK,
            dV,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            logsumexp.stride(0), logsumexp.stride(1),
            grad_output.stride(0), grad_output.stride(1), grad_output.stride(2),
            dQ.stride(0), dQ.stride(1), dQ.stride(2),
            dK.stride(0), dK.stride(1), dK.stride(2),
            dV.stride(0), dV.stride(1), dV.stride(2),
            
            N_QUERIES= q.shape[1],
            N_KEYS = k.shape[1],
            scale = q.shape[2]**-0.5,
            D = q.shape[2], # type: ignore
            Q_TILE_SIZE = q_tile_size, # type: ignore
            K_TILE_SIZE = k_tile_size, # type: ignore
            IS_CAUSAL = ctx.is_causal, # type: ignore
        )
        
        return (
            dQ,
            dK,
            dV,
            None,
        )
        
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