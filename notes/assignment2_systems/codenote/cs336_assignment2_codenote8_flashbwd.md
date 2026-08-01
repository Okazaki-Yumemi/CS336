# backward


backward整体简单一些

```py
def flash_bwd_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    grad_output: torch.Tensor,
    logsumexp: torch.Tensor,
    is_causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    FlashAttention2 backward pass implemented in PyTorch.

    Args:
        q: torch.Tensor
            Query tensor of shape [batch_size, n_queries, d]
        k: torch.Tensor
            Key tensor of shape [batch_size, n_keys, d]
        v: torch.Tensor
            Value tensor of shape [batch_size, n_keys, d]
        output: torch.Tensor
            Output tensor from the forward pass of shape [batch_size, n_queries, d]
        grad_output: torch.Tensor
            Gradient of the loss with respect to the output tensor of shape [batch_size, n_queries, d]
        logsumexp: torch.Tensor
            Log-sum-exp values from the forward pass of shape [batch_size, n_queries]
        is_causal: bool
            Whether to use causal attention (not implemented in this assignment)
    Returns:
        dq: torch.Tensor
            Gradient with respect to the query tensor of shape [batch_size, n_queries, d]
        dk: torch.Tensor
            Gradient with respect to the key tensor of shape [batch_size, n_keys, d]
        dv: torch.Tensor
            Gradient with respect to the value tensor of shape [batch_size, n_keys, d]
    """
    
    D = q.shape[-1]
    
    scale = 1 / math.sqrt(D)
    
    scores = q @ k.transpose(-2, -1) * scale  # [batch_size, n_queries, n_keys]
    
    # causal mask
    query_positions = torch.arange(q.shape[1], device=q.device)
    key_positions = torch.arange(k.shape[1], device=k.device)
    
    if is_causal:
        query_positions = torch.arange(
            q.shape[1],
            device=q.device,
        )[:, None]
        
        key_positions = torch.arange(
            k.shape[1],
            device=k.device,
        )[None, :]
        
        causal_mask = query_positions >= key_positions  # [n_queries, n_keys]
        
        scores = scores.masked_fill(
            ~causal_mask,
            -torch.inf,
        )
        
    # 利用l重构p
    p = torch.exp(scores - logsumexp[..., None])  # [batch_size, n_queries, n_keys]
    
    D_vector = torch.sum(output * grad_output, dim=-1)  # [batch_size, n_queries]
    
    dV = p.transpose(-2, -1) @ grad_output  # [batch_size, n_keys, d]
    dP = grad_output @ v.transpose(-2, -1)  # [batch_size, n_queries, n_keys]
    
    dS = p * (dP - D_vector[..., None])  # [batch_size, n_queries, n_keys]
    dQ = (dS @ k ) * scale # [batch_size, n_queries, d]
    dK = (dS.transpose(-2, -1) @ q) * scale # [batch_size, n_keys, d]
    
    return dQ, dK, dV

compiled_flash_bwd_torch = torch.compile(flash_bwd_torch)


@staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        logsumexp, q, k, v, output = ctx.saved_tensors
        
        dq, dk, dv = compiled_flash_bwd_torch(
            q, k, v, output, grad_output, logsumexp, ctx.is_causal
        )
        
        return dq, dk, dv, None
```


backward的计算过程是这样的

首先构建scores = q @ k.transpose(-2, -1) * scale，是用来计算注意力分数的。然后根据是否是causal来决定是否应用causal mask。

p 是重计算得到的注意力权重，利用logsumexp来稳定计算。

D_vector 是输出和grad_output的点积，用来计算梯度。

dV\ dP \ dS \ dQ \ dK 分别是对v、p、scores、q、k的梯度计算。计算方法是类似于差分矩阵那种感觉，就是线性代数的问题了。

这个应该写在Pytorch那里面的。。写错地方了


# Triton 版本

```py

    # HOST 端
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
        
        # dq_acc 当前tile的dQ累加
        dq_acc += tl.dot(
            ds,
            k,
            input_precision = "ieee",
        ) * scale
        
        # 当前 qk tile 对 dK,dV 的贡献
        dk_partial = tl.dot(
            tl.trans(ds),
            q,
            input_precision = "ieee",
        ) * scale
        
        dv_partial = tl.dot(
            tl.trans(p),
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
            dk_partial,
            mask = kv_mask,
        )
        
        tl.atomic_add(
            dv_ptrs,
            dv_partial,
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
        dq_acc,
        boundary_check = (0,1),
    )
```

这个地方主要是，dQ是可以直接累加的，因为每个tile的dQ是独立的，不会有冲突。而dK和dV是需要用atomic add的，因为不同的tile可能会对同一个key产生贡献，所以需要保证线程安全。

然后别的路径和backward PyTorch的类似。

注意host端初始化 dK和dV的时候要用torch.zeros_like，而不是torch.empty_like，因为我们要累加，所以初始值必须是0。不然会有随机值，导致结果不正确。

```bash
==== test session starts ====
platform linux -- Python 3.13.12, pytest-9.0.3, pluggy-1.6.0
rootdir: /home/soyo/projects/CS336-2026/assignments/a2-systems
configfile: pyproject.toml
plugins: timeout-2.4.0, jaxtyping-0.3.9
collected 14 items / 11 deselected / 3 selected                                                                                                                                                                                            
tests/test_attention.py::test_flash_backward_pytorch PASSED
tests/test_attention.py::test_flash_backward_triton[False] PASSED
tests/test_attention.py::test_flash_backward_triton[True] PASSED

====3 passed, 11 deselected in 7.69s ====
```

