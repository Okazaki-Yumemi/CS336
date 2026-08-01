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

