# Scaled Dot Product Attention

handle keys and queries of shape(batch_size , ... , seq_len ,d_k)

and values of shape (batch_size , ... , seq_len ,d_v)

implementation should return a tensor of shape (batch_size , ... , seq_len ,d_v)

```py
def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None
)-> torch.Tensor:
    
    scores = einsum(q, k, "... query d_k , ... key d_k -> ... query key")
    d_k = q.shape[-1]
    
    scaled_scores = scores / math.sqrt(d_k)
    
    if mask is not None:
        scaled_scores = scaled_scores.masked_fill(~mask, float('-inf'))
    
    softmax_scores = softmax(scaled_scores, -1)
    
    attention_output = einsum(
        softmax_scores,
        v,
        "... query key , ... key d_v -> ... query d_v"
    )
    
    return attention_output

```

这一节把前面学的串起来了，主要就是熟悉这个 einsum 的用法，和前面学的 softmax 结合起来，最后就是 scaled dot product attention 了。