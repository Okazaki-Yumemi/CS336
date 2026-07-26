# 1.这个 Problem 在做什么

输入
```
in_features: (..., sequence_length, d_model)
```

一次性为所有heads生成 Q K V

然后给d_model拆成 num_heads x head_dim

```
x                         (..., seq, d_model)

Q/K/V projection          (..., seq, d_model)

拆 heads                   (..., heads, seq, head_dim)

可选 RoPE(Q, K)           (..., heads, seq, head_dim)

scaled attention           (..., heads, seq, head_dim)

合并 heads                 (..., seq, d_model)

output projection          (..., seq, d_model)
```

# “optimized batched implementation” 是什么意思

Adapter 特别强调：

handle the key, query, and value projections for all heads in a single matrix multiply

一次性投影，然后重排

```py

q = rearrange(
  q,
  "... seq (heads head_dim) -> ... heads seq head_dim",
  heads = num_heads
)
```



# 代码实现

```py

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = Linear(d_model, d_model, device, dtype)
        self.k_proj = Linear(d_model, d_model, device, dtype)
        self.v_proj = Linear(d_model, d_model, device, dtype)
        self.o_proj = Linear(d_model, d_model, device, dtype)
        
        if theta is not None and max_seq_len is not None:
            self.rope = RoPE(theta, self.head_dim, max_seq_len, device)
        else:
            self.rope = None
            
    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ):
        seq_len = x.shape[-2]
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        Q = rearrange(
            Q,
            "... seq (heads head_dim) -> ... heads seq head_dim",
            heads=self.num_heads,
        )
        K = rearrange(
            K,
            "... seq (heads head_dim) -> ... heads seq head_dim",
            heads=self.num_heads,
        )
        V = rearrange(
            V,
            "... seq (heads head_dim) -> ... heads seq head_dim",
            heads=self.num_heads,
        )
        # 位置编码
        if self.rope is not None:           
            #token_positions 需要手动加head维度
            if token_positions is not None:
                token_positions = token_positions.unsqueeze(-2)
            else:
                token_positions = torch.arange(seq_len, device=Q.device)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        # causal mask
        
        mask = torch.tril(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
        )
        
        scaled_attention_output = scaled_dot_product_attention(Q, K, V, mask)
        
        scaled_attention_output = rearrange(
            scaled_attention_output,
            "... heads seq head_dim -> ... seq (heads head_dim)",
            heads=self.num_heads,
        )
        
        return self.o_proj(scaled_attention_output)
```

这个地方比较复杂的就是首先要给q,k,v,o投影，然后

比较复杂的点是:

多头注意力的多个head的参数是被拼起来了的，所以我们要给它拆开，拆开为head数目和每个head的维度，然后再做scaled attention，最后再把head拼回去。

里面用的 einops 的 rearrange 来做这个拆分和拼接。

把 最后一个维度拆开为 (heads head_dim) 

然后看位置编码,
因为token_positions 可能没有head维度，所以我们要手动加上去。

然后做scaled attention，注意要加上causal mask。

最后给output的 heads 和 head_dim 拼回去，最后再投影回 d_model。(用 o_proj)


# 测试函数

```py

def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_model"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_model"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    mha = MultiHeadSelfAttention(
        d_model= d_model,
        num_heads= num_heads,
        device= q_proj_weight.device,
        dtype= q_proj_weight.dtype,
    )
    with torch.no_grad():
        mha.q_proj.weight.copy_(q_proj_weight)
        mha.k_proj.weight.copy_(k_proj_weight)
        mha.v_proj.weight.copy_(v_proj_weight)
        mha.o_proj.weight.copy_(o_proj_weight)
    
    return mha(in_features)


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_model"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_model"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    mha = MultiHeadSelfAttention(
            d_model= d_model,
            num_heads= num_heads,
            theta= theta,
            max_seq_len= max_seq_len,
            device= q_proj_weight.device,
            dtype= q_proj_weight.dtype,
        )
    with torch.no_grad():
        mha.q_proj.weight.copy_(q_proj_weight)
        mha.k_proj.weight.copy_(k_proj_weight)
        mha.v_proj.weight.copy_(v_proj_weight)
        mha.o_proj.weight.copy_(o_proj_weight)
        
    return mha(in_features, token_positions=token_positions) 
```

要注意参数要传完整