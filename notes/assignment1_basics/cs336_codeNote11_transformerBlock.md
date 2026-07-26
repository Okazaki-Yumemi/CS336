# Transformer Block

```
x: (..., seq, d_model)
│
├─ RMSNorm
├─ Multi-Head Self-Attention
└─ 与原 x 做 residual addition
        ↓
        h
│
├─ RMSNorm
├─ SwiGLU
└─ 与 h 做 residual addition
        ↓
output: (..., seq, d_model)
```

```py

x = x + attention(attn_norm(x))

x = x + ffn(ffn_norm(x))
```

话不多说上代码,自己看就懂了

```py
class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
                
        
        
        self.rmsnorm1 = RMSNorm(d_model)
        self.rmsnorm2 = RMSNorm(d_model)
        
        self.swiGLU = SwiGLU(d_model, d_ff)
        self.attention = MultiHeadSelfAttention(d_model, num_heads, theta, max_seq_len)
    
    def forward(
        self,
        x: torch.Tensor,
    )-> torch.Tensor:
        
        x = x + self.attention(self.rmsnorm1(x))
        x = x + self.swiGLU(self.rmsnorm2(x))
        
        return x

def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    transformer_block = TransformerBlock(
        d_model= d_model,
        num_heads= num_heads,
        d_ff= d_ff, 
        max_seq_len= max_seq_len,
        theta= theta,
    )
    
    with torch.no_grad():
        transformer_block.attention.q_proj.weight.copy_(weights["attn.q_proj.weight"])
        transformer_block.attention.k_proj.weight.copy_(weights["attn.k_proj.weight"])
        transformer_block.attention.v_proj.weight.copy_(weights["attn.v_proj.weight"])
        transformer_block.attention.o_proj.weight.copy_(weights["attn.output_proj.weight"])
        
        
        transformer_block.swiGLU.w1.weight.copy_(weights["ffn.w1.weight"])
        transformer_block.swiGLU.w2.weight.copy_(weights["ffn.w2.weight"])
        transformer_block.swiGLU.w3.weight.copy_(weights["ffn.w3.weight"])
        
        transformer_block.rmsnorm1.weight.copy_(weights["ln1.weight"])
        transformer_block.rmsnorm2.weight.copy_(weights["ln2.weight"])

    return transformer_block(in_features)
```