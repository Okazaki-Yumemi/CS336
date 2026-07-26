# Transformer Language Model (LM) Implementation Notes


Transformer LM 接受 Transformer Block 的全部参数,然后还额外接受
`vocab_size`
`context_length`
`num_layers`


```py
class TransformerLM(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        vocab_size: int,
        num_layers: int,
        context_length: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.context_length = context_length
        
        self.token_embedding = Embedding(vocab_size, d_model, device, dtype)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, max_seq_len, theta, device, dtype)
            for _ in range(num_layers)
        ])
        
        self.rmsnorm_final = RMSNorm(d_model)
        
        self.output_projection = Linear(d_model, vocab_size, device, dtype)
        
    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None
    )-> torch.Tensor:
        
        x =  self.token_embedding(x)
        
        for block in self.transformer_blocks:
            x = block(x, token_positions)
        
        x = self.rmsnorm_final(x)
        x = self.output_projection(x)

        return x
```

adapter

```py

def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    transformer_lm = TransformerLM(
        d_model= d_model,
        num_layers= num_layers,
        num_heads= num_heads,
        d_ff= d_ff,
        context_length= context_length,
        vocab_size= vocab_size,
        theta= rope_theta,
        device = weights["token_embeddings.weight"].device,
        dtype = weights["token_embeddings.weight"].dtype,
        max_seq_len= context_length,
    )
    
    with torch.no_grad():
        transformer_lm.token_embedding.weight.copy_(weights["token_embeddings.weight"])
        transformer_lm.rmsnorm_final.weight.copy_(weights["ln_final.weight"])
        transformer_lm.output_projection.weight.copy_(weights["lm_head.weight"])
        
        for layer_idx in range(num_layers):
            block = transformer_lm.transformer_blocks[layer_idx]
            assert isinstance(block, TransformerBlock)
            
            block.attention.q_proj.weight.copy_(weights[f"layers.{layer_idx}.attn.q_proj.weight"])
            block.attention.k_proj.weight.copy_(weights[f"layers.{layer_idx}.attn.k_proj.weight"])
            block.attention.v_proj.weight.copy_(weights[f"layers.{layer_idx}.attn.v_proj.weight"])
            block.attention.o_proj.weight.copy_(weights[f"layers.{layer_idx}.attn.output_proj.weight"])
            
            block.swiGLU.w1.weight.copy_(weights[f"layers.{layer_idx}.ffn.w1.weight"])
            block.swiGLU.w2.weight.copy_(weights[f"layers.{layer_idx}.ffn.w2.weight"])
            block.swiGLU.w3.weight.copy_(weights[f"layers.{layer_idx}.ffn.w3.weight"])
            
            block.rmsnorm1.weight.copy_(weights[f"layers.{layer_idx}.ln1.weight"])
            block.rmsnorm2.weight.copy_(weights[f"layers.{layer_idx}.ln2.weight"])
            
    
    return transformer_lm(in_indices)

```

