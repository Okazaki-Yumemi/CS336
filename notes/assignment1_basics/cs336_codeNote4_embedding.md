# Implementing the embedding module.

官方要求的接口

```py

def __init__(
  self,
  num_embeddings: int, # size of the vocabulary
  embedding_dim: int, # Dimension of the embedding vectors
  device: torch.device | None = None, # device to store the parameters on
  dtype: torch.dtype | None = None, # Data type of the parameters
  )

def forward(
  self,
  token_ids: torch.Tensor, # Tensor of shape (batch_size, sequence_length) containing token IDs
  ) -> torch.Tensor: # Returns a tensor of shape (batch_size, sequence_length, embedding_dim) containing the corresponding embeddings
```

Make sure to
- subclass nn.Module
- call super().__init__() in the constructor
- initialize your embedding matrix as an nn.Parameter of shape (num_embeddings, embedding_dim) with the specified device and dtype
- store the embedding matrix with d_model being  the final dimension


代码实现

```py

class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings : int,
        embedding_dim : int,
        device: torch.device | None = None,
        dtype : torch.dtype | None = None
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        
        self.weight = nn.Parameter(
            torch.empty(
                self.num_embeddings ,
                self.embedding_dim,
                device= self.device,
                dtype= self.dtype
                )
            )
        nn.init.trunc_normal_(
            self.weight,
            std= 1,
            a = -3,
            b = 3,
        )
    
    def forward(
        self,
        token_ids:torch.Tensor
    ) -> torch.Tensor:
        return self.weight[token_ids]
        
```

这边不太一样的是题目要求的 embedding截断维度和linear不一样

然后 forward的方法就是直接用token_ids去索引embedding矩阵就可以了，返回的tensor shape是(batch_size, sequence_length, embedding_dim)

因为embedding本来也就是映射，所以就是查表

# adapter 适配

```py
def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """
    embedding = Embedding(
        num_embeddings= vocab_size,
        embedding_dim= d_model,
        device= weights.device,
        dtype=  weights.dtype,
    )
    
    with torch.no_grad():
        embedding.weight.copy_(weights)
    
    return embedding(token_ids)

```

逻辑和linear类似。


# 测试

```bash

uv run pytest -k test_embeddin

== test session starts ==
platform linux -- Python 3.13.12, pytest-9.0.2, pluggy-1.6.0
rootdir: /home/soyo/projects/CS336-2026/assignments/a1-basics
configfile: pyproject.toml
plugins: timeout-2.4.0, jaxtyping-0.3.9
collected 48 items / 47 deselected / 1 selected                                                           

tests/test_model.py::test_embedding PASSED

== 1 passed, 47 deselected in 0.12s ==
```
