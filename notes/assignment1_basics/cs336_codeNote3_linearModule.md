# 这个不难，直接给完整的代码就行

```py
import math

import torch
from einops import einsum
from torch import nn

class Linear(nn.Module):
    def __init__(
        self,
        in_features:int,
        out_features:int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        
        self.sigma = math.sqrt(2 / (self.in_features + self.out_features))

        # 创建shape为(out_features, in_features)的权重张量，并使用正态分布初始化
        self.weight = nn.Parameter(
            torch.empty(
                self.out_features,
                self.in_features,
                device= self.device,
                dtype= self.dtype,
            )
        )
        
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std= self.sigma,
            a = -3*self.sigma,
            b = 3*self.sigma,
        )
        
    def forward(self,
                x: torch.Tensor
        ) -> torch.Tensor:
        output = einsum(x, self.weight, "... d_in , d_out d_in -> ... d_out")
        return output
```

上面主要是注意 初始化权重张量时使用了截断正态分布，截断范围为[-3σ, 3σ]，其中σ = sqrt(2 / (in_features + out_features))。

forward用了einops的einsum函数来实现矩阵乘法，输入张量x的最后一维是in_features，输出张量的最后一维是out_features。

# Adapter适配
```py
def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    linear = Linear(
        in_features=d_in,
        out_features=d_out,
        device=weights.device,
        dtype=weights.dtype
        )
    
    with torch.no_grad():
        linear.weight.copy_(weights)
        
    return linear(in_features)
```

adapter直接传入了weights参数，所以我们要在Linear类中初始化权重张量后，使用`copy_`方法将传入的weights复制到Linear实例的权重张量中。然后调用Linear实例的forward方法来计算输出。

# 测试

```bash

uv run pytest -k test_linear
== test session starts ===
platform linux -- Python 3.13.12, pytest-9.0.2, pluggy-1.6.0
rootdir: /home/soyo/projects/CS336-2026/assignments/a1-basics
configfile: pyproject.toml
plugins: timeout-2.4.0, jaxtyping-0.3.9
collected 48 items / 47 deselected / 1 selected                                                           

tests/test_model.py::test_linear PASSED

== 1 passed, 47 deselected in 0.08s ==
```