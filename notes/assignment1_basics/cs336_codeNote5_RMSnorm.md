# RMSNorm

官方要的代码框架

```py
def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None)
```

- d_model: int  Hidden dimension of the model
- eps: float = 1e-5  Epsilon value for numerical stability
- device: torch.device | None = None  Device to store the parameters on
- dtype: torch.dtype | None = None  Data type of the parameters


```py
def forward(self, x: torch.Tensor) -> torch.Tensor
```

Note: Remember to upcast your input to torch.float32 before performing the normalization 
(and later downcast to the original dtype), as described above.



**可学习参数** 
g.shape = (d_model,)  # learnable gain parameter

通常写作

```py
self.weight = nn.Parameter(
  torch.ones(
    d_model,
    device=device,
    dtype=dtype,
  )
)
```


**沿着最后一个维度归一化**

输入x.shape = (..., d_model)  # shape of the input tensor

```
x.pow(2).mean(dim=-1, keepdim=True).sqrt() + eps
```

代码

```py

class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model:int,
        eps:float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.eps = eps
        
        self.weight = nn.Parameter(
            torch.ones(
                self.d_model,
                device= device,
                dtype=  dtype,
            )
        )
        
    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        in_dtype = x.dtype
        
        x_float = x.to(torch.float32)
        
        mean_square = torch.mean(x_float ** 2, dim=-1, keepdim=True)
        
        inverse_rms = torch.rsqrt(mean_square + self.eps)
        
        normalized_x = x_float * inverse_rms
        
        x_original = normalized_x.to(in_dtype)
        return x_original * self.weight
```

难点主要是 RMSNorm 的公式，其他的都是照着 Linear 和 Embedding 写的。

$$ RMSNorm(a_i) = \frac{a_i}{RMS(a)} * g_i $$

$$ RMS(a) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} a_i^2 + \epsilon} $$

# Adapter 适配

没啥难的不提了

# 测试

通过