# SwiGLU的实现，SwiGLU是Gated Linear Unit的一个变体，使用了Swish激活函数。SwiGLU的公式如下：

SwiGLU 不会让token之间交流，只对token的最后一维进行相同的非线性变换

swiglu的公式伪代码:

```

SwiGLU(x) = W2(SiLU(W1x) * W3x)

拆开算就是

a = W1x
b = W3x
SiLU(a) = a * sigmoid(a)
h = SiLU(a) * b
y = W2h

```

对应代码逻辑

```py
a = self.w1(x)
b = self.w3(x)

silu_a = a * torch.sigmoid(a)
h = silu_a * b

y = self.w2(h)
```

其中, W1 生成门控信号

从 d_model维度放大到d_ff维度

，W3 生成候选信号

传递候选内容
从 d_model维度放大到d_ff维度

，W2 生成输出信号。

从 d_ff维度压缩回d_model维度

几种要区分的计算


## Linear

本质是
```py

einsum(
    x,
    weight,
    "... d_in, d_out d_in -> ... d_out",
)
```

## sigmoid

逐元素非线性

torch.sigmoid(x)

## * 逐个元素相乘

## @ 矩阵乘法


SwiGLU起到的作用就是充当了FFN子层

(区分一下，SiLU是激活函数，SwiGLU是FFN子层)

# 完整代码

```py

class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model:int ,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        
        self.w1 = Linear(d_model,d_ff,device,dtype)
        self.w2 = Linear(d_ff,d_model,device,dtype)
        self.w3 = Linear(d_model,d_ff,device,dtype)
        
        
    def forward(
        self,
        in_features: torch.Tensor
    ) -> torch.Tensor:
        a = self.w1(in_features)
        b = self.w3(in_features)
        silu_a = a * torch.sigmoid(a)
        h = silu_a * b
        y = self.w2(h)
        
        return y
```

adapter适配略

测试通过，略

