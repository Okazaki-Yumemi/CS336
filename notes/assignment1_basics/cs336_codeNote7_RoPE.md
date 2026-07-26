# RoPE（Rotary Positional Embedding）实现笔记

## 1. RoPE 的作用

Self-Attention 本身只根据内容计算：

\[
QK^\top
\]

它并不知道 token 位于序列中的什么位置。

RoPE 的做法是：

> 根据 token 的位置，对 Query 和 Key 的 hidden features 两两分组，并进行二维旋转。

例如最后一维：

```text
[x0, x1, x2, x3, x4, x5]
```

被分成：

```text
(x0, x1)
(x2, x3)
(x4, x5)
```

每一对都作为一个二维向量，根据 token position 旋转不同的角度。

RoPE：

- 通常作用于 Query 和 Key；
- 不作用于 Value；
- 不改变 tensor shape；
- 没有可学习参数；
- 可以预计算 `sin` 和 `cos` 查找表。

作业要求输入 shape 为：

```text
(..., seq_len, d_k)
```

输出保持同样的 shape，并允许任意数量的前导 batch dimensions。RoPE 的 `sin/cos` 应作为 buffer，而不是 `nn.Parameter`。:contentReference[oaicite:0]{index=0}

---

## 2. 旋转公式

对于一对特征：

\[
(x_{\text{even}},x_{\text{odd}})
\]

旋转角度为 \(\phi\) 时：

\[
y_{\text{even}}
=
x_{\text{even}}\cos\phi
-
x_{\text{odd}}\sin\phi
\]

\[
y_{\text{odd}}
=
x_{\text{even}}\sin\phi
+
x_{\text{odd}}\cos\phi
\]

即：

\[
\begin{bmatrix}
y_{\text{even}}\\
y_{\text{odd}}
\end{bmatrix}
=
\begin{bmatrix}
\cos\phi & -\sin\phi\\
\sin\phi & \cos\phi
\end{bmatrix}
\begin{bmatrix}
x_{\text{even}}\\
x_{\text{odd}}
\end{bmatrix}
\]

---

## 3. 不同维度对的旋转频率

令：

\[
j=0,1,\ldots,\frac{d_k}{2}-1
\]

第 \(j\) 对特征的逆频率为：

\[
\omega_j
=
\Theta^{-2j/d_k}
\]

token position 为 \(p\) 时，旋转角度为：

\[
\phi_{p,j}
=
p\omega_j
\]

代码中：

```python
dimension_indices = [0, 2, 4, ..., d_k - 2]

exponents = dimension_indices / d_k

inv_freq = theta ** (-exponents)
```

例如：

```text
d_k = 8
theta = 10000
```

则：

```text
dimension_indices = [0, 2, 4, 6]
exponents         = [0, 0.25, 0.5, 0.75]
inv_freq          = [1, 0.1, 0.01, 0.001]
```

---

## 4. Cache 的 shape

位置向量：

```text
positions.shape = (max_seq_len, 1)
```

逆频率：

```text
inv_freq.shape = (d_k / 2,)
```

通过 broadcasting：

```text
(max_seq_len, 1)
×
(d_k / 2,)
----------------
(max_seq_len, d_k / 2)
```

得到：

```text
angles.shape    = (max_seq_len, d_k / 2)
cos_cache.shape = (max_seq_len, d_k / 2)
sin_cache.shape = (max_seq_len, d_k / 2)
```

每一行对应一个 token position，每一列对应一对 hidden features。

---

## 5. RoPE 参考实现

```python
import torch
from torch import nn


class RoPE(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()

        # hidden features 要两两配对。
        assert d_k % 2 == 0, "d_k must be even for RoPE"

        # [0, 2, 4, ..., d_k - 2]
        # 每一个元素对应一对二维特征。
        dimension_indices = torch.arange(
            start=0,
            end=d_k,
            step=2,
            device=device,
            dtype=torch.float32,
        )

        # [0/d_k, 2/d_k, 4/d_k, ...]
        exponents = dimension_indices / d_k

        # theta^(-2j/d_k)
        # shape: (d_k / 2,)
        inv_freq = torch.pow(theta, -exponents)

        # [0, 1, 2, ..., max_seq_len - 1]
        # 使用 [:, None] 变成列向量。
        # shape: (max_seq_len, 1)
        positions = torch.arange(
            max_seq_len,
            device=device,
            dtype=torch.float32,
        )[:, None]

        # Broadcasting:
        # (max_seq_len, 1) * (d_k / 2,)
        # -> (max_seq_len, d_k / 2)
        angles = positions * inv_freq

        cos_cache = torch.cos(angles)
        sin_cache = torch.sin(angles)

        # RoPE 没有可学习参数。
        # buffer 会跟随 model.to(device) 移动，
        # 但不会被 optimizer 更新。
        self.register_buffer(
            "cos_cache",
            cos_cache,
            persistent=False,
        )

        self.register_buffer(
            "sin_cache",
            sin_cache,
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:
                Shape (..., seq_len, d_k).

            token_positions:
                Shape (..., seq_len).
                指明 x 中每个 token 对应的实际位置。

        Returns:
            Shape (..., seq_len, d_k).
        """

        # 根据 token position 查找对应的旋转角度。
        #
        # cos/sin shape:
        # (..., seq_len, d_k / 2)
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]

        # cache 一般以 float32 保存。
        # 转成 x 的 dtype，避免输出被自动提升为 float32。
        cos = cos.to(dtype=x.dtype)
        sin = sin.to(dtype=x.dtype)

        # 拆分每个二维特征对：
        #
        # x_even = [x0, x2, x4, ...]
        # x_odd  = [x1, x3, x5, ...]
        #
        # shape:
        # (..., seq_len, d_k / 2)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        # 对每一对特征执行二维旋转。
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        # stack 后：
        #
        # (..., seq_len, d_k / 2, 2)
        #
        # 最后两个维度中的顺序是：
        # [[even_0, odd_0],
        #  [even_1, odd_1],
        #  ...]
        rotated_pairs = torch.stack(
            (rotated_even, rotated_odd),
            dim=-1,
        )

        # 将 (d_k / 2, 2) 合并回 d_k：
        #
        # (..., seq_len, d_k / 2, 2)
        # ->
        # (..., seq_len, d_k)
        output = rotated_pairs.flatten(start_dim=-2)

        return output
```

---

## 6. `register_buffer` 的正确用法

错误写法：

```python
self.cos_cache = torch.cos(angles)

self.register_buffer(
    "cos_cache",
    self.cos_cache,
    persistent=False,
)
```

这里在注册前已经存在同名属性，可能报：

```text
KeyError: attribute 'cos_cache' already exists
```

正确写法：

```python
cos_cache = torch.cos(angles)

self.register_buffer(
    "cos_cache",
    cos_cache,
    persistent=False,
)
```

`register_buffer` 执行后会自动创建：

```python
self.cos_cache
```

---

## 7. 为什么不是 `nn.Parameter`

以下内容是固定数学规则：

```text
inv_freq
cos_cache
sin_cache
```

它们不需要通过训练学习，因此不能写成：

```python
self.cos_cache = nn.Parameter(cos_cache)
```

否则优化器会把它们当成模型参数。

使用 buffer 后：

```python
rope.parameters()
```

不会包含 `cos_cache` 和 `sin_cache`，但：

```python
rope.to("cuda")
```

会把它们自动移动到 GPU。

---

## 8. 为什么使用 `token_positions` 查表

不能总是假设当前 token 的位置是：

```text
0, 1, 2, ..., seq_len - 1
```

例如自回归生成时，KV cache 中已经有 100 个 token，新输入可能只有一个 token：

```text
x.shape = (batch, 1, d_k)
```

但这个 token 的实际位置是：

```text
100
```

所以需要：

```python
cos = self.cos_cache[token_positions]
sin = self.sin_cache[token_positions]
```

而不是：

```python
cos = self.cos_cache[:seq_len]
sin = self.sin_cache[:seq_len]
```

---

## 9. `stack + flatten` 为什么能恢复交错顺序

假设：

```text
x_even = [x0, x2, x4]
x_odd  = [x1, x3, x5]
```

旋转后：

```text
rotated_even = [y0, y2, y4]
rotated_odd  = [y1, y3, y5]
```

执行：

```python
torch.stack(
    (rotated_even, rotated_odd),
    dim=-1,
)
```

得到：

```text
[[y0, y1],
 [y2, y3],
 [y4, y5]]
```

再执行：

```python
flatten(start_dim=-2)
```

得到：

```text
[y0, y1, y2, y3, y4, y5]
```

因此恢复了正确的偶数、奇数交错顺序。

不能直接写：

```python
torch.cat((rotated_even, rotated_odd), dim=-1)
```

因为它会得到：

```text
[y0, y2, y4, y1, y3, y5]
```

顺序不正确。

---

## 10. Shape 总结

初始化阶段：

```text
dimension_indices   (d_k / 2,)
exponents           (d_k / 2,)
inv_freq            (d_k / 2,)
positions           (max_seq_len, 1)
angles              (max_seq_len, d_k / 2)
cos_cache           (max_seq_len, d_k / 2)
sin_cache           (max_seq_len, d_k / 2)
```

Forward 阶段：

```text
x                   (..., seq_len, d_k)
token_positions     (..., seq_len)

cos                 (..., seq_len, d_k / 2)
sin                 (..., seq_len, d_k / 2)

x_even              (..., seq_len, d_k / 2)
x_odd               (..., seq_len, d_k / 2)

rotated_pairs       (..., seq_len, d_k / 2, 2)
output              (..., seq_len, d_k)
```

---

## 11. 最小正确性检查

### 检查一：position 0 不应改变输入

因为：

\[
\cos(0)=1,\qquad \sin(0)=0
\]

所以位置 0 的旋转应为恒等变换。

```python
rope = RoPE(
    theta=10000.0,
    d_k=4,
    max_seq_len=8,
)

x = torch.randn(2, 1, 4)
positions = torch.zeros(
    2,
    1,
    dtype=torch.long,
)

output = rope(x, positions)

torch.testing.assert_close(output, x)
```

### 检查二：旋转保持每个二维向量的长度

二维旋转不会改变：

\[
x_{\text{even}}^2+x_{\text{odd}}^2
\]

测试：

```python
rope = RoPE(
    theta=10000.0,
    d_k=8,
    max_seq_len=16,
)

x = torch.randn(2, 5, 8)

positions = torch.arange(5).expand(2, 5)

output = rope(x, positions)

x_pairs = x.reshape(2, 5, 4, 2)
output_pairs = output.reshape(2, 5, 4, 2)

input_squared_norm = x_pairs.pow(2).sum(dim=-1)
output_squared_norm = output_pairs.pow(2).sum(dim=-1)

torch.testing.assert_close(
    input_squared_norm,
    output_squared_norm,
)
```

---

## 12. 常见错误

### 错误一：把 cache 设成 `nn.Parameter`

RoPE 的三角函数值固定，不需要学习。

### 错误二：注册 buffer 前创建同名属性

不要同时写：

```python
self.cos_cache = ...
self.register_buffer("cos_cache", ...)
```

### 错误三：最后一维不是偶数

RoPE 要把特征两两配对，因此：

```python
assert d_k % 2 == 0
```

### 错误四：直接用 `cat` 拼接

`cat` 会把所有 even 放前面、所有 odd 放后面，无法恢复交错顺序。

应使用：

```python
torch.stack(..., dim=-1).flatten(-2)
```

### 错误五：忽略 `token_positions`

自回归生成或 KV cache 场景中，输入 tensor 内部的序号不一定等于真实 token position。

### 错误六：构造完整的旋转矩阵

不需要构造：

```text
(max_seq_len, d_k, d_k)
```

只需存：

```text
cos_cache: (max_seq_len, d_k / 2)
sin_cache: (max_seq_len, d_k / 2)
```

---

## 13. 一句话记忆

```text
初始化：
位置 × 频率 → angles → sin/cos cache

Forward：
按位置查 cache → 拆 even/odd → 二维旋转 → 交错拼回
```