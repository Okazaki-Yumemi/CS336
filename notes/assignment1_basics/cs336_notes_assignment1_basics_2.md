# 此笔记为Transformer架构知识笔记

## 3.1 Transformer LM

**An overview of Transformer language model**

```

inputs
->  Token embedding
-> Transformer block
-> Transformer block
-> ...
-> Transformer block
-> Norm
-> Linear
-> Softmax
-> Output probabilities
```
**A pre-norm Transformer block**
```
inputs
 |_______
 |      norm 
 |      |
 |      causal multi-head self-attention
 |      |
 |______|
 |
 add
 |
 |_______
 |      |
 |     norm
 |      |
 |    Position-wise feed-forward network
 |      |
 |______|
 |
  add
```


**Token Embedding**: 将输入的token映射到一个高维空间中，通常使用一个可训练的嵌入矩阵。

takes in a tensor of shape (batch_size, sequence_length) and outputs a tensor of shape (batch_size, sequence_length, d_model).

**Pre-norm Transformer Block**:  A standard decoder-only Transformer language model consists of `num_layer` identical layers(commonly called `blocks`)

## 3.2 Remark: Batching , Einsum , and Efficient Computation
Transformer架构中我们会对输入数据进行批处理（batching），以提高计算效率。批处理允许我们同时处理多个输入序列，从而充分利用硬件资源。

- Element of a batch: 应用相同的forward
- Sequence of length: position-wise operations like RMSNorm 和 feed-forward
- Attention heads: the attention operation is batched across attention heads in a multi-head attention mechanism.


**Example of einsum**:  `einsum` 是一种强大的张量操作工具，它允许我们使用爱因斯坦求和约定来表示复杂的张量操作。通过使用 `einsum`，我们可以更简洁地表达矩阵乘法、张量收缩等操作，从而提高代码的可读性和计算效率。


```py

import torch
from einops import rearrange , einsum

## Basic implementation

Y = D @ A.T
# hard to tell the input and output shapes and what they mean
# What shapes can D and A have?and do any of these have unexpected shapes?


# Einsum is self-documenting and robust

Y = einsum(D, A, "batch sequence d_in , d_out d_in -> batch sequence d_out")

# Or , a batched version where D can have any leading dimensions but A is constrained
Y = einsum(D, A, "... d_in , d_out d_in -> ... d_out")

```


```py

images = torch.randn(64,128,128,3) # (batch, height, width, channels)
dim_by = torch.linspace(start = 0.0 , end = 1.0 , steps = 10)

## Reshape and multiply
dim_value = rearrange(dim_by, "dim_value -> 1 dim_value 1 1 1") # (1, dim_value, 1, 1 1)
images_rearr = rearrange(images, "b height width channel -> b 1 height width channel") # (b, 1, height, width, channel)

dimmed_images = images_rearr * dim_value # (b, dim_value, height, width, channel)

# or i one go

dimmed_images = einsum(
    images, dim_by,
    "batch height width channel , dim_value -> batch dim_value height width channel"
)
```

**Example 3 pixel mixing**:  `einsum` 可以用于实现像素混合操作，例如在图像处理中对不同通道的像素值进行加权平均。通过使用 `einsum`，我们可以方便地对图像张量进行操作，从而实现各种图像处理效果。

```py

channels_last = torch.randn(64,32,32,3) # (batch, height, width, channels)
B = torch.randn(32*32,32*32) # (height*width, height*width)

## Rearrange an image tensor for mixing across all pixels

channels_last_flat = channels_last.view(
    -1, channels_last.size(1)*channels_last.size(2), channels_last.size(3)
)
channels_first_flat = channels_last_flat.transpose(1,2)
channels_first_flat_transformed = channels_first_flat @ B.T
channels_last_flat_transformed = channels_first_flat_transformed.transpose(1,2)
channels_last_transformed = channels_last_flat_transformed.view(*channels_last.shape)

# Instead, using einops

height = width = 32

channels_first = rearrange(
    channels_last, "batch height width channel -> batch channel (height width)"
)

channels_first_transformed = einsum(
    channels_first, B,
    "batch channel pixel_in ,pixel_out pixel_in -> batch channel pixel_out"
)

channels_last_transformed = rearrange(
    channels_first_transformed, "batch channel (height width) -> batch height width channel", height = height, width = width
)

# or all in one go using einx.dot

height = width = 32

channels_last_transformed = einx.dot(
    "batch row_in col_in channel , (row_out col_out) (row_in col_in)"
    "-> batch row_out col_out channel",
    channels_last, B, col_in = height, col_out = width
)
```

## 3.3 Basic Building Blocks: Linear and Embedding Modules.

### 3.3.1 Parameter initialization

- Linear weights:  𝒩︀(𝜇 = 0,𝜎2 = 2 / (𝑑in+𝑑out)) truncated at [−3𝜎,3𝜎]
- Embedding:  𝒩︀(𝜇 = 0,𝜎2 = 1) truncated at [−3,3]
- RMSNorm:  1

### 3.3.2 Linear Module


开始写代码咯，写Linear module. 见代码笔记.

`cs336_codeNote3_linearModule.md`



### 3.3.3 Embedding Module

Transformer的第一层是一个embedding layer. 把 integer token IDs 映射到 d_model 维度的向量

我们要写一个Embedding class, 继承nn.Embedding module

forward method要select the embedding vector for each token ID by indexing into an embedding matrix of shape (vocab_size , d_model) using a torch.LongTensor of token IDs with shape (batch_size , sequence_length)

开始写代码咯，写Embedding module. 见代码笔记.

`cs336_codeNote4_embedding.md`


## 3.4 Pre-Norm Transformer Block

Each Transformer block has two sub-layers: a multi-head self-attention mechanism and a position-wise feed-forward network

人们发现Pre Norm强多了

### 3.4.1 Root Mean Square Layer Normalization (RMSNorm)

You should upcast your input to torch.float32 to prevent overflow when you square the input. Overall, 
your forward method should look like:

```py

in_dtype = x.dtype

x = x.to(torch.float32)

# your code here performing RMSNorm

...
result = ...

return result.to(in_dtype)

```

开始写代码咯，写RMSNorm module. 见代码笔记.

`cs336_codeNote5_RMSnorm.md`

### 3.4.2 position-wise Feed-Forward Network (FFN)

- 第一个点是我们已经不用ReLU了，而是用SwiGLU了

Putting the SiLU/Swish and GLU together, we get SwiGLU. The formula is:

```
FFN(x) = SwiGLU(x,W1,W2,W3) = W2(SiLU(W1x) * W3x)
```

写代码了，见
`cs336_codeNote6_SwiGLU.md`

### 3.4.3 Relative Positional Embeddings

RoPE是位置编码，每个小矩阵是 cos/sin的旋转矩阵，旋转角度是相对位置编码

然后一整个大的R矩阵就是对交线是 2x2 的小矩阵的block diagonal matrix，别的都是0

代码见
`cs336_codeNote7_RoPE.md`

### 3.4.4 Scaled Dot-Product Attention

As a preliminary step, the definition of the Attention operation will make use of softmax,an operation that takes an unnormalized vector of scores and turns it into normalized distribution.

softmax is defined as follows:

```
softmax(x)_i = exp(x_i) / sum_j exp(x_j)
```

代码见
`cs336_codeNote8_softmax.md`



现在,我们可以开始写attention了

$$ Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V $$

Q 属于 nxd_k, K 属于 mx d_k, V 属于 mx d_v

**masking** 


``scaled dot-product attention``

见 `cs336_codeNote9_scaledDotProductAttention.md`
