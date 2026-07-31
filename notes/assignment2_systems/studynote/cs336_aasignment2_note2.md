# This note is for cs336 Assignment 2, Part 3. Single GPU memory

The most common way to speed up training is gradient checkpointing.

## 3.1 Autograd Residuals

In order to perform a backward pass, we need to save the activiations that were produced in the forward pass.

While this is obviously the case for some operations, by default it'll happen for many more than you might expect.

The tensors saved for the backward pass are called "residuals", or simply "saved tensors".


sample code:

```py

import torch
from torch import nn

x = torch.randn((4,512,2560), requires_grad=True)

class RMSNorm(nn.Module):
    def __init__(
      self,
      hidden_size: int,
      eps: float = 1e-5,
      device = None,
    ):
      super().__init__()
      self.weight = nn.Parameter(torch.ones(hidden_size, device=device))
      self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
      # compute the mean square of the input tensor along the last dimension
      rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
      x = x * rms
      return self.weight * x
    
def pack_hook(t):
    shape,dtype,grad_fn = t.shape,t.dtype,t.grad_fn
    print(f"Loading residual: {shape=}, {dtype=}, {grad_fn=}")
    return t

def unpack_hook(t):
    shape,dtype,grad_fn = t.shape,t.dtype,t.grad_fn
    print(f"Loading residual: {shape=}, {dtype=}, {grad_fn=}")
    return t
  
```

```bash

$ uv run scripts/autograd_experiment.py
Saving residual: shape=torch.Size([4, 512, 2560]), dtype=torch.float32, grad_fn=None
Saving residual: shape=torch.Size([4, 512, 1]), dtype=torch.float32, grad_fn=<RsqrtBackward0 
object at 0x7f7dd319b5e0>
Saving residual: shape=torch.Size([4, 512, 1]), dtype=torch.float32, grad_fn=<RsqrtBackward0 
object at 0x7f7dd319b5e0>
Saving residual: shape=torch.Size([4, 512, 2560]), dtype=torch.float32, grad_fn=None
Saving residual: shape=torch.Size([4, 512, 2560]), dtype=torch.float32, grad_fn=<MulBackward0 
object at 0x7f7dd319b5e0>
Saving residual: shape=torch.Size([2560]), dtype=torch.float32, grad_fn=None

Loading residual: shape=torch.Size([4, 512, 2560]), dtype=torch.float32, 
grad_fn=<MulBackward0 object at 0x7f7cf14e6740>
Loading residual: shape=torch.Size([2560]), dtype=torch.float32, grad_fn=None
Loading residual: shape=torch.Size([4, 512, 1]), dtype=torch.float32, grad_fn=<RsqrtBackward0 
object at 0x7f7cf14e6740>
Loading residual: shape=torch.Size([4, 512, 2560]), dtype=torch.float32, grad_fn=None
Loading residual: shape=torch.Size([4, 512, 1]), dtype=torch.float32, grad_fn=<RsqrtBackward0 
object at 0x7f7cf14e6740>
Loading residual: shape=torch.Size([4, 512, 2560]), dtype=torch.float32, grad_fn=None
```

### 3.1.1 Operator Fusion

We want a single op that takes in the RMSNorm weights and the activation, and spits out the output, as well as for that operation to be unitary in the backward pass.

This is one motivation for kernel fusion. Since the RMSNorm is fairly well behaved,we can even automatically fuse it using torch.compile.

```py

ln = torch.compile(RMSNorm(x.shape[-1], device=x.device))

with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y = ln(x)
    y.sum().backward()
```
The new output is significantly better:

```bash
Saving residual: shape=torch.Size([4, 512, 2560]), dtype=torch.float32, grad_fn=None
Saving residual: shape=torch.Size([2560]), dtype=torch.float32, grad_fn=None
Saving residual: shape=torch.Size([4, 512, 1]), dtype=torch.float32, grad_fn=None
Loading residual: shape=torch.Size([4, 512, 2560]), dtype=torch.float32, grad_fn=None
Loading residual: shape=torch.Size([2560]), dtype=torch.float32, grad_fn=None
Loading residual: shape=torch.Size([4, 512, 1]), dtype=torch.float32, grad_fn=None
```

We only need to save a single full-size activation tensor for the backward pass —— namely, the input to the RMSNorm function.

Notice also how the order of loading is no longer the reverse of saving, and each residual no longer has a grad_fn dependency. —— PyTorch is treating the entirety of our RMSNorm as a single function.

## 3.2 Activation Checkpointing

While fusion is undoubtedly useful, it can only get us so far in saving memory. For instance, let's fuse a single TransformerBlock at size xl.

```py

import torch
from cs336_basics.model import RotaryEmbedding, TransformerBlock


# num_layers for this model is 32
d_model,d_ff,num_heads,context_length = 2560,10240,16,2048
block = TransformerBlock(
  d_model=d_model,
  d_ff=d_ff,
  num_heads=num_heads,
)
positional_encoder = RotaryEmbedding(dim = d_model // num_heads, context_length = context_length)

# Fuse as much torch.compile will allow
block = torch.compile(block, fullgraph = True)
x = torch.randn((4,context_length,d_model), requires_grad=True)

...

# Niw logs the number of bytes saved
total_size_bytes = 0
def pack_hook(t):
    if isinstance(t, torch.nn.Parameter):
        return t
    global total_size_bytes

    shape, dtype , grad_fn = t.shape, t.dtype, t.grad_fn
    total_size_bytes += t.numel() * t.element_size()
    print(f"Saving residual: {shape=}, {dtype=}, {grad_fn=}")
    return t

...

# Run a forward pass, saving for backward

with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y = block(x, positional_encoder)

print(f"Total size of saved tensors in single TransformerBlock: {total_size_bytes / (1024*1024):.2f} MiB")
```


```bash
...
Total size of saved tensors in single TransformerBlock: 3651.31 MiB
```

3.6 GiB for every layer. If we do this for all layers, we get 114 GiB of activations, just saved for backward! 

There's a nontrivial amount of waste in the attention operation's residuals, which we will fix in Section 4. But even with this fix, the memory use will grow linearly with batch size,sequence length and embedding size.

### 3.2.1 Recomputation

Instead of holding on to every tensor we generate, it's possible to save only periodic checkpoints of our results, and recompute the values in-between. PyTorch has an interface we can call to handle this in a simple fashion. `torch.utils.checkpoint.checkpoint` takes in a function, and arguments to that function.

It then modifies the behavior of the function passed by:

1. In the forward pass:
  1. Saving the input values to the function
  2. Suppressing the saving of tensors in the forward pass
2. In the backward pass:
  1. Prepending a recomputation step where the forward pass is recomputed from the previously saved inputs, and values are saved for backward.
  2. The backward pass is run and all tensors can be freed.

In the simple case of running through 4 transformer blocks, we see that our memory adds up as we would expect.

```py

...

def four_blocks(x):
    x = block(x)
    x = block(x)
    x = block(x)
    x = block(x)
    return x
  
with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y = four_blocks(x)

print(f"Total size of saved tensors in four TransformerBlocks: {total_size_bytes / (1024*1024):.2f} MiB")
```

```bash
Total size of saved tensors in four TransformerBlocks: 14605.25 MiB
```

But we can employ gradient checkpointing as follows:

```py

from torch.utils.checkpoint import checkpoint
def two_blocks(x):
    x = block(x)
    x = block(x)
    return x
  
def four_blocks_checkpoint(x):

    # Checkpoint throws out all the saved tensors until the backward pass
    # When getting to the checkpointed block in the backward pass,
    # it reruns a forward pass to produce the saved tensors
    # then completes normal backward pass.
    x = checkpoint(two_blocks, x)
    x = checkpoint(two_blocks, x)
    return x
  
with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y = four_blocks_checkpoint(x)
  
print(f"Total size of saved tensors in four TransformerBlocks with checkpointing: {total_size_bytes / (1024*1024):.2f} MiB")
```

```bash
Saving residual: shape=torch.Size([0]), dtype=torch.float32, grad_fn=None
Saving residual: shape=torch.Size([4, 2048, 2560]), dtype=torch.float32, grad_fn=None
Saving residual: shape=torch.Size([0]), dtype=torch.float32, grad_fn=None
Saving residual: shape=torch.Size([4, 2048, 2560]), dtype=torch.float32, 
grad_fn=<torch.autograd.function.CompiledFunctionBackward object at 0x7aa0657a19d0>
Total size of saved tensors in four TransformerBlocks with checkpointing: 160.00 MiB
```
Keep in mind this hasn’t eliminated the memory use. 

Rather, it's factored our memory use into two categories:
1. The longer term storage we save to prepare for recomputation at the entry point of each checkpoint call
2. The short term memory generated in the recomputation pass within the checkpointed block to facilitate a backward pass through it.

**Problem: Gradient_checkpointing**:
Consider a Transformer with 𝑁 identical blocks stacked sequentially. Without any checkpointing, all 𝑁 blocks’ worth of residuals are kept alive simultaneously, giving 𝑂(𝑁) peak activation 
memory. We have a free hand to wrap any subset of the forward pass in checkpoint, including nesting checkpoint calls inside one another.

(a) What checkpointing strategy minimizes peak activation memory? Ignoring the compute cost?
>平衡递归 checkpointing：先把模型分成前后两半并分别 checkpoint，再在每一半内部继续二分和嵌套 checkpoint，直到单个 TransformerBlock。
>平衡二分树深度为 O(logN)。在 backward 的某一时刻，只需要保存递归路径上的 O(logN) 个 checkpoint 输入，并物化一个叶子区域的 residual

```
checkpoint(整个模型)
    checkpoint(前半段)
        checkpoint(前半段的前半段)
        checkpoint(前半段的后半段)
    checkpoint(后半段)
        ...
```



(b) consider the xl model config with batch size 4 and sequence length 2048 as above. If you only have the time/compute budget to run one step of recomputation,what is the best checkpointing strategy to reduce peakmemory?

| 每个 checkpoint 包含层数 (k) |                                估算峰值激活显存 |
| ---------------------: | --------------------------------------: |
|                      1 |         ($32\times80+3651\approx6.1$) GiB |
|                      2 |  ($16\times80+2\times3651\approx8.4$) GiB |
|                      4 |  ($8\times80+4\times3651\approx14.9$) GiB |
|                     16 | ($2\times80+16\times3651\approx57.2$) GiB |

```
x = checkpoint(block_1, x)
x = checkpoint(block_2, x)
...
x = checkpoint(block_N, x)
```


> With only one level of recomputation, I would checkpoint every TransformerBlock individually. A checkpoint input has shape [4,2048,2560] and occupies only 80 MiB in FP32, while the saved residuals for one block occupy approximately 3651 MiB. For a chunk size k, peak activation memory is approximately 80⌈32/k⌉+3651k MiB, which is minimized at the boundary k=1. Therefore, unlike the equal-cost square-root heuristic, this model benefits from the smallest possible checkpointed block because recomputed block residuals are much larger than checkpoint tensors.