# 4.GPU kernels

## 4.1 Optimizing Attention with FlashAttention-2

### 4.1.1 Benchmarking PyTorch Attention

The naive attention implementation needs to save attention score matrices of shape `seq_len x seq_len` for each batch/head element, which can grow very large with long sequence lengths, causing out-of-memory errors for any tasks with long inputs or outputs.

We will implement an attention kernel following the FlashAttention-2 paper,which computes attention by tiles and avoids ever explicitly materializing the `seq_len x seq_len` attention score matrices,enabling scaling to much longer sequence lengths.

**Problem Pytorch_attention benchmarking**:

(a) Benchmark your attention implementation at different scales. Write a script that will:
  - Fix the batch size to 8 and don't use multihead attention.
  - Iterate through the cartesian product of [16,32,64,128] for the head embedding dimension d_model , and [256 , 1024, 4096, 8192, 16384] for sequence length
  - Create random inputs Q , K , V for appropriate size
  - Time 100 forward passes through attention using the inputs
  - measure how much memory is in use before the backward pass starts, and time 100 backwards.
  - make sure to warmup,and to call sync after each forward/backward pass to get accurate timing.

实现见
`cs336_assignment2_codenote5_attnbench.md`


## 4.2 Benchmarking JIT-Compiled Attention

Since version 2.0, Pytorch also ships with a powerful just-in-time compiler that automatically tries to apply a number of optimizations to PyTorch functions.

In particular , it will try to automatically generate fused Triton kernels by dynamically analyzing your computaiton graph.

The interface for using the pytorch compiler is very simple. For instance , if we wanted to apply it to a single layer of our model:

just:

```py

layer = SomePyTorchModel(...)
compiled_layer = torch.compile(layer)
```

Now,compiled_layer functionally behaves just like layer. 
We can also compile our entire PyTorch model with torch.compile(model) , or even a Python function that calls PyTorch operations.

**Problem: Torch Compile**


(a) Extend your attention benchmarking script to include a compiled version of your PyTorch implementation of attention, and compare its performance to the uncompiled version with the same configuration as the pytorch_attention problem above

(b) Now, compile your entire Transformer model in your end-to-end benchmarking script. How does the performance of the forward pass change? What about the combined forward and backward passes and optimizer steps?

见
`cs336_assignment2_codenote6_compilebench.md`

