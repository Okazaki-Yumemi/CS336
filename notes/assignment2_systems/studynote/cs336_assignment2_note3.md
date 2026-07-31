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