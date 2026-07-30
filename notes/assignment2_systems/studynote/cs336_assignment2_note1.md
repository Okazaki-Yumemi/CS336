# 2. Profiling and Benchmarking

explore how to optimize the performance of our Transformer model to make the most efficient use of the GPU.

## 2.1 Profiling

it's helpful to first profile our program to understand where it spends resources(e.g., time and memory). Otherwise,we risk optimizing parts of the model that don't account for significant time or memory,and therefore not seeing measuerable end-to-end improvements.

We will implement 3 performance evaluation paths.

1. Simple end to end benchmarking using the python standard library to time our forward and backward passes.
2. Compute profiling with the NVIDIA Nsight systems tool to understand how that time is distributed across operations on both the CPU and GPU.
3. Memory profiling

### 2.1.3 End - to -end Benchmarking

For benchmarking GPU code,one caveat is that CUDA calls are asynchronous.

When you call a CUDA kernel,such as when you invoke torch.matmul, the PyTorch function call returns control to your code without waiting for the matrix multiplication to finish.

In PyTorch,we can call torch.cuda.synchronize() to wait for all scheduled GPU kernels to complete, allowing us to get more accurate measurements of CUDA kernel runtime. The synchronnization in this operation refers to synchronizing the CPU runtime with the GPU runtime.

**Problem_benchmarking_script**:

(A) Write a script to perform basic end-to-end benchmarking of the forward pass, backward pass, and optimizer step in your model.

Specifically, your script should support the following:

- Given Hyperparameters (e.g., number of layers), initialize a model.
- Generate a random batch of data.
- Run w warm-up steps (before you start measuring time), then time the execution of n steps.(either only forward, forward and backward, or forward and backward with optimizer step, depending on an argument). For timing, you can use the Python timeit module (e.g., either using the timeit function,or using timeit.default_timer(),which gives you the system's highest resolution clock, thus a better default for benchmarking than time.time()).

- Call torch.cuda.synchronizer() after the step.

(B) Time the forward, backward, and optimizer step for the model sizes described in Section 2.1.2. Use 5 warmup steps and compute the average and standard deviation of timings over 10 measurement steps. How long does a forward pass take? How about a backward pass? Do you see high variability across measurements, or is the standard deviation small?

(C) One caveat of benchmarking is not performing the warm-up steps. Repeat your analysis without the warm-up steps. How does this affect your results? Why do you think this happens? Also try to run the script with 1 or 2 warm-up steps. Why might the result still be different?

实现见
`cs336_assignment2_codenote1_benchmarking.py`

### 2.1.4 Nsight Systems Profiler

To know how much time our program spends in each component(e.g. function) , we can use a profiler.

An execution profiler instruments the code by inserting guards when functions begin and finish running, and thus can give detailed execution stastics at the function level.

Nvidia ships a profiler that we can use via the CLI nsys.  

Using nsys is straightforward: run your python script from the previous sctions with nsys profile prepended.

> $ uv run nsys profile -- python benchmark.py

a more comprehensive profiling run may look like
> $ uv run nsys profile --trace=cuda,cudnn,cublas,osrt,nvtx --pytorch=functions-trace,autograd-shapes-nvtx --cudabacktrace=all --python-backtrace=cuda --gpu-metrics-devices=0 --python benchmark.py

**Problem nsys_profile**

profile your forward pass , backward pass and optimizer step using nsys with tow model size from tabel 1 of your choice as well as three power of two context length larger thant 128,where the largest available size should be the longest context length you can fit in memory. Pick the combinations you think would be the most interesting to look at. For each profile answer the following questions:

(a) what is the total time spent on your forward pass?
(b) what cuda kernel takes the most cumulative GPU time during the forward pass? how many times this kernel is invoked during a single forward pass?
(c) Although the vast majority of FLOPs take place in matrix multiplications, you will notice 
that several other kernels still take a non-trivial amount of the overall runtime. What other 
kernels besides matrix multiplies do you see accounting for non-trivial CUDA runtime in the 
forward pass?
(d)  Profile running one complete training step with your implementation of AdamW.  How does the fraction of time spent on matrix multiplication change, compared to doing inference (forward pass only)? How about other kernels?
(e) Compare the runtime of the softmax operation versus the matrix multiplication operations 
within the self-attention layer of your model during a forward pass. How does the difference 
in runtimes compare to the difference in FLOPs?

解答见
`cs336_assignment2_codenote2_nsys.md`

### 2.1.5 Mixed Precision

hybrid precision training is good. 

For ex.  B200 AT FP32 is 80 TFLOPS, but for FP16 or BF16, it is 2500 TFLOPS.  This is a 30x speedup.  So we want to use mixed precision training to speed up our training.

**Problem mixed_precision_accumulation**

Run the following code and comment on the accuracy

```py

s = torch.tensor(0, dtype=torch.float32, device="cuda")
for i in range(1000):
    s += torch.tensor(0.01, dtype = torch.float32)
print(s)

s = torch.tensor(0, dtype=torch.float16, device="cuda")
for i in range(1000):
    s += torch.tensor(0.01, dtype = torch.float16)
print(s)

s = torch.tensor(0, dtype=torch.float32, device="cuda")
for i in range(1000):
    s += torch.tensor(0.01, dtype = torch.float16)
print(s)

s = torch.tensor(0,dtype=torch.float32)
for i in range(1000):
  x = torch.tensor(0.01,dtype=torch.float16)
  s += x.type(torch.float32)
print(s)
```
输出是
```
FP32 加数 + FP32 累加器：10.0001335
FP16 加数 + FP16 累加器： 9.953125
FP16 加数 + FP32 累加器：10.0021362
显式转 FP32 后累加：      10.0021362
```
区别来自于
1. 表示误差： FP16 不能精确表示 0.01
2. 累加误差： FP16 累加器每一次都要舍入，1000次后误差累计明显




**Problem  benchmarking_mixed_precision**
(a) consider following model

```py

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features,10,bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10,out_features,bias=False)
        self.relu = nn.ReLU()
    
def forward(self, x):
    x = self.relu(self.fc1(x))
    x = self.ln(x)
    x = self.fc2(x)
    return x
```
Suppose we are training the model on a GPU and that the model parameters are originally in 
FP32. We’d like to use autocasting mixed precision with FP16. What are the data types of:
- the model parameters within the autocast context?
- the output of the first feed-forward layer (ToyModel.fc1)?
- the output of layer norm (ToyModel.ln)?
- the model’s predicted logits?
- the loss?
- the model’s gradients?

| 部分                   | dtype           |
| -------------------- | --------------- |
| 模型参数                 | `torch.float32` |
| `fc1` 输出             | `torch.float16` |
| `LayerNorm` 输出       | `torch.float32` |
| 最终 logits，即 `fc2` 输出 | `torch.float16` |
| loss                 | `torch.float32` |
| 参数梯度                 | `torch.float32` |

原因是 autocast 不会永久修改参数 dtype。linear 被列为适合 FP16 的算子，而 layer_norm、log_softmax、cross_entropy 等数值敏感算子会以 FP32 执行。参数仍是 FP32 leaf tensor，所以最终保存在 parameter.grad 中的梯度也是 FP32.

```
FP32 input
  ↓ fc1
FP16
  ↓ ReLU
FP16
  ↓ LayerNorm
FP32
  ↓ fc2
FP16 logits
  ↓ cross entropy
FP32 loss
  ↓ backward
FP32 parameter gradients
```

(b) You should have seen that FP16 mixed precision autocasting treats the layer normalization 
layer differently than the feed-forward layers. What parts of layer normalization are sensitive 
to mixed precision? If we use BF16 instead of FP16, do we still need to treat layer 
normalization differently? Why or why not?

Layer norm 的核心过程是
```
计算均值
计算方差
减去均值
除以 sqrt(方差 + epsilon)
```

其中均值和方差需要对大量元素进行 reduction。低精度累加会产生明显误差；计算方差时还有平方、相减和倒数平方根，可能出现溢出、下溢或数值不稳定。

> Layer normalization is sensitive to mixed precision because computing the mean and variance involves reductions, squaring, subtraction, and reciprocal square roots, all of which can accumulate numerical error. BF16 has a much wider dynamic range than FP16, reducing overflow and underflow, but its mantissa is still limited, so performing the normalization reductions in FP32 remains beneficial.

(c) Modify your benchmarking script to optionally run the model using mixed precision with 
BF16. Time the forward and backward passes with and without mixed-precision for each 
language model size described in 
Section 2.1.2. Compare the results of using full precision 
versus mixed precision, and comment on any trends as model size changes. You may find the 
nullcontext no-op context manager to be useful.

要改代码，见
`cs336_assignment2_codenote3_mixedprecision.md`


### 2.1.6 Profiling Memory

So far, we have been looking at compute performance. We’ll now shift our attention to memory, another 
major resource in language model training and inference. PyTorch also ships with a powerful memory 
profiler, which can keep track of allocations over time.

TO use the memory profiler, we can use the following code snippet:

```py

... # warm-up phase in your benchmarking script

# start recording memory history

torch.cuda.memory._record_memory_history(max_entries=1000000)

... # what you want to profile in your benchmarking script

# save a pickle file to be loaded by pytorch's online tool.

torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")

# stop recording histroy
torch.cuda.memory._record_memory_history(enabled = None)
```

把文件送去 pytorch.org/memory_viz，可视化。


**Problem memory_profiling**

(a) Add an option to your profiling script to run your model through the memory profiler

(b) What is the peak memory usage of each context length when doing a forward pass? What 
about when doing a full training step?

(c) Find the peak memory usage of the xl model when using mixed-precision, for both a forward 
pass and a full training step. Does mixed-precision significantly affect memory usage?

(d) Consider the xl model. Given our reference hyperparameters, what is the size of a tensor of 
activations in the Transformer residual stream, in single-precision? Give this size in MiB (i.e., 
divide the number of bytes by 10242)

(e) Now look closely at the “Active Memory Timeline” from pytorch.org/memory_viz of a 
memory snapshot of the xl model doing a forward pass. When you reduce the “Detail” level, 
the tool hides the smallest allocations to the corresponding level (e.g., putting “Detail” at 
10% only shows the 10% largest allocations). What is the size of the largest allocations 
shown? Looking through the stack trace, can you tell where those allocations come from?

(f) Nsight Systems also has flags for memory profiling. You can combine these with the Nsight 
flags from before to understand what allocations are happening at different steps in your 
model’s lifespan. Use the PyTorch-provided NVTX labels to determine how much memory is 
saved for backward (these tensors are often called residuals) by a single TransformerBlock in 
your model. Note the 5 largest contributing operations, and what percentage of the overall 
memory they contribute.

见 `cs336_assignment2_codenote4_memoryprofiling.md`