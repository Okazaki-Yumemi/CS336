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