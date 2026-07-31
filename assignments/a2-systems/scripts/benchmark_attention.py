from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
import timeit
import numpy
import gc

from cs336_basics.model import scaled_dot_product_attention as attention

D_MODEL = [16,32]
SEQUENCE_LENGTH = [256,1024,4096]
WARMUP_STEPS = 10
BATCH_SIZE = 8
NUM_HEADS = 1

def benchmark_attention(
    batch_size: int,
    seq_len: int,
    d_model: int,
    num_heads: int,
    warmup_steps: int,
    measurement_steps: int,
):
    
    Q = torch.randn(batch_size,
                    seq_len,
                    d_model,
                    requires_grad=True,
                    device='cuda',
                    dtype=torch.float32)
    K = torch.randn(batch_size,
                    seq_len,
                    d_model,
                    requires_grad=True,
                    device='cuda',
                    dtype=torch.float32)
    V = torch.randn(batch_size,
                    seq_len,
                    d_model,
                    requires_grad=True,
                    device='cuda',
                    dtype=torch.float32)

    for _ in range(warmup_steps):
        output = attention(Q,K,V)
    torch.cuda.synchronize()
        
        
    times : list[float] = []
    for _ in range(measurement_steps):
        torch.cuda.synchronize()
        start_time = timeit.default_timer()
        output = attention(Q,K,V)
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        times.append(end_time-start_time)
    
    mean = sum(times)/len(times)
    std = numpy.std(times)
    return mean,std

def backward_benchmark(
    batch_size: int,
    seq_len: int,
    d_model: int,
    num_heads: int,
    warmup_steps: int,
    measurement_steps: int,
):
    Q = torch.randn(batch_size,
                    seq_len,
                    d_model,
                    requires_grad=True,
                    device='cuda',
                    dtype=torch.float32)
    K = torch.randn(batch_size,
                    seq_len,
                    d_model,
                    requires_grad=True,
                    device='cuda',
                    dtype=torch.float32)
    V = torch.randn(batch_size,
                    seq_len,
                    d_model,
                    requires_grad=True,
                    device='cuda',
                    dtype=torch.float32)
    
    grad_output = torch.randn(batch_size,
                              seq_len,
                              d_model,
                              device='cuda',
                              dtype=torch.float32)
    
    
    for _ in range(warmup_steps):
        Q.grad = None
        K.grad = None
        V.grad = None
        
        output = attention(Q,K,V)
        output.backward(grad_output)
        
    torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(measurement_steps):
        Q.grad = None
        K.grad = None
        V.grad = None
        
        output = attention(Q,K,V)
        
        torch.cuda.synchronize()
        start_time = timeit.default_timer()
        output.backward(grad_output)
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        times.append(end_time-start_time)
    
    mean = sum(times)/len(times)
    std = numpy.std(times)
    return mean,std

def before_backward_memory_profiling(
    batch_size: int,
    seq_len: int,
    d_model: int,
    warmup_steps: int,
):
    Q = torch.randn(batch_size,
                        seq_len,
                        d_model,
                        requires_grad=True,
                        device='cuda',
                        dtype=torch.float32)
    K = torch.randn(batch_size,
                    seq_len,
                    d_model,
                    requires_grad=True,
                    device='cuda',
                    dtype=torch.float32)
    V = torch.randn(batch_size,
                    seq_len,
                    d_model,
                    requires_grad=True,
                    device='cuda',
                    dtype=torch.float32)
        
    grad_output = torch.randn(batch_size,
                              seq_len,
                              d_model,
                              device='cuda',
                              dtype=torch.float32)

    
    for _ in range(warmup_steps):
        Q.grad = None
        K.grad = None
        V.grad = None
        
        output = attention(Q,K,V)
        output.backward(grad_output)
    
    Q.grad = None
    K.grad = None
    V.grad = None
    del output
    
    torch.cuda.synchronize()
    
    baseline = torch.cuda.memory_allocated()
    
    output = attention(Q,K,V)
    torch.cuda.synchronize()
    
    before_backward_memory = torch.cuda.memory_allocated()
    
    print(
    f"baseline: {baseline / (1024**2):.3f} MiB, "
    f"before backward: {before_backward_memory / (1024**2):.3f} MiB, "
    f"forward saved: {(before_backward_memory - baseline) / (1024**2):.3f} MiB"
)  
    
        
def main() -> None:
    for d_model in D_MODEL:
        for seq_len in SEQUENCE_LENGTH:
            print("=========================")
            try:
                mean, std = benchmark_attention(
                    batch_size=BATCH_SIZE,
                    seq_len=seq_len,
                    d_model=d_model,
                    num_heads=NUM_HEADS,
                    warmup_steps=WARMUP_STEPS,
                    measurement_steps=100
                )
                print(
                    f"batch_size={BATCH_SIZE}, seq_len={seq_len}, d_model={d_model}, num_heads={NUM_HEADS}, mean_time={mean:.6f}s, std_time={std:.6f}s"
                )
                mean, std = backward_benchmark(
                    batch_size=BATCH_SIZE,
                    seq_len=seq_len,
                    d_model=d_model,
                    num_heads=NUM_HEADS,
                    warmup_steps=WARMUP_STEPS,
                    measurement_steps=100
                )
                print(
                    f"batch_size={BATCH_SIZE}, seq_len={seq_len}, d_model={d_model}, num_heads={NUM_HEADS}, mean_backward_time={mean:.6f}s, std_backward_time={std:.6f}s"
                )
                
                before_backward_memory_profiling(
                    batch_size=BATCH_SIZE,
                    seq_len=seq_len,
                    d_model=d_model,
                    warmup_steps=WARMUP_STEPS,
                )
            except torch.OutOfMemoryError:
                print("OOM")
            finally:
                gc.collect()
                torch.cuda.empty_cache()
                
            print("=========================")
            


if __name__ == "__main__":
    main()