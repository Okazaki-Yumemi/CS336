```py

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
import timeit
import numpy

from cs336_basics.model import scaled_dot_product_attention as attention

D_MODEL = [16]
SEQUENCE_LENGTH = [256]
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
            mean, std = benchmark_attention(
                batch_size=BATCH_SIZE,
                seq_len=seq_len,
                d_model=d_model,
                num_heads=NUM_HEADS,
                warmup_steps=WARMUP_STEPS,
                measurement_steps=10
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
                measurement_steps=10
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


if __name__ == "__main__":
    main()
```

然后加入OOM处理...其实正常跑没问题，主要是笔记本显卡8GB显存扛不住

```py

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
                    measurement_steps=10
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
                    measurement_steps=10
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
```

```bash

$ uv run python scripts/benchmark_attention.py
=========================
batch_size=8, seq_len=256, d_model=16, num_heads=1, mean_time=0.000130s, std_time=0.000048s
batch_size=8, seq_len=256, d_model=16, num_heads=1, mean_backward_time=0.000663s, std_backward_time=0.000321s
baseline: 16.750 MiB, before backward: 20.898 MiB, forward saved: 4.148 MiB
=========================
=========================
batch_size=8, seq_len=1024, d_model=16, num_heads=1, mean_time=0.001097s, std_time=0.000107s
batch_size=8, seq_len=1024, d_model=16, num_heads=1, mean_backward_time=0.003101s, std_backward_time=0.000423s
baseline: 18.250 MiB, before backward: 82.844 MiB, forward saved: 64.594 MiB
=========================
=========================
batch_size=8, seq_len=4096, d_model=16, num_heads=1, mean_time=0.018238s, std_time=0.000612s
batch_size=8, seq_len=4096, d_model=16, num_heads=1, mean_backward_time=0.047863s, std_backward_time=0.000802s
baseline: 24.250 MiB, before backward: 1050.625 MiB, forward saved: 1026.375 MiB
=========================
=========================
batch_size=8, seq_len=256, d_model=32, num_heads=1, mean_time=0.000200s, std_time=0.000073s
batch_size=8, seq_len=256, d_model=32, num_heads=1, mean_backward_time=0.000703s, std_backward_time=0.000323s
baseline: 17.250 MiB, before backward: 21.523 MiB, forward saved: 4.273 MiB
=========================
=========================
batch_size=8, seq_len=1024, d_model=32, num_heads=1, mean_time=0.001164s, std_time=0.000037s
batch_size=8, seq_len=1024, d_model=32, num_heads=1, mean_backward_time=0.003055s, std_backward_time=0.000091s
baseline: 20.250 MiB, before backward: 85.344 MiB, forward saved: 65.094 MiB
=========================
=========================
batch_size=8, seq_len=4096, d_model=32, num_heads=1, mean_time=0.019066s, std_time=0.000234s
batch_size=8, seq_len=4096, d_model=32, num_heads=1, mean_backward_time=0.047526s, std_backward_time=0.000329s
baseline: 32.250 MiB, before backward: 1060.625 MiB, forward saved: 1028.375 MiB
=========================
```


| (d) |  (N) |   Forward |  Backward | Forward saved memory |
| --: | ---: | --------: | --------: | -------------------: |
|  16 |  256 |  0.130 ms |  0.663 ms |            4.148 MiB |
|  16 | 1024 |  1.097 ms |  3.101 ms |           64.594 MiB |
|  16 | 4096 | 18.238 ms | 47.863 ms |         1026.375 MiB |
|  32 |  256 |  0.200 ms |  0.703 ms |            4.273 MiB |
|  32 | 1024 |  1.164 ms |  3.055 ms |           65.094 MiB |
|  32 | 4096 | 19.066 ms | 47.526 ms |         1028.375 MiB |

d ∈ {64, 128}：未完成，受本地 8GB GPU 资源限制
N ∈ {8192, 16384}：OOM

1. 显存呈二次增长
2. 大规模下运行时间也接近二次增长
3. d 从 16 增加到 32，显存变化很小

