先做了attention的分支

```py

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
import timeit
import numpy

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.model import scaled_dot_product_attention as attention

from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = "Benchmark compiled attn or Transformer"
    )
    
    parser.add_argument(
        "--target",
        choices = ("attention", "transformer"),
        required = True,
    )
    
    parser.add_argument(
        "--implementation",
        choices= ("eager","compiled"),
        required= True,
    )
    
    parser.add_argument(
        "--mode",
        choices = ("forward", "backward"),
        required = True,
    )
    
    parser.add_argument(
        "--sequence-length",
        type = int,
        default= 32
    )
    
    parser.add_argument(
        "--d-model",
        type = int,
        default= 32
    )
    
    parser.add_argument(
        "--warmup-steps",
        type = int,
        default = 10,
    )
    
    parser.add_argument(
        "--measurement-steps",
        type = int,
        default = 10,
    )
    
    return parser.parse_args()


def attention_benchmark(
    batch_size: int,
    seq_len: int,
    d_model: int,
    warmup_steps: int,
    measurement_steps: int,
    mode: str,
    implementation: str,    
):
    attention_fn = torch.compile(attention) if implementation == "compiled" else attention
    
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
    
    #warmup
    for _ in range(warmup_steps):
        Q.grad = None
        K.grad = None
        V.grad = None
        
        output = attention_fn(Q,K,V)
        
        if mode == "backward":
            output.backward(grad_output)
    
    torch.cuda.synchronize()
    
    times: list[float] = []
    
    for _ in range(measurement_steps):
        Q.grad = None
        K.grad = None
        V.grad = None
        
        if mode == "forward":
            torch.cuda.synchronize()
            start_time = timeit.default_timer()
            output = attention_fn(Q,K,V)
            torch.cuda.synchronize()
            end_time = timeit.default_timer()
        else:
            #构图不计入backward时间
            output = attention_fn(Q,K,V)
            
            torch.cuda.synchronize()
            start_time = timeit.default_timer()
            output.backward(grad_output)
            torch.cuda.synchronize()
            end_time = timeit.default_timer()
        
        times.append(end_time-start_time)
  
    mean = sum(times)/len(times)
    std = numpy.std(times)
    
    return mean,std

def main() -> None:
    args = parse_args()
    
    if args.target == "attention":
        mean, std = attention_benchmark(
            batch_size=8,
            seq_len=args.sequence_length,
            d_model=args.d_model,
            warmup_steps=args.warmup_steps,
            measurement_steps=args.measurement_steps,
            mode=args.mode,
            implementation=args.implementation
        )
        
        print(f"mean_time={mean:.6f}s, std_time={std:.6f}s")
    else:
        raise NotImplementedError("Transformer benchmark not implemented yet")
    
if __name__ == "__main__":
    main()

```


```bash

$ uv run python scripts/compiled_benchmark.py   --target attention   --implementation compiled   --mode forward   --sequence-length 256   --d-model 16   --warmup-ste
ps 5   --measurement-steps 100
mean_time=0.000277s, std_time=0.000092s

$ uv run python scripts/compiled_benchmark.py   --target attention   --implementation compiled   --mode backward   --sequence-length 256   --d-model 16   --warmup-st
eps 5   --measurement-steps 100
mean_time=0.000519s, std_time=0.000209s

$ uv run python scripts/compiled_benchmark.py   --target attention   --implementation eager   --mode forward   --sequence-length 256   --d-model 16   --warmup-steps 5   --measurement-steps 100
mean_time=0.000161s, std_time=0.000053s

$ uv run python scripts/compiled_benchmark.py   --target attention   --implementation eager   --mode backward   --sequence-length 256   --d-model 16   --warmup-steps 5   --measurement-steps 100
mean_time=0.000733s, std_time=0.000347s
```

| 模式       |    Eager | Compiled | 加速比 ($T_\text{eager}/T_\text{compiled}$) |
| -------- | -------: | -------: | -------------------------------------: |
| Forward  | 0.161 ms | 0.277 ms |                              **0.58×** |
| Backward | 0.733 ms | 0.519 ms |                              **1.41×** |


```bash
$ uv run python scripts/compiled_benchmark.py   --target attention   --implementation compiled   --mode forward   --sequence-length 4096   --d-model 16   --warmup-st
eps 5   --measurement-steps 100
mean_time=0.006029s, std_time=0.000119s
$ uv run python scripts/compiled_benchmark.py   --target attention   --implementation eager   --mode forward   --sequence-length 4096   --d-model 16   --warmup-steps
 5   --measurement-steps 100
mean_time=0.019642s, std_time=0.001472s
```


# Transformer的

```py

def build_model(
    vocab_size = 10000,
    context_length : int = 32,
    d_model : int = 32,
    num_layers : int = 4,
    num_heads: int = 4,
    d_ff : int = 128,
    implementation : str = "eager",
)-> torch.nn.Module:
    model = BasicsTransformerLM(
        vocab_size = vocab_size,
        context_length = context_length,
        d_model = d_model,
        num_layers = num_layers,
        num_heads = num_heads,
        d_ff = d_ff
    )
    
    model = model.to("cuda")
    model.train()
    
    
    if implementation == "compiled":
        model = torch.compile(model)
    
    return model

def transformer_benchmark(
    model:torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mode: str,
    warmup_steps: int,
    measurement_steps: int,
):
    #warmup
    for _ in range(warmup_steps):
        
        if mode == "backward":
            model.zero_grad(set_to_none=True)
        
        logits = model(inputs)
        
        if mode == "backward":
            loss = cross_entropy(logits,targets)
            loss.backward()
    torch.cuda.synchronize()
    times: list[float] = []
    # ensure start_time and end_time are always defined to avoid unbound errors
    start_time: float = 0.0
    end_time: float = 0.0
    for _ in range(measurement_steps):
        if mode == "forward":
            torch.cuda.synchronize()
            start_time = timeit.default_timer()
            logits = model(inputs)
            torch.cuda.synchronize()
            end_time = timeit.default_timer()
        if mode == "backward":
            model.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = cross_entropy(logits,targets)
            torch.cuda.synchronize()
            start_time = timeit.default_timer()
            loss.backward()
            torch.cuda.synchronize()
            end_time = timeit.default_timer()
        times.append(end_time-start_time)
    
    mean = sum(times)/len(times)
    std = numpy.std(times)
    
    return mean,std

def main() -> None:
    args = parse_args()
    
    if args.target == "attention":
        mean, std = attention_benchmark(
            batch_size=8,
            seq_len=args.sequence_length,
            d_model=args.d_model,
            warmup_steps=args.warmup_steps,
            measurement_steps=args.measurement_steps,
            mode=args.mode,
            implementation=args.implementation
        )
        
        print(f"mean_time={mean:.6f}s, std_time={std:.6f}s")
    else:
        model = build_model(
            context_length=args.sequence_length,
            d_model=args.d_model,
            implementation=args.implementation
        )
        mean,std = transformer_benchmark(
            model=model,
            inputs=torch.randint(0,10000,(8,args.sequence_length),device="cuda"),
            targets=torch.randint(0,10000,(8,args.sequence_length),device="cuda"),
            mode=args.mode,
            warmup_steps=args.warmup_steps,
            measurement_steps=args.measurement_steps
        )
        print(f"mean_time={mean:.6f}s, std_time={std:.6f}s")
```

测试

```bash

uv run python scripts/compiled_benchmark.py \
  --target transformer \
  --implementation eager \
  --mode forward \
  --sequence-length 128 \
  --d-model 768 \
  --warmup-steps 5 \
  --measurement-steps 100

mean_time=0.032098s, std_time=0.003462s

uv run python scripts/compiled_benchmark.py \
  --target transformer \
  --implementation compiled \
  --mode forward \
  --sequence-length 128 \
  --d-model 768 \
  --warmup-steps 5 \
  --measurement-steps 100

mean_time=0.026821s, std_time=0.000337s

uv run python scripts/compiled_benchmark.py \
  --target transformer \
  --implementation eager \
  --mode backward \
  --sequence-length 128 \
  --d-model 768 \
  --warmup-steps 5 \
  --measurement-steps 100

mean_time=0.063187s, std_time=0.002214s

uv run python scripts/compiled_benchmark.py \
  --target transformer \
  --implementation compiled \
  --mode backward \
  --sequence-length 128 \
  --d-model 768 \
  --warmup-steps 5 \
  --measurement-steps 100

mean_time=0.050332s, std_time=0.000739s
```

| 模式       |     Eager |  Compiled |       加速比 |
| -------- | --------: | --------: | --------: |
| Forward  | 32.098 ms | 26.821 ms | **1.20×** |
| Backward | 63.187 ms | 50.332 ms | **1.26×** |


