# 着重看代码学习吧，没啥好说的。
# 不难

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



VOCAB_SIZE = 10_000
BATCH_SIZE = 4

@dataclass(frozen = True)
class ModelConfig:
    d_model : int
    d_ff : int
    num_layers : int
    num_heads : int
    
MODEL_CONFIGS : dict[str, ModelConfig] = {
    "small" : ModelConfig(
        d_model = 768,
        d_ff = 3072,
        num_layers = 12,
        num_heads = 12
    ),
    "medium" : ModelConfig(
        d_model = 1024,
        d_ff = 4096,
        num_layers = 24,
        num_heads = 16
    ),
    "large" : ModelConfig(
        d_model = 1280,
        d_ff = 5120,
        num_layers = 36,
        num_heads = 20
    ),
    "xl" : ModelConfig(
        d_model = 2560,
        d_ff = 10240,
        num_layers = 32,
        num_heads = 32
    ),
    "10B": ModelConfig(
        d_model = 4608,
        d_ff = 12288,
        num_layers = 50,
        num_heads = 36,
    )
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = "Benchmark the CS336 Transformer model."
    )
    
    parser.add_argument(
        "--model-size",
        choices= MODEL_CONFIGS.keys(),
        default= "small"
    )
    
    parser.add_argument(
        "--mode",
        choices=("forward","backward","train"),
        default= "forward",
        help=(
            "forward: forward only"
            "backward: forward + backward"
            "train: forward + backward + optimizer step"
        ),
    )
    
    parser.add_argument(
        "--context-length",
        type= int,
        default= 512,
    )
    
    parser.add_argument(
        "--warmup-steps",
        type = int,
        default= 5
    )
    
    parser.add_argument(
        "--measurement-steps",
        type = int,
        default= 10
    )
    
    parser.add_argument(
        "--device",
        type= str,
        default= "cuda"
    )
    
    return parser.parse_args()

def build_model(
    model_size : str,
    context_length : int,
    device: torch.device,
) -> BasicsTransformerLM:
    
    config = MODEL_CONFIGS[model_size]
    
    model = BasicsTransformerLM(
        vocab_size= VOCAB_SIZE,
        context_length= context_length,
        d_model= config.d_model,
        num_layers= config.num_layers,
        num_heads= config.num_heads,
        d_ff  = config.d_ff,
    )
    
    return model.to(device)

def build_random_batch(
    context_length : int,
    device: torch.device,
) -> tuple[torch.Tensor , torch.Tensor]:
    
    input = torch.randint(
        low = 0,
        high= VOCAB_SIZE,
        size = (BATCH_SIZE,context_length),
        dtype= torch.long,
        device = device,
    )
    
    targets = torch.randint(
        low  = 0,
        high= VOCAB_SIZE,
        size = (BATCH_SIZE,context_length),
        dtype = torch.long,
        device = device
    )
    
    return input,targets

def run_step(
    model : BasicsTransformerLM,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mode: str,
    optimizer : AdamW,
):
    if mode != "forward":
        optimizer.zero_grad(set_to_none= True)
        
    logits = model(inputs)
    
    if mode == "forward":
        return

    loss = cross_entropy(logits,targets)
    
    loss.backward()
    
    if mode == "train":
        optimizer.step()
        
def benchmark(
    model: BasicsTransformerLM,
    inputs : torch.Tensor,
    targets : torch.Tensor,
    mode : str,
    optimizer : AdamW,
    warmup_steps : int,
    measurement_steps : int,
): 
    
    for _ in range(warmup_steps):
        run_step(
            model,
            inputs,
            targets,
            mode,
            optimizer
        )
        torch.cuda.synchronize()
        
    times : list[float] = []
    
    for _ in range(measurement_steps):
        torch.cuda.synchronize()
        
        start = timeit.default_timer()
        
        run_step(
            model,
            inputs,
            targets,
            mode,
            optimizer
        )
        torch.cuda.synchronize()
        end = timeit.default_timer()
        
        times.append(end-start)
    
    mean = sum(times)/len(times)
    std = numpy.std(times)
    
    return mean,std


def main() -> None:
    args = parse_args()
    
    if args.context_length <= 0:
        raise ValueError("context_length must be positive.")
    
    if args.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative.")
    
    if args.measurement_steps <= 0:
        raise ValueError("measurement_steps must be positive.")
    
    device = torch.device(args.device)
    
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    
    model = build_model(
        model_size = args.model_size,
        context_length= args.context_length,
        device= device
    )
    
    inputs, targets = build_random_batch(
        context_length= args.context_length,
        device = device,
    )
    
    num_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    
    print(f"device: {device}")
    print(f"model size: {args.model_size}")
    print(f"mode: {args.mode}")
    print(f"context length: {args.context_length}")
    print(f"parameters: {num_parameters:,}")
    print(f"input shape: {tuple(inputs.shape)}")
    print(f"target shape: {tuple(targets.shape)}")
    
    
    optimizer = AdamW(model.parameters())
    
    
    mean,std = benchmark(
        model= model,
        inputs= inputs,
        targets= targets,
        mode = args.mode,
        optimizer= optimizer,
        warmup_steps= args.warmup_steps,
        measurement_steps= args.measurement_steps
    )
    
    print(f"mean time(ms): {mean*1000}")
    print(f"std(ms) : {std*100}")
    

if __name__ == "__main__":
    main()
    
```


题目要求测试，5 warmup, 10 测量， batchsize 默认 4 词表 10000, context length 512

```bash
uv run python scripts/benchmark.py \
  --model-size small \
  --context-length 512 \
  --mode forward \
  --warmup-steps 5 \
  --measurement-steps 10

device: cuda
model size: small
mode: forward
context length: 512
parameters: 128,625,408
input shape: (4, 512)
target shape: (4, 512)
mean time(ms): 78.5851458000252
std(ms) : 0.10643547187363386

```
```bash
uv run python scripts/benchmark.py \
  --model-size small \
  --context-length 512 \
  --mode backward \
  --warmup-steps 5 \
  --measurement-steps 10

device: cuda
model size: small
mode: backward
context length: 512
parameters: 128,625,408
input shape: (4, 512)
target shape: (4, 512)
mean time(ms): 226.18507680008406
std(ms) : 0.07435551090416478
```
```bash
uv run python scripts/benchmark.py \
  --model-size small \
  --context-length 512 \
  --mode train \
  --warmup-steps 5 \
  --measurement-steps 10

device: cuda
model size: small
mode: train
context length: 512
parameters: 128,625,408
input shape: (4, 512)
target shape: (4, 512)
mean time(ms): 267.0520071001192
std(ms) : 1.1726178453452063
```

| 模式                 |       平均耗时 |      标准差 |
| ------------------ | ---------: | -------: |
| Forward            |  78.585 ms | 0.106 ms |
| Forward + Backward | 226.185 ms | 0.074 ms |
| 完整训练步              | 267.052 ms | 1.173 ms |

## 验证warmup作用

```bash
(base) soyo@localhost:~/projects/CS336-2026/assignments/a2-systems$ uv run python scripts/benchmark.py   --model-size small   --context-length 512   --mode forward   --warmup-steps 0   --measurement-steps 10
device: cuda
model size: small
mode: forward
context length: 512
parameters: 128,625,408
input shape: (4, 512)
target shape: (4, 512)
mean time(ms): 90.47552180009006
std(ms) : 4.679415858448415
(base) soyo@localhost:~/projects/CS336-2026/assignments/a2-systems$ uv run python scripts/benchmark.py   --model-size small   --context-length 512   --mode forward   --warmup-steps 1   --measurement-steps 10
device: cuda
model size: small
mode: forward
context length: 512
parameters: 128,625,408
input shape: (4, 512)
target shape: (4, 512)
mean time(ms): 75.65579049996813
std(ms) : 0.1122930127906839
(base) soyo@localhost:~/projects/CS336-2026/assignments/a2-systems$ uv run python scripts/benchmark.py   --model-size small   --context-length 512   --mode forward   --warmup-steps 2   --measurement-steps 10
device: cuda
model size: small
mode: forward
context length: 512
parameters: 128,625,408
input shape: (4, 512)
target shape: (4, 512)
mean time(ms): 82.69733280003493
std(ms) : 0.30852336186521595
```

| Warmup |      Mean |      Std |
| -----: | --------: | -------: |
|      0 | 90.476 ms | 4.679 ms |
|      1 | 75.656 ms | 0.112 ms |
|      2 | 82.697 ms | 0.309 ms |
|      5 | 78.585 ms | 0.106 ms |

无 warmup 时，均值更高且方差大很多，说明第一次正式测量包含了 CUDA 初始化、缓存冷启动、GPU 升频等额外开销。1、2、5 次 warmup 后结果都明显稳定，但均值不严格单调，因为每条命令是独立进程，笔记本 GPU 的频率、温度和功耗状态会变化。题目本身正是要求分析这种现象。


## 测试medium的运行性

```bash
uv run python scripts/benchmark.py \
  --model-size medium \
  --context-length 512 \
  --mode forward \
  --warmup-steps 5 \
  --measurement-steps 10


device: cuda
model size: medium
mode: forward
context length: 512
parameters: 423,183,360
input shape: (4, 512)
target shape: (4, 512)
mean time(ms): 19559.970957400037
std(ms) : 294.7853893797014
```

可以看出，笔记本带不动这个模型，forward 需要 20 秒左右，backward 估计要 60 秒以上，训练步大约 70 秒以上。
