from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
import timeit
import numpy

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy

from pathlib import Path


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
    
    parser.add_argument(
        "--precision",
        choices=["fp32","bf16"],
        default= "fp32",
    )
    
    parser.add_argument(
        "--memory-profile",
        action= "store_true",
    )
    
    parser.add_argument(
        "--snapshot-path",
        type=str,
        default="profiles/memory_snapshot.pickle",
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
    precision: str
):
    if mode != "forward":
        optimizer.zero_grad(set_to_none= True)
        
    use_bf16 = precision == "bf16"
        
    with torch.autocast(
        device_type= "cuda",
        dtype= torch.bfloat16,
        enabled= use_bf16,
    ):
        logits = model(inputs)
        
        if mode != "forward":
            loss = cross_entropy(logits,targets)
    
    if mode == "forward":
        return
    
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
    precision : str,
): 
    
    for _ in range(warmup_steps):
        run_step(
            model,
            inputs,
            targets,
            mode,
            optimizer,
            precision,
        )
        torch.cuda.synchronize()
        
    times : list[float] = []
    with torch.cuda.nvtx.range("measurement"):
        for _ in range(measurement_steps):
            torch.cuda.synchronize()
            
            start = timeit.default_timer()
            
            run_step(
                model,
                inputs,
                targets,
                mode,
                optimizer,
                precision
            )
            torch.cuda.synchronize()
            end = timeit.default_timer()
            
            times.append(end-start)
    
    mean = sum(times)/len(times)
    std = numpy.std(times)
    
    return mean,std

def profile_memory(
    model: BasicsTransformerLM,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mode : str,
    optimizer : AdamW,
    precision: str,
    warmup_steps :int,
    snapshot_path:str,
):
    if mode not in {"forward","train"}:
        raise ValueError("Memory profiling only supports forward or train mode")

    def profile_step():
        if mode == "forward":
            # 题目书评inference only, 不考虑反向传播
            with torch.inference_mode():
                run_step(
                    model,
                    inputs,
                    targets,
                    mode,
                    optimizer,
                    precision
                )
        else:
            run_step(
                model,
                inputs,
                targets,
                mode,
                optimizer,
                precision,
            )
    for _ in range(warmup_steps):
        profile_step()
    
    torch.cuda.synchronize()
    
    Path(snapshot_path).parent.mkdir(parents=True,exist_ok=True)
    
    # 从稳态显存统计
    torch.cuda.reset_peak_host_memory_stats()
    
    torch.cuda.memory._record_memory_history(
        max_entries = 1_000_000
    )
    
    try:
        #分析一个step
        profile_step()
        torch.cuda.synchronize()
        
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        
        torch.cuda.memory._dump_snapshot(snapshot_path)
    finally:
        torch.cuda.memory._record_memory_history(
            enabled= None
        )
    
    print(f"snapshot: {snapshot_path}")
    print(
        f"peak allocated: {peak_allocated / 1024**3:.3f} GiB"
    )
    print(
        f"peak reserved: {peak_reserved/1024**3:.3f} GiB"
    )
    


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
    print("=========================================")
    print(f"device: {device}")
    print(f"model size: {args.model_size}")
    print(f"mode: {args.mode}")
    print(f"context length: {args.context_length}")
    print(f"parameters: {num_parameters:,}")
    print(f"input shape: {tuple(inputs.shape)}")
    print(f"target shape: {tuple(targets.shape)}")
    print(f"precision: {args.precision}")
    
    
    optimizer = AdamW(model.parameters())
    
    if args.memory_profile:
        profile_memory(
            model = model,
            inputs= inputs,
            targets= targets,
            mode = args.mode,
            optimizer= optimizer,
            precision= args.precision,
            warmup_steps= args.warmup_steps,
            snapshot_path= args.snapshot_path,
        )
        print("=========================================")
    else:
        mean,std = benchmark(
            model= model,
            inputs= inputs,
            targets= targets,
            mode = args.mode,
            optimizer= optimizer,
            warmup_steps= args.warmup_steps,
            measurement_steps= args.measurement_steps,
            precision= args.precision
        )
    
        print(f"mean time(ms): {mean*1000}")
        print(f"std(ms) : {std*100}")
        print("=========================================")

if __name__ == "__main__":
    main()
    