from __future__ import annotations

import argparse
import csv
import os
import statistics
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from cs336_basics.model import BasicsTransformerLM

# 按照你实际存放 NaiveDDP 的文件修改这一行
from cs336_systems.ddp import NaiveDDP


MODEL_CONFIGS: dict[str, dict[str, int]] = {
    "small": {
        "d_model": 768,
        "d_ff": 3072,
        "num_layers": 12,
        "num_heads": 12,
    },
    "xl": {
        "d_model": 2560,
        "d_ff": 10240,
        "num_layers": 32,
        "num_heads": 32,
    },
}


@dataclass(frozen=True)
class BenchmarkConfig:
    model_size: str
    dtype: str
    world_size: int
    local_batch_size: int
    context_length: int
    vocab_size: int
    warmup_steps: int
    measure_steps: int
    learning_rate: float
    master_addr: str
    master_port: int
    output: str


def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(
        description="Benchmark the naïve per-parameter DDP implementation."
    )

    parser.add_argument(
        "--model-size",
        choices=MODEL_CONFIGS.keys(),
        default="small",
    )
    parser.add_argument(
        "--dtype",
        choices=("fp32", "bf16"),
        default="bf16",
    )
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--local-batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29_503)
    parser.add_argument(
        "--output",
        default="naive_ddp_benchmark.csv",
    )

    args = parser.parse_args()

    return BenchmarkConfig(
        model_size=args.model_size,
        dtype=args.dtype,
        world_size=args.world_size,
        local_batch_size=args.local_batch_size,
        context_length=args.context_length,
        vocab_size=args.vocab_size,
        warmup_steps=args.warmup,
        measure_steps=args.iters,
        learning_rate=args.learning_rate,
        master_addr=args.master_addr,
        master_port=args.master_port,
        output=args.output,
    )


def get_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "fp32":
        return torch.float32

    if dtype_name == "bf16":
        return torch.bfloat16

    raise ValueError(f"Unsupported dtype: {dtype_name}")


def setup_process_group(
    rank: int,
    config: BenchmarkConfig,
) -> None:
    os.environ["MASTER_ADDR"] = config.master_addr
    os.environ["MASTER_PORT"] = str(config.master_port)

    torch.cuda.set_device(rank)

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=config.world_size,
    )


def cleanup_process_group() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def build_model(
    config: BenchmarkConfig,
    device: torch.device,
) -> BasicsTransformerLM:
    model_config = MODEL_CONFIGS[config.model_size]
    dtype = get_dtype(config.dtype)

    model = BasicsTransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=model_config["d_model"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        d_ff=model_config["d_ff"],
        rope_theta=10_000.0,
    )

    return model.to(
        device=device,
        dtype=dtype,
    )


def run_training_step(
    ddp_model: NaiveDDP,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    vocab_size: int,
    device: torch.device,
) -> tuple[float, float]:
    """
    返回：
        total_time_ms
        communication_time_ms
    """

    # 所有 rank 尽量从同一个时刻开始该 iteration。
    # barrier 本身不计入训练时间。
    dist.barrier()
    torch.cuda.synchronize(device)

    total_start = torch.cuda.Event(enable_timing=True)
    communication_start = torch.cuda.Event(enable_timing=True)
    communication_end = torch.cuda.Event(enable_timing=True)
    total_end = torch.cuda.Event(enable_timing=True)

    total_start.record()

    optimizer.zero_grad(set_to_none=True)

    logits = ddp_model(inputs)

    # 在 FP32 中计算交叉熵，避免 BF16 reduction 的数值问题。
    loss = F.cross_entropy(
        logits.float().reshape(-1, vocab_size),
        targets.reshape(-1),
    )

    loss.backward()

    # naïve DDP 的通信发生在整个 backward 完成之后。
    communication_start.record()

    ddp_model.synchronize_gradients()

    communication_end.record()

    optimizer.step()

    total_end.record()

    # 等待当前 iteration 的所有 GPU 工作真正完成。
    torch.cuda.synchronize(device)

    total_time_ms = total_start.elapsed_time(total_end)
    communication_time_ms = communication_start.elapsed_time(
        communication_end
    )

    return total_time_ms, communication_time_ms


def reduce_timings_to_slowest_rank(
    total_time_ms: float,
    communication_time_ms: float,
    device: torch.device,
) -> tuple[float, float]:
    """
    DDP iteration 的进度取决于最慢的 rank，而不是各 rank 的平均值。

    因此对每个 iteration 取所有 rank 中的最大时间。
    """

    timings = torch.tensor(
        [total_time_ms, communication_time_ms],
        dtype=torch.float32,
        device=device,
    )

    dist.all_reduce(
        timings,
        op=dist.ReduceOp.MAX,
    )

    return float(timings[0].item()), float(timings[1].item())


def write_result(
    config: BenchmarkConfig,
    total_times_ms: list[float],
    communication_times_ms: list[float],
) -> None:
    mean_total_ms = statistics.mean(total_times_ms)
    mean_communication_ms = statistics.mean(communication_times_ms)

    std_total_ms = (
        statistics.stdev(total_times_ms)
        if len(total_times_ms) > 1
        else 0.0
    )
    std_communication_ms = (
        statistics.stdev(communication_times_ms)
        if len(communication_times_ms) > 1
        else 0.0
    )

    communication_fraction = (
        mean_communication_ms / mean_total_ms
    )

    print()
    print("Naïve DDP benchmark")
    print("-------------------")
    print(f"model_size          : {config.model_size}")
    print(f"dtype               : {config.dtype}")
    print(f"world_size          : {config.world_size}")
    print(f"local_batch_size    : {config.local_batch_size}")
    print(
        f"global_batch_size   : "
        f"{config.local_batch_size * config.world_size}"
    )
    print(f"context_length      : {config.context_length}")
    print(
        f"total_time_ms       : "
        f"{mean_total_ms:.3f} ± {std_total_ms:.3f}"
    )
    print(
        f"communication_ms    : "
        f"{mean_communication_ms:.3f} "
        f"± {std_communication_ms:.3f}"
    )
    print(
        f"communication_share : "
        f"{communication_fraction * 100:.2f}%"
    )

    output_path = Path(config.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "method",
                "model_size",
                "dtype",
                "world_size",
                "local_batch_size",
                "global_batch_size",
                "context_length",
                "mean_total_ms",
                "std_total_ms",
                "mean_communication_ms",
                "std_communication_ms",
                "communication_fraction",
            ],
        )

        writer.writeheader()
        writer.writerow(
            {
                "method": "naive_per_parameter",
                "model_size": config.model_size,
                "dtype": config.dtype,
                "world_size": config.world_size,
                "local_batch_size": config.local_batch_size,
                "global_batch_size": (
                    config.local_batch_size * config.world_size
                ),
                "context_length": config.context_length,
                "mean_total_ms": mean_total_ms,
                "std_total_ms": std_total_ms,
                "mean_communication_ms": mean_communication_ms,
                "std_communication_ms": std_communication_ms,
                "communication_fraction": communication_fraction,
            }
        )

    print(f"saved               : {output_path}")


def benchmark_worker(
    rank: int,
    config: BenchmarkConfig,
) -> None:
    setup_process_group(rank, config)

    try:
        device = torch.device("cuda", rank)

        # 所有 rank 用同一随机种子创建模型。
        # NaiveDDP 之后仍会显式广播 rank 0 参数。
        torch.manual_seed(2026)

        model = build_model(config, device)
        ddp_model = NaiveDDP(model)
        ddp_model.train()

        # 使用 SGD 是为了将 benchmark 集中在 DDP 通信上，
        # 并避免 AdamW 的两份额外 optimizer state 占用显存。
        optimizer = torch.optim.SGD(
            ddp_model.parameters(),
            lr=config.learning_rate,
        )

        # 每个 rank 使用不同的数据，相当于数据并行中的不同 shard。
        generator = torch.Generator(device=device)
        generator.manual_seed(10_000 + rank)

        inputs = torch.randint(
            low=0,
            high=config.vocab_size,
            size=(
                config.local_batch_size,
                config.context_length,
            ),
            device=device,
            dtype=torch.long,
            generator=generator,
        )

        targets = torch.randint(
            low=0,
            high=config.vocab_size,
            size=(
                config.local_batch_size,
                config.context_length,
            ),
            device=device,
            dtype=torch.long,
            generator=generator,
        )

        # Warm-up
        for _ in range(config.warmup_steps):
            run_training_step(
                ddp_model=ddp_model,
                optimizer=optimizer,
                inputs=inputs,
                targets=targets,
                vocab_size=config.vocab_size,
                device=device,
            )

        total_times_ms: list[float] = []
        communication_times_ms: list[float] = []

        # Measurement
        for _ in range(config.measure_steps):
            local_total_ms, local_communication_ms = (
                run_training_step(
                    ddp_model=ddp_model,
                    optimizer=optimizer,
                    inputs=inputs,
                    targets=targets,
                    vocab_size=config.vocab_size,
                    device=device,
                )
            )

            total_ms, communication_ms = (
                reduce_timings_to_slowest_rank(
                    total_time_ms=local_total_ms,
                    communication_time_ms=local_communication_ms,
                    device=device,
                )
            )

            if rank == 0:
                total_times_ms.append(total_ms)
                communication_times_ms.append(
                    communication_ms
                )

        if rank == 0:
            write_result(
                config=config,
                total_times_ms=total_times_ms,
                communication_times_ms=communication_times_ms,
            )

    finally:
        cleanup_process_group()


def main() -> None:
    config = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    available_gpus = torch.cuda.device_count()

    if available_gpus < config.world_size:
        raise RuntimeError(
            f"Need {config.world_size} GPUs, "
            f"but found {available_gpus}."
        )

    mp.spawn( # type: ignore
        fn=benchmark_worker,
        args=(config,),
        nprocs=config.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()