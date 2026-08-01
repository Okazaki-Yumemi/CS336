from __future__ import annotations

import csv
import gc
import itertools
from pathlib import Path
from typing import Callable

import torch
import triton

from cs336_systems.flash_attention_triton import flash_attention_triton


# 本机 RTX 5070 Laptop 的合理范围。
# 题目原始范围还包括 16384、32768、65536，但不建议在 8GB GPU 上硬跑。
SEQUENCE_LENGTHS = [
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
]

EMBEDDING_DIMS = [16, 32, 64, 128]

DTYPES = {
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}

BATCH_SIZE = 1
IS_CAUSAL = True

# do_bench 中 warmup 和 rep 是大致的毫秒时间预算，
# 不是固定循环次数。
WARMUP_MS = 25
REPETITION_MS = 100

OUTPUT_PATH = Path("flash_attention_benchmark.csv")


def pytorch_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal_mask: torch.Tensor,
) -> torch.Tensor:
    """
    显式的普通 PyTorch attention。

    不使用 F.scaled_dot_product_attention，因为后者可能自动调用
    FlashAttention 或其他 fused kernel，失去对比意义。
    """
    scale = q.shape[-1] ** -0.5

    scores = (
        q @ k.transpose(-2, -1)
    ) * scale

    scores = scores.masked_fill(
        ~causal_mask,
        -torch.inf,
    )

    probabilities = torch.softmax(
        scores,
        dim=-1,
    )

    return probabilities @ v


def make_inputs(
    sequence_length: int,
    embedding_dim: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (
        BATCH_SIZE,
        sequence_length,
        embedding_dim,
    )

    q = torch.randn(
        shape,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )

    k = torch.randn(
        shape,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )

    v = torch.randn(
        shape,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )

    return q, k, v


def benchmark_forward(
    implementation: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor],
        torch.Tensor,
    ],
    sequence_length: int,
    embedding_dim: int,
    dtype: torch.dtype,
) -> float:
    q, k, v = make_inputs(
        sequence_length,
        embedding_dim,
        dtype,
    )

    return triton.testing.do_bench(
        lambda: implementation(q, k, v), #type: ignore
        warmup=WARMUP_MS,
        rep=REPETITION_MS,
        return_mode="median",
    )


def benchmark_backward(
    implementation: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor],
        torch.Tensor,
    ],
    sequence_length: int,
    embedding_dim: int,
    dtype: torch.dtype,
) -> float:
    q, k, v = make_inputs(
        sequence_length,
        embedding_dim,
        dtype,
    )

    # Forward 不计入 backward 时间。
    output = implementation(q, k, v)
    grad_output = torch.randn_like(output)

    torch.cuda.synchronize()

    return triton.testing.do_bench(
        lambda: output.backward( # type: ignore
            grad_output,
            retain_graph=True,
        ),
        warmup=WARMUP_MS,
        rep=REPETITION_MS,
        grad_to_none=[q, k, v],
        return_mode="median",
    )


def benchmark_forward_backward(
    implementation: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor],
        torch.Tensor,
    ],
    sequence_length: int,
    embedding_dim: int,
    dtype: torch.dtype,
) -> float:
    q, k, v = make_inputs(
        sequence_length,
        embedding_dim,
        dtype,
    )

    grad_output = torch.randn_like(q)

    def step() -> None:
        output = implementation(q, k, v)
        output.backward(grad_output)

    return triton.testing.do_bench( # type: ignore
        step,
        warmup=WARMUP_MS,
        rep=REPETITION_MS,
        grad_to_none=[q, k, v],
        return_mode="median",
    )


def is_oom_error(error: BaseException) -> bool:
    return (
        isinstance(error, torch.OutOfMemoryError)
        or (
            isinstance(error, RuntimeError)
            and "out of memory" in str(error).lower()
        )
    )


def run_safely(
    benchmark: Callable[[], float],
) -> float | str:
    try:
        return float(benchmark())
    except BaseException as error:
        if not is_oom_error(error):
            raise

        return "OOM"
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def format_result(value: float | str) -> str:
    if isinstance(value, str):
        return value

    return f"{value:.4f}"


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # Triton 的 FP32 tl.dot 在 NVIDIA GPU 上通常使用 TF32 路径。
    # 允许 PyTorch matmul 使用高性能 TF32，以免 FP32 对比明显失衡。
    torch.set_float32_matmul_precision("high")

    gpu_name = torch.cuda.get_device_name()

    print(f"GPU: {gpu_name}")
    print(f"Output: {OUTPUT_PATH}")

    columns = [
        "gpu",
        "implementation",
        "dtype",
        "sequence_length",
        "embedding_dim",
        "forward_ms",
        "backward_ms",
        "forward_backward_ms",
    ]

    with OUTPUT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=columns,
        )
        writer.writeheader()

        configurations = itertools.product(
            DTYPES.items(),
            SEQUENCE_LENGTHS,
            EMBEDDING_DIMS,
        )

        for (
            dtype_name,
            dtype,
        ), sequence_length, embedding_dim in configurations:
            for implementation_name in ["pytorch", "triton"]:
                causal_mask = None

                try:
                    if implementation_name == "pytorch":
                        # 在 benchmark 外创建 mask，避免把 mask 构造时间
                        # 算入 attention forward。
                        causal_mask = torch.ones(
                            (
                                sequence_length,
                                sequence_length,
                            ),
                            device="cuda",
                            dtype=torch.bool,
                        ).tril()

                        def implementation( # type: ignore
                            q: torch.Tensor,
                            k: torch.Tensor,
                            v: torch.Tensor,
                            mask: torch.Tensor = causal_mask,
                        ) -> torch.Tensor:
                            return pytorch_attention(
                                q,
                                k,
                                v,
                                mask,
                            )

                    else:

                        def implementation(
                            q: torch.Tensor,
                            k: torch.Tensor,
                            v: torch.Tensor,
                        ) -> torch.Tensor:
                            return flash_attention_triton(
                                q,
                                k,
                                v,
                                is_causal=IS_CAUSAL,
                            )

                    forward_ms = run_safely(
                        lambda: benchmark_forward(
                            implementation,
                            sequence_length,
                            embedding_dim,
                            dtype,
                        )
                    )

                    backward_ms = run_safely(
                        lambda: benchmark_backward(
                            implementation,
                            sequence_length,
                            embedding_dim,
                            dtype,
                        )
                    )

                    forward_backward_ms = run_safely(
                        lambda: benchmark_forward_backward(
                            implementation,
                            sequence_length,
                            embedding_dim,
                            dtype,
                        )
                    )

                except BaseException as error:
                    if not is_oom_error(error):
                        raise

                    forward_ms = "OOM"
                    backward_ms = "OOM"
                    forward_backward_ms = "OOM"

                finally:
                    del causal_mask
                    gc.collect()
                    torch.cuda.empty_cache()

                row = {
                    "gpu": gpu_name,
                    "implementation": implementation_name,
                    "dtype": dtype_name,
                    "sequence_length": sequence_length,
                    "embedding_dim": embedding_dim,
                    "forward_ms": format_result(
                        forward_ms
                    ),
                    "backward_ms": format_result(
                        backward_ms
                    ),
                    "forward_backward_ms": format_result(
                        forward_backward_ms
                    ),
                }

                writer.writerow(row)
                output_file.flush()

                print(
                    f"{implementation_name:7s} "
                    f"{dtype_name:4s} "
                    f"N={sequence_length:5d} "
                    f"D={embedding_dim:3d} | "
                    f"fwd={row['forward_ms']:>8} ms | "
                    f"bwd={row['backward_ms']:>8} ms | "
                    f"e2e={row['forward_backward_ms']:>8} ms"
                )


if __name__ == "__main__":
    main()