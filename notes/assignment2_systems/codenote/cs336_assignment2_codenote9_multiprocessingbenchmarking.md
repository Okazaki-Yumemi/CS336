```py
from __future__ import annotations

import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

MASTER_ADDR = "127.0.0.1"
MASTER_PORT = "29500"

def setup(
    rank:int,
    world_size:int,
)-> None:
    os.environ["MASTER_ADDR"] = MASTER_ADDR
    os.environ["MASTER_PORT"] = MASTER_PORT
    
    torch.cuda.set_device(rank)
    
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )
    
    
def cleanup() -> None:
    dist.destroy_process_group()
    

def benchmark_worker(
    rank:int,
    world_size:int,
    size_mib:int,
    warmup_iters:int,
    measure_iters:int,
) -> None:
    
    setup(rank, world_size)
    
    try:
        
        device = torch.device("cuda", rank)
        
        num_elements = size_mib * 1024 * 1024 // 4
        
        tensor = torch.zeros(
            num_elements,
            dtype=torch.float32,
            device=device,
        )
        
        # 所有worker初始化后开始
        
        dist.barrier()
        
        for _ in range(warmup_iters):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            
        torch.cuda.synchronize(device)
        
        timings_ms: list[float] = []
        
        for _ in range(measure_iters):
            # 避免部分worker提前结束，导致all_reduce阻塞
            dist.barrier()
            
            #清空此前的CUDA事件
            torch.cuda.synchronize(device)
            
            start = time.perf_counter()
            
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            
            torch.cuda.synchronize(device)
            
            end = time.perf_counter()
            timings_ms.append((end - start) * 1000)
            
            all_timings: list[list[float] | None] = [
                    None for _ in range(world_size)
                ]
            
            dist.all_gather_object(all_timings, timings_ms)
                
                # 让rank 0的worker来计算平均时间和吞吐量
            if rank == 0:
                flattened = [
                    elapsed
                    for rank_timings in all_timings
                    if rank_timings is not None
                    for elapsed in rank_timings
                ]
                
                mean_ms = sum(flattened) / len(flattened)
                
                print(
                    f"world_size={world_size}", 
                    f"size={size_mib} MiB",
                    f"mean_time_ms={mean_ms:.2f}"
                )

    finally:
        cleanup()
    
    
    
    
        
def main() -> None:
    
    world_size = 2
    
    if torch.cuda.device_count() < world_size:
        raise RuntimeError(
            f"Need {world_size} GPUs,"
            f"but found {torch.cuda.device_count()}"
        )
    
    mp.spawn( # type: ignore
        fn =benchmark_worker,
        args=(world_size, 1, 5, 10),
        nprocs=world_size,
        join=True,
    )

if __name__ == "__main__":
    main()
    
    
```

状态不好，有点力竭，这个地方就参考csapp里面的多线程实现吧，无非是包装了一点库，没什么好讲的



