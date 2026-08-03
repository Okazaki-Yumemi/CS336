# scr code

```py

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from torch import nn


class NaiveDDP(nn.Module):
    
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module
        
        # 所有rank使用rank 0 的参数
        with torch.no_grad():
            for parameter in self.module.parameters():
                dist.broadcast(parameter, src=0)
            
            #对带buffer的 一般nn.Module进行广播
            for buffer in self.module.buffers():
                dist.broadcast(buffer, src=0)
    
    
    def forward(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        
        return self.module(*args, **kwargs)
    
    def synchronize_gradients(self) -> None:
        
        world_size = dist.get_world_size()
        
        for parameter in self.module.parameters():
            if parameter.grad is  None:
                continue
            
            dist.all_reduce(
                parameter.grad,
                op = dist.ReduceOp.SUM,
                async_op = False,
            )
            
            parameter.grad.div_(world_size) #type: ignore
```


# adapter

```py

def get_ddp(module: torch.nn.Module) -> torch.nn.Module:
    """
    Returns a torch.nn.Module container that handles
    parameter broadcasting and gradient synchronization for
    distributed data parallel training.

    This container should overlaps communication with backprop computation
    by asynchronously communicating gradients as they are ready
    in the backward pass. The gradient for each parameter tensor
    is individually communicated.

    Args:
        module: torch.nn.Module
            Underlying model to wrap with DDP.
    Returns:
        Instance of a DDP class.
    """
    # For example: return DDP(module)
    return NaiveDDP(module)


def ddp_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    # For example: ddp_model.finish_gradient_synchronization()
    del optimizer  # unused
    ddp_model.synchronize_gradients() #type: ignore
```

# 测试

```bash
==== test session starts ====
platform linux -- Python 3.13.12, pytest-9.0.3, pluggy-1.6.0
rootdir: /home/soyo/projects/CS336-2026/assignments/a2-systems
configfile: pyproject.toml
plugins: timeout-2.4.0, jaxtyping-0.3.9
collected 2 items                                                                                                                                                                                                                     

tests/test_ddp.py::test_DistributedDataParallel[ToyModel] PASSED
tests/test_ddp.py::test_DistributedDataParallel[ToyModelWithTiedWeights] PASSED

==== 2 passed in 7.62s ====
```

# summary

这个类其实就是先创建一个nn.Module的类，初始化的时候将rank0的参数广播到所有rank上，然后在每次backward之后调用synchronize_gradients()函数将所有rank的梯度进行all_reduce求平均。

