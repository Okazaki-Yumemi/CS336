from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from torch._utils import (
    _flatten_dense_tensors,
    _unflatten_dense_tensors
)


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
        #这个函数负责在backward之后同步梯度，确保所有rank的梯度一致
        
        world_size = dist.get_world_size()
        
        for parameter in self.module.parameters():
            if parameter.grad is  None: # 空梯度直接跳过
                continue
            
            dist.all_reduce(      # all_reduce操作会将所有rank的梯度进行求和，并将结果广播到所有rank
                parameter.grad,
                op = dist.ReduceOp.SUM,
                async_op = False,
            )
            
            parameter.grad.div_(world_size) #type: ignore
            

class FlatDDP(NaiveDDP):
    
    def synchronize_gradients(self) -> None:
        world_size = dist.get_world_size()
        
        gradients = [parameter.grad for parameter in self.module.parameters() if parameter.grad is not None]
        
        if not gradients:
            return

        flat_gradients = _flatten_dense_tensors(gradients)
        
        dist.all_reduce(
            flat_gradients,
            op = dist.ReduceOp.SUM,
            async_op = False,
        )
        
        flat_gradients.div_(world_size) #type: ignore
        
        synchronized_gradients = _unflatten_dense_tensors(flat_gradients, gradients)
        
        for original_gradient, synchronized_gradient in zip(
            gradients,
            synchronized_gradients,
            strict = True,
        ):
            original_gradient.copy_(
                synchronized_gradient
        )
            
            