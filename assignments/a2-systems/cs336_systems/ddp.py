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
            
            
class OverlappingDDP(nn.Module):
    
    def __init__(self, module: nn.Module) -> None:
        
        super().__init__()
        self.module = module
        
        self.world_size = dist.get_world_size()
        
        # 保存通信，每项保存 (handle,parameter) 的元组
        self.pending_communications = []
        
        #注册 hook_handles:
        self.hook_handles: list = []
        
        with torch.no_grad():
            for parameter in self.module.parameters():
                dist.broadcast(parameter, src=0)
            
            for buffer in self.module.buffers():
                dist.broadcast(buffer, src=0)
        
        for parameter in self.module.parameters():
            if parameter.requires_grad == False:
                continue
            
            #增加当前参数的hook
            self.hook_handles.append(
                parameter.register_post_accumulate_grad_hook(self._create_hook(parameter))
                )

    # 梯度hook
    def _create_hook(self, parameter: torch.Tensor):
        def hook(_: torch.Tensor) -> None:
            if parameter.grad is None:
                return

            handle = dist.all_reduce(
                parameter.grad,
                op=dist.ReduceOp.SUM,
                async_op=True,
            )
            
            self.pending_communications.append((handle, parameter))

        return hook
    
    def finish_gradient_synchronization(self) -> None:
        
        for handle, parameter in self.pending_communications:
            handle.wait()
            
            parameter.grad.div_(self.world_size)
        
        self.pending_communications.clear()
        
    def forward(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        
        return self.module(*args, **kwargs)
            