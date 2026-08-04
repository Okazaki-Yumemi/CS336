from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist

class OptimizerStateSharding(torch.optim.Optimizer):
    
    def __init__(
        self,
        params: Any,
        optimizer_cls: type[torch.optim.Optimizer],
        **kwargs: Any,
        ):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.parameter_owners = {}
        self.local_param_groups = []
        self.local_optimizer = None
        self.next_parameter_index = 0
        self.initializing = True
        
        super().__init__(params, kwargs)
        
        self.initializing = False
        self.local_optimizer = optimizer_cls(self.local_param_groups, **kwargs)
        
        self.state = self.local_optimizer.state
        
    def add_param_group(
        self,
        param_group: dict[str, Any],
    ):
        full_group = param_group.copy()
        super().add_param_group(full_group)
        
        local_params = []
        for param in full_group['params']:
            owner = self.next_parameter_index % self.world_size
            self.parameter_owners[param] = owner
            
            if owner == self.rank:
                local_params.append(param)
            
            self.next_parameter_index += 1
        
        local_group = full_group.copy()
        local_group['params'] = local_params
        

        if self.initializing:
            self.local_param_groups.append(local_group)
        else:
            self.local_optimizer.add_param_group(local_group) # type: ignore
        
            
            
    def step(
        self,
        closure = None,
        **kwargs: Any,
    ):
        loss = self.local_optimizer.step( # type: ignore
            closure=closure,
            **kwargs,
        )
        
        with torch.no_grad():
            for group in self.param_groups:
                for param in group['params']:
                    owner = self.parameter_owners[param]
                    dist.broadcast(param, src=owner)
        return loss