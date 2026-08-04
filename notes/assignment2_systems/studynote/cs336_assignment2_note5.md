# 6.Optimizer State Sharding
Reducing redundancy in data-parallel training by partitioning the (1) optimizer state (2) gradients and (3) parameters across ranks.

In this part of the assignment, we'll reduce per-rank memory consumption by implementing a simplified version of optimizer state sharding.

Rather than keeping the optimizer states for all parameters, each rank's optimizer instance will only handle a subset of the parameters ( approximately 1 / world_size) .

When each rank's optimizer takes an potimizer step, it'll only update the subset of model parameters in its shard.

Then each rank will broadcast its updated parameters to all other ranks to ensure that the model parameters remain synchronized after each optimizer step.

**Problem Optimizer State Sharding**:

Implement a python class to handle optimizer state sharding. This class should warp an arbitray input PyTorch optimizer and take care of synchronizing updated parameters after each optimizer step.

```py


def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs:Any):

def step(self, closure, **kwargs)

def add_param_group(self, param_group: dict[str, Any])
```

考虑到Chapter 6 只有一个作业，不单独设置code note.


```py
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
```

这个代码实现的是一个简化版的 optimizer state sharding，它将参数分配给不同的 rank，并在每次优化器步骤后广播更新的参数以保持同步。

假设有两个rank，6个参数 p0~p5

owner = parameter_index % world_size

rank 0 负责 p0, p2, p4
rank 1 负责 p1, p3, p5

负责只表示这个rank的 optimizer会更新这些参数，会保存Adam momentum 等optimizer state等待

代码继承 torch.optim.Optimizer，可以正常调用，内部还包裹了一个真正执行更新的self.local_optimizer

__init__ 负责标注rank， world_size，维护owner映射，同步参数的时候才知道谁是原始的rank
dist.broadcast(param, src = owner)

保存本地参数组 self.local_param_groups 保存当前rank负责的参数

暂时内部没有optimizer, 因为在调用 super().__init__() 的时候，参数还没有分配完毕，所以在初始化的时候 self.local_optimizer = None


self.next_parameter_index 用于标记当前参数的索引，方便计算 owner

initializing 标记当前是否在初始化阶段，避免在初始化阶段调用 add_param_group 时创建 local_optimizer

因为 super().__init__() 内部会调用 self.add_param_group()，然后内部又会调用 add_param_group()，所以需要在初始化阶段避免创建 local_optimizer

等初始化结束之后，再把local_param_groups传给local_optimizer，创建真正的优化器实例

**add_param_group** 

首先复制完整参数组 full_group = param_group.copy()
不只是有参数，还有 lr 等。。

注册到外层 optimizer, super().add_param_group(full_group)
self.param_groups 保存完整参数

```
self.param_groups
    完整参数组
    所有 rank 都相同

self.local_optimizer.param_groups
    当前 rank 的参数 shard
    不同 rank 不同
```

然后给每个参数分配owner
self.parameter_owners[param] = owner

收集rank参数
if owner == self.rank:
    local_params.append(param)
随后更新index

local_group = full_group.copy()
local_group["params"] = local_params
保存了完整的参数，没有丢掉lr,weight_decay等信息

如果 initializing，就只有进入local_param_groups，等初始化结束之后再创建 local_optimizer
否则直接调用 self.local_optimizer.add_param_group(local_group)  就直接把参数组添加到 local_optimizer


step() 只更新本rank的参数，然后广播更新后的参数