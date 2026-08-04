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

**Problem**: Optimizer state Sharding Accounting
(a) Create a script to profile the peak memory usage when training language models with and without optimizer state sharding. Using the standard configuration(1 node, 2 GPUs, xl model size), report the peak memory usage after model initialization.

(b) How does our implementation of optimizer state sharding affect training speed? Measure the time taken per iteration with and without optimizer state sharding for the standard configuration(1 node, 2 GPUs, xl model size).

(c) How does our approach to optimizer state sharding differ from ZeRO stage 1
(described as ZeRO-DP $P_{os}$ in the paper)


## 1.先定义参数内存P
```
vocab_size = 10000
d_model = 2560
d_ff = 10240
num_layers = 32
```

参数数量:

$$ N_{params} = 2VD + L(4D^2 + 3D D_{ff} + 2D) + D $$

代入

$$ N_{params} = 3.4068 x 10^9$$

3.41B参数

假设全部是FP32, 大小为 12.69GiB

## 2.AdamW 到底保存了什么

| 内容            |    每个参数 |
| ------------- | ------: |
| 模型参数 (\theta) | 4 bytes |
| 梯度 (g)        | 4 bytes |
| 一阶矩 (m)       | 4 bytes |
| 二阶矩 (v)       | 4 bytes |

所以AdamW 稳定持久内存为 4P

参数、梯度和两个 optimizer states

## 3.三个时间点的内存为什么不同

PyTorch AdamW 的 optimizer state 通常是惰性初始化的，构造optimizer 时不会立即分配 m和v，而是第一次step() 的时候创建。

因此题目才要求在模型初始化之后测量内存峰值，而不是在第一次step()之后。


初始化后只有完整模型参数，理论为 12.69GiB

第一次optimizer step之前，forward 和 backward 已经完成，拥有完整参数和梯度

大小为 2P = 25.38GiB

第一次 optimizer step 之后，创建2个states，大小为 4P = 50.76GiB

而双卡的话，每个rank只保存一半 optimizer state, 每个rank的内存是 P + P + 2P/2 = 3P = 38.07GiB

| 每个 rank          |    参数 |    梯度 | Adam states |        合计 |
| ---------------- | ----: | ----: | ----------: | --------: |
| 普通 AdamW         | 12.69 | 12.69 |       25.38 | 50.77 GiB |
| 双卡 sharded AdamW | 12.69 | 12.69 |       12.69 | 38.07 GiB |

一般化到N个rank

Moss = P + P + 2P/N 当 P -> ∞，最小还是2P


## 4.与 ZeRO Stage 1 的主要差异

分片粒度是整个parameter

如果参数差异很大，即使每张卡分到相同的参数数量，内存占用也可能不均衡

我们的是逐参数broadcast

ZeRO Stage 1  通常会将更新后的参数shard 进行 bucketed/coalesced all-gather,而不是每个参数单独发一个collective

工业ZeRO还有bucket\异步\通信调度

- flatten optimizer states..