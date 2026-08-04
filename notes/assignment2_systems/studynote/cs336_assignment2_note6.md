# Fully Shared Data Parallel

With optimizer state sharding and data parallel, we're able to split the optimizer state and activations across our data-parallel axis.

However, our model weights remain duplicated —— we're storing a full copy of them on each GPU.


We can solve this by turning our data parallel (DP) axis into a fully sharded data parallel axis (FSDP). With FSDP, each GPU stores only its own slice of every weight tensor, but has to pull slices from other GPUs to form the full weight tensor using an all-gather to prepare for a forward or backward pass.

To avoid keeping GPU compute waiting around for communication to finish , most FSDP implementations schedule the layer's all-gather in advance of the operation, meaning the relevant weights are ready before they are needed, preventing communication from blocking computation. This keeps weight sharding communication off the critical path, meaning it has no cost as long as communication can keep up with compute and si scheduled well.


Some layers are small enough in memory and compute thta the lantency overhead of a transfer is not worth it. You should mark these layers not to be sharded by FSDP. In our architecture,this will mostly be the case for norms. This leaves us with the embedding layer and every linear layer.

While it is necessary to store master weights in FP32 (any values that are repeatedly accumulated into are sensitive to precision), the weights do not need to be used in FP32. In mixed precision,we always convert to the low-precision compute datatype before use, so we may as well convert even before the weight is communicated to save on bandwidth.


**Problem: Fully-Shared Data Parallel (FSDP)**

Implement a Python class for fully-shared data parallel training. The class should warp an arbitrary PyTorch nn.Module and hook into or warp any Linear or Embedding layer within it.

```py

def __init__(self, module: nn.Module,compute_dtype:torch.dtype | None = None)
"""
Given an instantiated PYtorch nn.Module to be parallelized, consruct an FSDP module that will handle weight all-gather the weights in time for the forward pass. To limit memory use, only start gathering after the layer two before the current one has completed its forward pass. In the backward pass, your hooks or module warppers should all-gather to have the weights available for the computation.When the gradients are available,they should be reduce-scatterd to the appropriate ranks. Make sure to free the gathered weights after use.
When `compute_dtype` is provided,cast the weights to that dtype before communicating or using them for compute , while keeping master weights and potimizer updates in FP32.
"""


def forward(self, *inputs, **kwargs)

def finish_gradient_synchronization(self)
```

```
阶段 1：纯参数分片
完整 weight → padding → local shard → 模拟重建
不涉及模型，不涉及 distributed

阶段 2：单个 ShardedLinear 的同步 forward
local shard → all-gather → F.linear
暂时不优化显存和 backward

阶段 3：自定义 autograd backward
反向重新 all-gather weight
计算 full grad_weight
reduce-scatter → local shard.grad

阶段 4：Embedding + 递归替换模型中的层
FSDP 包装任意 Module
AdamW 能看到并更新 local shards

阶段 5：异步通信和两层提前预取
finish_gradient_synchronization()
compute_dtype
释放临时 full weights
```

## 阶段1

```py

@staticmethod
def shard_weight(
    full_weight: torch.Tensor,
    rank:int,
    world_size:int,
    ):
    # 转fp32 后 flatten
    full_flat = full_weight.detach().to(torch.float32).flatten()
    
    original_shape = full_weight.shape
    original_numel = full_weight.numel()
    
    shard_numel = math.ceil(original_numel / world_size)
    
    padded_numel = shard_numel * world_size
    
    # 创建长度为 padded_numel 的 FP32 0 张量
    padded_flat = torch.zeros(padded_numel, dtype=torch.float32, device=full_weight.device)
    
    #复制full_flat进来
    padded_flat[:original_numel] = full_flat
    
    start = rank * shard_numel
    end = start + shard_numel
    
    local_tensor = padded_flat[start:end].clone()
    
    local_shard = nn.Parameter(local_tensor)
    
    return local_shard,original_shape,original_numel,shard_numel,padded_numel
```

## 阶段2

```py

@staticmethod
def gather_full_weight(
    local_shard: torch.Tensor,
    original_shape: tuple[int,...],
    original_numel: int,
    shard_numel: int,
    padded_numel: int,
    compute_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
    # local_shard 长期保存的 FP32 nn.Parameter
    # full_weight 临时恢复的完整tensor
    
    assert local_shard.numel() == shard_numel, "local_shard numel does not match shard_numel"
    assert padded_numel == shard_numel * dist.get_world_size(), "padded_numel does not match shard_numel * world_size"
    
    
    if compute_dtype is not None:
        communication_shard = local_shard.detach().to(compute_dtype)
    else:
        communication_shard = local_shard.detach()
    
    #确保communication_shard连续
    communication_shard = communication_shard.contiguous()
    
    gathered_padded = torch.empty(padded_numel,
                                    device = local_shard.device,
                                    dtype = communication_shard.dtype)
    #所有rank执行 all-gather
    # gathered_padded 凭借美髯rank的communication shard
    dist.all_gather_into_tensor(
        gathered_padded,
        communication_shard,
    )
    
    full_flat = gathered_padded[:original_numel]
    
    full_weight = full_flat.reshape(original_shape)
    
    return full_weight
```

## 阶段3

```py
class FSDP(nn.Module):
    
    def __init__(
        self,
        module: nn.Module,
        compute_dtype: torch.dtype | None = None,
        ):
        super().__init__()
        self.module = module
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.compute_dtype = compute_dtype
        
        self.sharded_modules = []
        self.shard_metadata = {}
        #遍历 module.modules()，对每个子模块进行处理
        for submodule in module.modules():
            #如果是Linear or Embedding
            if isinstance(submodule, (Linear, Embedding)):
                #调用sharedweight
                local_shard, original_shape, original_numel, shard_numel, padded_numel = self.shard_weight(
                    submodule.weight,
                    self.rank,
                    self.world_size,
                )
                # 用返回的local_shard 替换module.weight
                submodule.weight = local_shard
                self.sharded_modules.append(submodule)
                
                #保存元数据
                self.shard_metadata[submodule] = {
                    "original_shape": original_shape,
                    "original_numel": original_numel,
                    "shard_numel": shard_numel,
                    "padded_numel": padded_numel,
                    "local_shard_data": local_shard.detach(),  # 保存本地shard的副本
                }
```

这个函数用来方便的从submodel恢复到完整的权重
```py
def gather_module_weight(
    self,
    submodule,
):
    #从shard_metadata中获取元数据
    metadata = self.shard_metadata[submodule]
    original_shape = metadata["original_shape"]
    original_numel = metadata["original_numel"]
    shard_numel = metadata["shard_numel"]
    padded_numel = metadata["padded_numel"]
    
    return self.gather_full_weight(
        submodule.weight,
        original_shape,
        original_numel,
        shard_numel,
        padded_numel,
        compute_dtype=self.compute_dtype,
    )
```
## 阶段4
利用python hook

```py

#从 shard 恢复
    def _unshard_module(
        self,
        submodel,
    ):
        
        full_weight = self.gather_module_weight(submodel)
        
        submodel.weight.data = full_weight
    
    
    #恢复本地shard
    def _reshard_module(
        self,
        submodel,
    ):
        metadata = self.shard_metadata[submodel]
        
        submodel.weight.data = metadata["local_shard_data"]
        
    # forward hook自动调用
    def _pre_forward_hook(
        self,
        submodule,
        inputs,
    )-> None:
        self._unshard_module(submodule)
    
    def _post_forward_hook(
        self,
        submodule,
        inputs,
        outputs,
    )-> None:
        self._reshard_module(submodule)
```

__init__ 里面增加
```py
self.forward_hook_handles = []
        
        for submodule in self.sharded_modules:
            pre_handle = submodule.register_forward_pre_hook(self._pre_forward_hook)
            post_handle = submodule.register_forward_hook(self._post_forward_hook)
            
            self.forward_hook_handles.extend(
                [pre_handle, post_handle]
            )
```
这样forward 只用 return self.module(*input, **kwargs) 
就会自动
```
进入某个 Linear / Embedding
    ↓ pre_forward_hook
all-gather 完整 weight
    ↓
执行原模块 forward
    ↓ post_forward_hook
恢复 local shard
```


## 阶段5

增加backward hook

```py

def _pre_backward_hook(
        self,
        submodule: nn.Module,
        grad_output: tuple[torch.Tensor | None, ...],
    ) ->None:
        self._unshard_module(submodule)
        
    def _post_accumulate_grad_hook(
        self,
        parameter:torch.Tensor,
    )-> None:
        submodule = self.parameter_to_module[parameter]
        
        metadata = self.shard_metadata[submodule]
        
        if parameter.grad is None:
            return
        
        # 当前是完整权重的完整梯度
        full_grad = parameter.grad.detach().flatten().contiguous()
        
        # padding 到 worldsize可以整除的长度
        
        padded_grad = torch.zeros(
            metadata["padded_numel"],
            dtype = full_grad.dtype,
            device = full_grad.device,
        )
        padded_grad[:metadata["original_numel"]].copy_(full_grad)
        
        # 当前rank接受自己的梯度shard
        local_grad = torch.empty(
            metadata["shard_numel"],
            dtype = full_grad.dtype,
            device = full_grad.device,
        )
        
        handle = dist.reduce_scatter_tensor(
            local_grad,
            padded_grad,
            op = dist.ReduceOp.SUM,
            async_op = True,
        )
        
        # 完整梯度已经传出去，不长期保存
        parameter.grad = None
        
        # 参数恢复为local shard
        self._reshard_module(submodule)
        
        # padded_grad 保存
        self.pending_gradients.append(
            (handle, parameter, local_grad,padded_grad)
        )
```

init里面
```py

#前面for submodule 加一行
#记录对应关系
                self.parameter_to_module[submodule.weight] = submodule



self.backward_hook_handles = []
        
for submodule in self.sharded_modules:
    handle = submodule.register_full_backward_pre_hook(
        self._pre_backward_hook
    )
    self.backward_hook_handles.append(handle)

for submodule in self.sharded_modules:
    if submodule.weight.requires_grad == False:
        continue
    handle = submodule.weight.register_post_accumulate_grad_hook(
        self._post_accumulate_grad_hook
    )
    self.gradient_hook_handles.append(handle)
```

补上一些别的，这个地方代码比较长debug比较难，留一份笔记吧

```
长期状态：
每个 rank 只保存 FP32 local weight shard
optimizer 也只管理这个 shard

Forward：
local shard
→ all-gather
→ 临时 full weight
→ 计算
→ 释放 full weight，恢复 local shard

Backward：
local shard
→ 再次 all-gather
→ 临时 full weight
→ 计算完整梯度
→ reduce-scatter
→ 得到 local gradient shard
→ 释放 full weight

Optimizer：
local shard + local grad shard
→ 本地更新
```
至于 RMSNorm 这种不分片的小参数：

每个 rank 都保存完整参数
→ backward 后做普通 all-reduce


代码职责
```
shard_weight
    初始化时把完整权重切成 local shard

gather_full_weight
    把所有 rank 的 shard 拼回完整权重

forward hooks
    forward 前 gather，forward 后 reshard

backward pre-hook
    backward 开始前重新 gather

post-accumulate grad hook
    完整梯度产生后执行 reduce-scatter

finish_gradient_synchronization
    等待异步通信，将 local grad 交给 optimizer
```


用一层 Linear看的例子
```
调用 linear(x)

1. PyTorch 自动调用 forward_pre_hook
   → all-gather 完整 weight

2. PyTorch 调用 linear.forward(x)
   → 得到 output

3. PyTorch 自动调用 forward_hook
   → 恢复 local shard


调用 loss.backward()

4. 反向传播到达 Linear
   → 自动调用 full_backward_pre_hook
   → 再次 all-gather 完整 weight

5. autograd 计算：
   grad_input
   grad_weight

6. grad_weight 累积进 parameter.grad 后
   → 自动调用 post_accumulate_grad_hook
   → reduce-scatter
   → 恢复 local weight shard

7. finish_gradient_synchronization()
   → 等待异步通信完成
   → 设置 local shard.grad

8. optimizer.step()
   → 更新 local weight shard
```


**Problem FSDP accounting**
(a) how much memory do you expect to save from the peak by implementing FSDP?
You can ignore the size of the preallocated buffers needed to all-gather weights to each GPU in your calculation
(b) Profile the xl model on two GPUs and pay attention to the all-gather of weights.Does the communication finish in time for the forward pass?

(a) Moss = 2P + 2P/N  

额外节省为 2P(1 - 1/N) = 2P(N-1)/N 

带入双卡xl模型，为每张GPU 12.69GiB

(b) 当前同步 hook 实现中的 all-gather 位于每个 Linear/Embedding forward 的 critical path，因此当前层计算会等待通信完成，不能达到题目所期望的通信隐藏效果。要完成该实验，需要在双 GPU 环境下加入异步两层预取，然后使用 Nsight 比较 NCCL all-gather 的结束时间和相应层 GEMM 的开始时间。