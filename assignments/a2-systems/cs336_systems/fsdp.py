from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from torch import nn
import math
from cs336_basics.model import Linear,Embedding

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
        
        self.pending_gradients = []
        self.gradient_hook_handles = []
        self.parameter_to_module = {}
        
        self.pending_replicated_gradients = []
        self.replicated_gradient_hook_handles = []
        
        
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
                
                #记录对应关系
                self.parameter_to_module[submodule.weight] = submodule
                
                
                #保存元数据
                self.shard_metadata[submodule] = {
                    "original_shape": original_shape,
                    "original_numel": original_numel,
                    "shard_numel": shard_numel,
                    "padded_numel": padded_numel,
                    "local_shard_data": local_shard.detach(),  # 共享storage的别名
                }
        
        self.forward_hook_handles = []
        
        for submodule in self.sharded_modules:
            pre_handle = submodule.register_forward_pre_hook(self._pre_forward_hook)
            post_handle = submodule.register_forward_hook(self._post_forward_hook)
            
            self.forward_hook_handles.extend(
                [pre_handle, post_handle]
            )
        
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
        
        sharded_parameters = set(self.parameter_to_module.keys())
        
        for parameter in self.module.parameters():
            
            # 已经通过reduce-scatter处理
            if parameter in sharded_parameters:
                continue
            # 冻结参数不同步
            if not parameter.requires_grad:
                continue
            
            handle = parameter.register_post_accumulate_grad_hook(
                self._post_accumulate_replicated_grad_hook
            )
            self.replicated_gradient_hook_handles.append(handle)
        
        
    def forward(
        self,
        *inputs: Any,
        **kwargs: Any,
    ):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        # Linear / Embedding
        for handle,parameter,local_grad,_padded_grad in self.pending_gradients:
            handle.wait()
            
            # SUM 转换为平均值
            local_grad.div_(self.world_size)
            
            # local master parameter是FP32 梯度转回FP32
            parameter.grad = local_grad.to(dtype = parameter.dtype)
        
        self.pending_gradients.clear()
        
        # RMSNorm
        for handle,parameter,gradbuffer in self.pending_replicated_gradients:
            handle.wait()
            
            gradbuffer.div_(self.world_size)
            
            parameter.grad = gradbuffer
        self.pending_replicated_gradients.clear()
    
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
        
        local_shard = nn.Parameter(local_tensor,requires_grad= full_weight.requires_grad)
        
        return local_shard,original_shape,original_numel,shard_numel,padded_numel
    
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
    
    def gather_module_weight(
        self,
        submodule,
        *,
        use_compute_dtype: bool = True,
    ):
        #从shard_metadata中获取元数据
        metadata = self.shard_metadata[submodule]
        
        gather_dtype = (
            self.compute_dtype
            if use_compute_dtype
            else None
        )
        
        
        
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
            compute_dtype=gather_dtype,
        )
    
    #从 shard 恢复
    def _unshard_module(
        self,
        submodule   ,
    ):
        
        full_weight = self.gather_module_weight(submodule)
        
        submodule.weight.data = full_weight
    
    
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
        
    def _post_accumulate_replicated_grad_hook(
        self,
        parameter: torch.Tensor,
    )-> None:
        if parameter.grad is None:
            return
        
        grad_buffer = parameter.grad
        
        handle = dist.all_reduce(
            grad_buffer,
            op = dist.ReduceOp.SUM,
            async_op = True,
        )
        
        # 保存grad_buffer
        self.pending_replicated_gradients.append(
            (handle, parameter, grad_buffer)
        )
        
    def gather_full_params(self) -> dict[str, torch.Tensor]:
        
        full_params: dict[str, torch.Tensor] = {}

        # 必须遍历底层模型，避免名字出现 "module." 前缀
        for name, parameter in self.module.named_parameters():

            # Linear / Embedding 的本地 shard
            if parameter in self.parameter_to_module:
                submodule = self.parameter_to_module[parameter]

                full_parameter = self.gather_module_weight(
                    submodule,
                    use_compute_dtype=False,
                )

            # RMSNorm 等 replicated parameters
            else:
                full_parameter = parameter

            full_params[name] = full_parameter.detach().clone()

        return full_params