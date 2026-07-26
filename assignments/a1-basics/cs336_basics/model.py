import math
from typing import Any

import torch
from einops import einsum
from torch import nn

class Linear(nn.Module):
    def __init__(
        self,
        in_features:int,
        out_features:int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        
        self.sigma = math.sqrt(2 / (self.in_features + self.out_features))

        # 创建shape为(out_features, in_features)的权重张量，并使用正态分布初始化
        self.weight = nn.Parameter(
            torch.empty(
                self.out_features,
                self.in_features,
                device= self.device,
                dtype= self.dtype,
            )
        )
        
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std= self.sigma,
            a = -3*self.sigma,
            b = 3*self.sigma,
        )
        
    def forward(self,
                x: torch.Tensor
        ) -> torch.Tensor:
        output = einsum(x, self.weight, "... d_in , d_out d_in -> ... d_out")
        return output
        
class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings : int,
        embedding_dim : int,
        device: torch.device | None = None,
        dtype : torch.dtype | None = None
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        
        self.weight = nn.Parameter(
            torch.empty(
                self.num_embeddings ,
                self.embedding_dim,
                device= self.device,
                dtype= self.dtype
                )
            )
        nn.init.trunc_normal_(
            self.weight,
            std= 1,
            a = -3,
            b = 3,
        )
    
    def forward(
        self,
        token_ids:torch.Tensor
    ) -> torch.Tensor:
        return self.weight[token_ids]
        
        
class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model:int,
        eps:float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.eps = eps
        
        self.weight = nn.Parameter(
            torch.ones(
                self.d_model,
                device= device,
                dtype=  dtype,
            )
        )
        
    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        in_dtype = x.dtype
        
        x_float = x.to(torch.float32)
        
        mean_square = torch.mean(x_float ** 2, dim=-1, keepdim=True)
        
        inverse_rms = torch.rsqrt(mean_square + self.eps)
        
        normalized_x = x_float * inverse_rms
        
        x_original = normalized_x.to(in_dtype)
        return x_original * self.weight
    
class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model:int ,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        
        self.w1 = Linear(d_model,d_ff,device,dtype)
        self.w2 = Linear(d_ff,d_model,device,dtype)
        self.w3 = Linear(d_model,d_ff,device,dtype)
        
        
    def forward(
        self,
        in_features: torch.Tensor
    ) -> torch.Tensor:
        a = self.w1(in_features)
        b = self.w3(in_features)
        silu_a = a * torch.sigmoid(a)
        h = silu_a * b
        y = self.w2(h)
        
        return y
    
class RoPE(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k :int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        
        assert d_k % 2 == 0, "d_k must be even for RoPE"
        
        mid_list = torch.arange(0, d_k , 2 , device=device,dtype=torch.float32)
        exponents = mid_list / (d_k)
        
        inv_freq = torch.pow(theta, -exponents)
        
        positional_indices = torch.arange(max_seq_len, 
                                          device=device,
                                          dtype=torch.float32,
                                          )[:, None]
        
        angles = positional_indices * inv_freq
        
        cos_cache = torch.cos(angles)
        sin_cache = torch.sin(angles)
        
        self.register_buffer("cos_cache", cos_cache, persistent=False)
        self.register_buffer("sin_cache", sin_cache, persistent=False)
        self.cos_cache: torch.Tensor
        self.sin_cache: torch.Tensor
        
    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor
    ) -> torch.Tensor:
        #查找对应位置的cos和sin值
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]
        #将x拆分为偶数和奇数索引的部分
        x1, x2 = x[..., ::2], x[..., 1::2]
        
        # 二维旋转，交错拼回
        x_rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        x_rotated = x_rotated.flatten(-2)
        
        return x_rotated
        