# 本节笔记为 Part4  Training a Transformer LM


We now have the stpes to preprocess the data and the model.

What remains is to build all of the code to support training.

This consists of the following:

- **Loss**: We need to define the loss function
- **Optimizer**: We need to define the optimizer to minimize this loss
- **Training Loop**: We need all the supporting infrasturcture that loads data,saves checkpoints, and manages training.

## 4.1 Cross-entropy loss

我们写好了的 Transformer LM 返回 logits
```
logits :(... , vocab_size)
```
目标token是
```
targets: (...)
```
例如语言模型训练中:
```
inputs: [the,cat,sat]
targets: [cat,sat,down]
```
对某个位置，设模型输出logits
```
o = (o_1, o_2, ..., o_v)  # v = vocab_size
```
真实token的id是 y， 那么交叉熵为
```
l = -log(softmax(o)_y)
```
展开

$$ l = -log(\frac{exp(o_y)}{\sum_i exp(o_i)}) = -o_y + \log(\sum_i exp(o_i)) $$

为什么不能先softmax再log


直接计算softmax(logits)

- 较大的logits会导致exp(logits)溢出，得到inf，softmax结果为nan
- 下溢
- log(0) = -inf

作业要求
- 减去最大值
- 消去log和exp
- 支持任意前导 batch-like dimensions
- 对所有位置取平均

**稳定形式**

```py

m = torch.max(logits, dim=-1, keepdim=True).values
logits = logits - m
softmax = torch.exp(logits) / torch.sum(torch.exp(logits), dim=-1, keepdim=True)
log_softmax = logits - torch.log(torch.sum(torch.exp(logits), dim=-1, keepdim=True))
```

shape流程

假设
```
logits: (batch, seq_len, vocab_size)
targets: (batch, seq_len)
```

依次得到
```
max_logits: (batch, seq_len, 1)
shifted_logits: (batch, seq_len, vocab_size)
log_partition: (batch, seq_len)
target_logits: (batch, seq_len)
loss_per_token: (batch, seq_len)
final_loss: scalar
```

实现见
`cs336_codeNote13_crossentropyLoss.md`


## 4.2 The SGD optimizer

The simplest gradient-based optimizer is Stochastic Gradient Descent

We start with randomly initialized parameters . Then for each step t = 0,...,T-1, we perform the following update:

$$ \theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t;B_t) $$

where $\eta$ is the learning rate, and $L(\theta)$ is the loss function,$B_t$ is the batch of data at step t.


### 4.2.1 Implementing SGD in PyTorch

We will subclass the Pytorch torch.optim.Optimizer class to implement our own SGD optimizer.

```py

def __init__(self, params,...) 
```
should initialize your optimizer.

Here , params will be a collection of parameters to be optimized.

Make sure to pass params to the init method of the base class , which will store these parameters for use in step.

```py
def step(self)
```

should make one update of the parameters. During the training loop, this will be called after the backward pass , so you have access to the gradients on the last batch.

This method should iterate through each parameter tensor p and modify them in place.


代码示例

```py

from collections.abc import Callable, Iterable

from typing import Optional

import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr = 1e-3,
    ):
      if lr < 0:
          raise ValueError(f"Invalid learning rate: {lr}")
      defaults = {"lr": lr}
      super().__init__(params, defaults)

    
    def step(self,closure: Optional[Callable] = None):
      loss = None if closure is None else closure()
      for group in self.param_groups:
          lr = group["lr"] # Get the learning rate for this group of parameters
          for p in group["params"]:
              if p.grad is None:
                  continue

              state = self.state[p] # get state associated with this parameter
              t = state.get("t", 0) # get iteration number from the state, or 0.
              grad = p.grad.data # Get the gradient of loss with respect to p.
              p.data -= lr / math.sqrt(t + 1) * grad # Update the parameter in place.
              state["t"] = t + 1 # increment the iteration number for this parameter

      return loss
```

## 4.3 AdamW
![alt text](image.png)

代码见
`cs336_codeNote14_adamW.md`


**Problem_ Adamw_accounting:Resource accounting for training with AdamW**

(a): How much peak memory does running AdamW require? 

设 B = batch size
T = context length
L = number of layers
D = d_model
H = number of attention heads
F = feedforward dimension
V = vocab size

模型参数
P = 2VD + L(4D^2 + 3DF + 2D) + D

题目令 F = 8/3 D
所以 P = 2VD + L(12D^2 + 2D) + D

FP32下: 于batch无关的固定部分 16P bytes

**activation**
| 中间结果                        |      元素数 |
| --------------------------- | -------: |
| 两次 RMSNorm                  |   (2BTD) |
| Q、K、V                       |   (3BTD) |
| (QK^\top) 和 Softmax         | (2BHT^2) |
| Attention weighted sum      |    (BTD) |
| Attention output projection |    (BTD) |
| SwiGLU 两个输入投影、SiLU、逐元素乘法    |   (4BTF) |
| SwiGLU 输出投影                 |    (BTD) |

单个BLOCK
Ablock = 8BTD + 2BHT^2 + 4BTF

模型末尾再算
- final RMSNorm: BTD
- LM head logits: BTV
- cross-entropy loss: BTV

所以activation总数
A = L * Ablock + BTD + 2BTV = L(8BTD + 2BHT^2 + 4BTF) + BTD + 2BTV

activation内存 4A bytes

总量为 16P + 4A bytes

(b): GPT2-XL
带入GPT2-XL参数

P = 1,640,452,800

16P = 26.25 GB

每增加一个batch element， activation增加
16.37 GB

所以80GB内顶多3个batch element

(c) adamw 本身的FLOPS
| 操作            | FLOPs/parameter |
| ------------- | --------------: |
| Weight decay  |               2 |
| 更新 (m)        |               3 |
| 更新 (v)        |               4 |
| 根据 (m,v) 更新参数 |               5 |

Fadamw = 14P

所以对于GPT2-XL，Fadamw = 22.97 GFLOPs

(d) H100训练时间

前面已经算出来1024的样本 forward约为 3.5168TFLOPS

假设backward是2倍的forward FLOPS, 那么总共是 3.5168 * 3 = 10.5504 TFLOPS

再假设 batch size 1024, 那么每个样本的FLOPS为 10.5504 / 1024 = 10.3 GFLOPS

400k steps: Ftotal = 4.321 x 10^21 FLOPS
H100 理论 495 TLOPS/s, 50% MFU

对应 4850 hours, 即为 202days

## 4.4 Learning rate scheduling

实现一个函数，输入当前训练步数 t，输出这一刻AdamW的学习率

学习率分为3个阶段

- Warmup: if t < T_w then lr = t / T_w * lr_max
- Cosine annealing: if T_w <= t < T_c 

then lr = lr_min + 0.5 *( 1 + cos(pi * (t - T_w) / (T_c - T_w))) * (lr_max - lr_min)

- Post-annealing: if t >= T_c then lr = lr_min

这个就不写codenote了

```py
def cosine_lr_schedule(
    it:int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_steps: int,
    cosine_cycle_iters:int,
)-> float:
    
    if it < warmup_steps:
        return max_learning_rate * (it / warmup_steps)
    
    if it >= warmup_steps and it < cosine_cycle_iters:
        return min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (1 + math.cos(math.pi * (it - warmup_steps) / (cosine_cycle_iters - warmup_steps)))
    
    return min_learning_rate

```
翻译一下逻辑就行

## 4.5 Gradient clipping

During Training,we can sometimes hit training examples that yield large gradients,which can destabilize training.

To mitigate this , one technique often employed in practice is gradient clipping.

The idea is to enforce a limit on the norm of the gradient after each backward pass before taking an optimizer step.

Given the gradient (for all parameters) g, we compute its l2-norm,if this norm is less than a maximum value M,then we leave g as is; otherwise, we scale g down by a factor of M/||g||_2 + eps

Note that the resulting norm will be just under M.

这个也不难,直接笔记写这里

```py
@torch.no_grad()
def gradient_clipping(
    parameters: Any,
    max_l2_norm: float
)-> None:
    grads = [p.grad for p in parameters if p.grad is not None]
    
    total_squared_norm = sum(torch.sum(g ** 2) for g in grads)
    total_norm = math.sqrt(total_squared_norm)
    
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + 1e-6)
        
        for grad in grads:
            grad.mul_(scale)
```

torch.no_grad() 是为了在这个函数中不计算梯度，避免影响训练过程。

然后首先先从 parameters 中提取出所有非 None 的梯度，计算它们的平方和，然后开方得到总的 l2 范数。

然后计算total_norm 是否大于 max_l2_norm，如果大于，就计算一个缩放因子 scale = max_l2_norm / (total_norm + 1e-6)，然后对每个梯度进行缩放。