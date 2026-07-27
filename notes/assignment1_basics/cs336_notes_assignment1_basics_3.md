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