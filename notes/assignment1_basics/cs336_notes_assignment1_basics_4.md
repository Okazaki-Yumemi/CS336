# This note is for Chapter 5 : Training Loop

We will now finally put together the major components we've built so far: The tokenized data, the model , and the optimizer.


## 5.1 Data loader

The tokenized data (e.g that you prepared in tokenizer_experiments) is a single sequence of tokens x = (x1,x2,...,xn). 

Even though the source data might consist of separate documents (e.g., different web pages, or source code files), a common practice is to concatenate all of those into a single sequence of tokens, adding a delimiter between them (such as the <|endoftext|> token).

A data loader turns this into a stream of batchs, where each batch consists of B sequences of length m, paired with the corresponding next tokens,also with length m. For example, for B = 1 , m = 3 ,
([x2,x3,x4], [x3,x4,x5]) would be one potential batch

Loading data in this way simplifies training for a number of reasons. First , any 1 <= i <= n-m gives a valid training sequence, so sampling training sequences is trivial. Since all training sequences have the same length, there's no need to pad input sequences, which improves hardware utilization (also by increasing batch size B). Finally,we also don't need to load the full dataset to sample training data,making it easy to handle large datasets that might not otherwise fit in memory.


**Problem**:

Write a function that takes a numpy array x (integer array with token IDs), a batch_size, a context_length and a PyTorch device string (e.g. "cpu" or "cuda:0") and returns a pair of tensorsLthe sampled input sequences and the corresponding next-token targets.
Both tensors should have shape (batch_size, context_length) containing token IDs, and both should be placed on the requested device.


不难,不写codenote了,就写在下面


```py

def data_loader(
    x: numpy.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    
    used_device = torch.device(device)
    
    num_samples = x.shape[0]
    indices = numpy.random.randint(0, num_samples - context_length, size=batch_size)
    
    x_batch = numpy.stack([x[i:i + context_length] for i in indices])
    y_batch = numpy.stack([x[i + 1 : i + context_length + 1] for i in indices])
    
    x_batch_tensor = torch.tensor(x_batch, device=used_device, dtype=torch.long)
    y_batch_tensor = torch.tensor(y_batch, device=used_device, dtype=torch.long)
    
    return x_batch_tensor, y_batch_tensor
```

例子就是，现在假设 

```py

x = [10,21,35,48,52,67,71,83,96,99]
```

假设
```py
context_length = 4
```
那么一个合法样本需要取5个token,前4个作为输出，后面4个作为目标

例如起点 i = 2
```py
x_batch = [35,48,52,67]
y_batch = [48,52,67,71]
```

**`indices`代表什么**:

```py

indices = numpy.random.randint(
  0,
  num_samples - context_length,
  size=batch_size
)
```
是
> 每个训练序列在整个token起始的位置

假设batch = 3
indices可能随机得到 [2, 0, 5]

那么本轮batch 取3个窗口，就有可能从2，0，5开始。

x_batch (3,4) = [[35,48,52,67],[10,21,35,48],[67,71,83,96]]
y_batch (3,4) = [[48,52,67,71],[21,35,48,52],[71,83,96,99]]


然后stack做的是从i开始，切片，堆叠成目标batch

最后把batch转成tensor，放到指定device上。