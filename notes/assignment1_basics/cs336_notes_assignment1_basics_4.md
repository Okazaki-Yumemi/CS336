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


# 5.2 Checkpointing

In addition to loading data. 

We will also need to save models as we train. When running jobs, we often want to be able to resume a training run that stopped midway through. (e.g: due to your job timing out， machine failure, etc).

Even when all goes well, we might also want to later have access to intermediate models (e.g, to study training dynamics post-hoc, take samples from models at different stages of training, etc).

A checkpoint should have all the states that we need to resume training. We of course want to be able to restore model weights at a minimum. If using a stateful optimizer (Such as AdamW), we will also need to save the optimizer's state (e.g, in the case of AdamW, the moment estimates).

Finally, to resume the learning rate schedule,we will need to know the iteation number we stopped at.  PyTorch makes it easy to save all of these : every nn.Module has a state_dict() method that returns a dictionary with all lernable weights; we can restore these weights later with the sister method load_state_dict().

The same goes for any torch.optim.Optimizer.  Finally, torch.save(obj,dest) can dump an object (e.g., a dictionary containing tensors as some values, but also regular Python objects like integers) to a file(path) or file-like object, which can then be loaded back into memory with torch.load(src)


**Problem**:

Implement the following two functions to load and save checkpoints:

```py

def save_checkpoint(
  model: torch.nn.Module,
  optimizer: torch.optim.Optimizer,
  iteration: int,
  out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
)

def load_checkpoint(
  src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
  model: torch.nn.Module,
  optimizer: torch.optim.Optimizer,
)

```

save_checkpoint should dump all the state from the model,optimizer and iteration into the file-like object out.

You can use state_dict method of both the model and the optimizer to get their relevant states adnd use torch.save(obj,out) to dump obj into out(PyTorch supports either a path or file-like object here). 
A typical choice is to have obj be a dictionary, but you can use whatever format you want as long as you can load your checkpoint later.

load_checkpoint should load a checkpoint from src (path or file-like object) and then recover the model and potimizer states from that checkpoint. Your function should return the iteration number that was saved to the checkpoint. You can use torch.load(src) to recover what you saved in your save_checkpoint implementation, and the load_state_dict method in both the model and optimizer to return them to zhe previous states.

代码笔记

```py
def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)
    
def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
) -> int:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    iteration = checkpoint["iteration"]
    return iteration
```
很简单的两个啊，就是利用torch.save可以保存序列化文件的性质。因为考虑到字典是最方便的，所以就用字典了。


# Assignment 1 : Finished!