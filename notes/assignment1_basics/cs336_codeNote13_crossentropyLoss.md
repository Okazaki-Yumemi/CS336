# Implement cross-entropy

Write a function to compute the cross-entropy loss, which takes in predicted logits and targets and computes the cross-entropy li = -log(softmax(logits)_y) 

- subtract the largest element for numerical stability
- cancel out log and exp whenever possible
- Handle any additional batch dimensions and return the average across the batch. As with Section 3.2 , We assume batch-like dimensions always come first. before the vocabulary size dimension.

```py
def cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
)-> torch.Tensor:
    # logits shape: (batch_size, seq_len, vocab_size)
    # targets shape: (batch_size, seq_len)
    
    # Step 1: Subtract the max for numerical stability
    max_logits = logits.max(dim=-1, keepdim=True).values
    stable_logits = logits - max_logits
    
    log_partition = torch.logsumexp(stable_logits, dim=-1)
    target_logit = torch.gather(stable_logits, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
   
    loss_per_item = log_partition - target_logit
   
   
   
    return loss_per_item.mean()
```

test均通过


复盘

max_logits 是用来稳定数值的

dim=-1 是因为 vocab_size 是最后一维

keepdim=True 是为了保持维度一致，方便后续计算，让沿着维度相减，stable_logits每一行最大值为0

第二步

计算log partition
用logsumexp是最方便的，直接对每一行计算 log + sum + exp

输入(N,V) 输出(N,)

第三步

取得正确类别的logit

```py

target_logit = torch.gather(
  stable_logits,
  dim = -1,
  index = targets.unsqueeze(-1),
).squeeze(-1)
```

假设
```py
stable_logits = torch.tensor([[0.0, -1.0, -2.0], [-2.5, 0.0, -2.0]])
```
目标 targets = torch.tensor([0,2])

我们想取得第一行第0个元素，第二行第2个元素 -2.0

`unsqueeze(-1)` 是为了把 targets 从 (N,) 变成 (N,1)，这样可以在 gather 时指定每一行的索引

`gather(dim = -1)` 

```py

torch.gather(stable_logits, dim= -1, index = tensors([[0],[2]]))
```
意思是沿着最后一维，第0行取第0个元素，第1行取第2个元素，得到
```py
tensor([[0.0], [-2.0]])
```

squeeze(-1) 是为了把结果从 (N,1) 变回 (N,)

第四步

```
loss_per_item = log_partition - target_logit
```

然后求平均值

