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