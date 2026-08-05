# Analyzing Parallelism Strategies

- Data Parallelism —— Batches of data are split across multiple devices , and each device computes gradients for its own batch. Gradients are then averaged across devices.

- Fully-shared Data Parallelism ——  On top of data parallelism, we also split optimizer states , gradients, and weights across devices to reduce memory usage. Devices then need to gather weight shards from other devices during the forward and backward pass.

- Tensor Parallelism —— Weight matrices are sharded across the input or output dimension. Devices compute the activations corresponding to their shard, and activations are then reduced or gathered across devices.

- Pipeline Parallelism —— The model is split layerwise into multiple stages, where each stage is run on a different device. Each device computes the output for its own expert.


## 8.1 Communication Primitives

Our first step will be to understand the communication primitives.

In our simplified setting,suppose we have N devices numbered 0,....N-1, and each pair of devices is connected by a link. We'll also assume each device has W egress (i.e outgoing) bandwidth; in other words, each device can send data to another device at a rate of W egress bandwidth; in other words, each device can send data to another device at a rate of W bytes per second. How might we implement gather and reduce?


One common way to implement the all-gather operation is the ring all-gather. Recall that in an all-gather,each device i strats with a chunk x_i of size S/N, and ends up with the entire x = [x_0, x_1, ..., x_{N-1}]. 

In a ring all-gather , we arrange the devices in a circle. In each step , each device sends its current chunk to the next device to its right, and stores the chunk it received from the device to its left. This process repeats, where each device passes the chunk it just received to the right.and receives a new chunk from the left.

After N-1 steps, each device has the entire tensor.


In our idealized setting, each device simultaneously transmits a chunk of size S/N in each step,with egress bandwidth W, and there are N-1 steps, so the ring all-gather takes ((N-1)/N) * (S/W) seconds.

```
rank 0 → rank 1 → rank 2 → ... → rank N-1
   ↑                                  ↓
   └──────────────────────────────────┘
```


Next,let's analyze the ring reduce-scatter.  In a reduce-scatter, each device i starts with a full tensor x of size S. We then want to compute the reduction  y = sum(x) from i =0 to N-1, but where each device i ends up wit just a chunk y_i of size S/N.
We'll start by arranging the devices in a circle. Each device will first devide its tensor x into N chunks [X ~ Xn-1],each of size S/N.

We'll then pass chunks around just like the ring all-gather,except before passing the chunk on,each device adds its contribution to the chunk.

All reduce为什么是两倍？
```
第一步：reduce-scatter
每张卡得到求和结果的一个 shard

第二步：all-gather
把这些求和后的 shards 重新拼成完整 tensor
```

| 操作 | 开始时 | 结束时 | 典型用途 |
|---|---|---|---|
| all-gather | 每卡一个 shard | 每卡完整 tensor | FSDP 恢复完整权重 |
| reduce-scatter | 每卡完整 tensor | 每卡一个已规约 shard | FSDP 分片梯度 |
| all-reduce | 每卡完整 tensor | 每卡完整规约结果 | DDP 同步完整梯度 |

**Problem: Alternate ring All-reduce**

Instead pf implementing all-reduce as a ring reduce-scatter followed by a ring all-gather, let's use the following algorithm:

For step t = 1 ... n-1, device i does the following

- if t = 1, initialize y <- xi, which stores the partial sum so far.
- Send x((i-t+1)mod N) to device i+1 mod N
- receive x((i-t)mod N) from device i-1 mod N
- update your cpoy of partial sum y <—— y + x((i-t)mod N)


>不把 tensor 切成 chunk，而是让每张设备的整个 tensor 沿环传一圈，同时每个设备把收到的 tensor 加到自己的局部和里。这样要多久？

标准ring all-reduce 是 reduce-scatter + all-gather, 每一轮只发送一个大小为 S/N 的chunk，总时间是 2(N-1)/N * S/W

而 alternate ring all-reduce 每一轮发送整个 tensor，总时间是 (N-1) * S/W

设备越多，alternate ring all-reduce 的时间复杂度越高，因为每轮传输的数据量是整个 tensor，而不是分块后的数据。标准的 ring all-reduce 通过分块传输减少了每轮的数据量，从而提高了效率。

## 8.2 Analyzing Data Parallel

> 把一个全局batch 切给 N_DP 张卡之后，每张卡的计算量会下降; 但梯度同步的通信量不会按同样速度下降。 究竟拓展到多少张卡之后，通信会超过计算，成为瓶颈?

为了避免整套 Transformer 太复杂，只分析一个SwiGLU FFN 层

x shape (B,D) batch size B, hidden dimension D, D_ff 中间维度， 权重W1 W2 DxDff   W3 Dff x D

前向传播 x_1 = xW1 , x_2 = xW2 |  z = f(x_1) * x_2 , y = zW3

结构上
```
                 ┌─ W1 ─→ x1 ─→ activation ─┐
x ───────────────┤                          × ─→ z ─→ W3 ─→ y
                 └─ W2 ─→ x2 ───────────────┘
```
为什么要先写出 backward
1. 后面要计算 backward 的 FLOPs 和通信量
2. backward通信梯度
3. 计算通信对比

dy (B,D)   先通过 dz = dyWT  

再穿过门控 dx2 = dz f(x1)   |  dx1 = dz f'(x1) x2 |  最后 两条路径对输入 x的梯度相加  dx =  dx1 W1T + dx2 W2T

权重梯度  dW3 = zT dy  |  dW2 = xT dx2  |  dW1 = xT dx1 


**Data Parallel 到底切了什么**

DP不切权重，切 batch.  每个

x = [x_0, x_1, ..., x_{B-1}]  切成 N_DP 张卡，每张卡的 batch size = B/N_DP

forward不需要通信，每张卡都有完整权重，可以独立做 x-> y  ，各个rank的输出只是对应 不同的 batch shard。 不需要互相拼接才能训练.

**Backward 必须通信**

每张卡只看到了局部batch, 所有计算出的权重梯度只是局部贡献

backward后必须要对三个权重梯度做 all-reduce。  为什么是all-reduce？而不i是all-gather
- 每张卡已经有完整的 dW
- 里面只是部分和
- 还要求和
- 求和结果还必须出现在所有rank上，因为每张卡有完整optimizer state, 需要更新完整权重

**计算量**

(A B)(B C) -> (A C)  = 2ABC FLOPs

每张卡局部batch大小为

b = B/N_DP

计算activation gradient 。 计算dx 有两个乘法 dx1W1T + dx2W2T  每个乘法 2bD D_ff FLOPs  总共 4bD D_ff FLOPs

计算 weight gradient, dW3 = zTdy 为 2bD D flops， 每个w都这样，所以有6个

每张卡计算量随着 BDD_ff / Ndp 正比。

**通信量**:

同步权重 dW1 dW2 dW3 分别有 DDff, D Dff, DffD 参数，为 3DDff， 假设FP16，每个元素2bytes，总大小为 6DDff bytes

Ring All-reduce 是 2(N_DP-1)/N_DP * S / W seconds

设备数增加，计算时间近似按 1/N 下降，但是通信时间趋于常数。

**通信计算overlap**:
当后面某一层的梯度准备好后，可以立刻启动异步all-reduce

```
时间 →

后层 backward：       [compute]
后层 grad all-reduce:          [communication.......]
前层 backward：                [compute..............]
```


最终拿到的数据是 T_compute = 12BDD_FF / NdpC   T_communication = 12(N_DP - 1) D Dff / NdpW

设Tcommunication <= T_compute，解得 N_dp <= 1+ BW/C

- B 越大：每一步有更多计算，可以容纳更多 DP 设备；
- W 越大：网络越快，可以容纳更多设备；
- C 越大：GPU 算得越快，通信相对更容易成为瓶颈，因此可有效使用的 DP 设备反而更少。

## 8.3 Analyzing Fully Shared Data Parallel.

**FSDP 和 DP 的关键区别**:

DP中，每张设备长期保存完整权重

```
rank 0 : 完整 w1、w2、w3
rank 1 : 完整 w1、w2、w3
...
```

FSDP中，权重被分片

W(i)_k 表示设备i持有wk的一个shard，每个shard大小约为 DDff/N_FSDP

```
rank 0 : W1 的前一半
rank 1 : W1 的后一半
```


但这里有一个区别

FSDP不是tensor parallel, 矩阵乘法执行的时候，仍然需要完整权重。

计算前需要
```
各个rank 的 weight shards
       |all gather
       v
每个rank得到完整的weight
       |
       v
执行普通的batch-sharded matmul
```

FSDP只改变权重的长期存储方式，不改变单次矩阵乘法本身

**Forward发生什么**:

输入batch 切开，计算前需要恢复完整权重

W1 = all-gather(W1_shard)  W2 = all-gather(W2_shard)  W3 = all-gather(W3_shard)

每个rank对自己的batch shard做普通forward
```
all-gather W1 -> 使用W1 -> 丢弃W1
all-gather W2 -> 使用W2 -> 丢弃W2
all-gather W3 -> 使用W3 -> 丢弃W3
```


Forward计算和 DP完全相同，为 6BDDff / N_FSDP FLOPs

**backward 发生什么**:

forward之后为了节约显存，完整的权重已经被释放，只保留local shard

但是backward计算dx还需要权重，所以backward 前必须再次 all-gather

W1 W2 W3  <—— all gather

每个rank对自己的batch shard做普通backward，计算出局部权重梯度

但是每个rank最终只长期保存 W1 的一个shard，因此不需要完整全局 dW1


dW(i)1 = reduce-scatter({dW(r)1,local})

```
每个 rank：
    有完整形状的局部 dW

reduce-scatter：
    先跨 rank 求和
    再把最终梯度切成 shard

结果：
    每个 rank 只得到自己负责的 gradient shard
```

Backward计算和DP一样，为   12BDDff / N_FSDP FLOPs

**Forward通信时间**:

三个权重每个含 DDff 个FP16， 所以大小Sw = 2DDff

一次ring all-gather 需要 Nfsdp-1/Nfsdp * Sw / W 

三个权重就是三倍，所以总计  6(N_FSDP-1)/N_FSDP * DDff / W

**backward通信时间**:

重新all gather权重，为 6(N_FSDP-1)/N_FSDP * DDff / W

reduce-scatter梯度:， 三个梯度，每个也是2DDff bytes，一次reduce-scatter 与 一次 all-gather 的ring时间相同，三个梯度的reduce-scatter总计 6(N_FSDP-1)/N_FSDP * DDff / W

总计 12(N_FSDP-1)/N_FSDP * DDff / W


**对比**:

FSDP 的backward 和 DP的backward结果刚刚好，因为DP backward 使用一次完整gradient all-reduce = reduce-scatter + all-gather
而FSDP
```
weight all-gather
+
gradient reduce-scatter
```

通信时间刚好相同，但是要注意语义不同
```
DP：
    权重本来完整
    梯度需要 all-reduce

FSDP：
    权重需要 all-gather
    梯度只需要 reduce-scatter
```

**Forward 的边界**

Tcomm <= Tcompute 得到  Nfsdp <= 1 + BW/C

**backward 的边界**
Tcomm <= Tcompute 得到  Nfsdp <= 1 + BW/C

| 项目              | DP                  | FSDP                                        |
| --------------- | ------------------- | ------------------------------------------- |
| batch           | 分片                  | 分片                                          |
| weight          | 每卡完整                | 长期分片，计算前 gather                             |
| gradient        | 每卡完整                | 分片                                          |
| optimizer state | 每卡完整                | 分片                                          |
| forward 通信      | 无                   | weight all-gather                           |
| backward 通信     | gradient all-reduce | weight all-gather + gradient reduce-scatter |
| 每卡计算量           | 除以 (N)              | 除以 (N)                                      |

>FSDP 与 DP 的计算量相同；它通过把参数、梯度和 optimizer state 分片来省显存，但代价是 forward 和 backward 都需要临时 all-gather 权重。
而且整个训练 step 的通信量实际上比 DP 更多.

## 8.4 Analyzing Tensor Parallel

前面的 DP/FSDP 都是: 每张设备处理不同的数据，但某一次矩阵乘法仍然由单张设备完整完成。

Tensor Parallelism (TP 张量并行) 则进一步把一个矩阵乘法拆到多张设备上:

> 每张设备只持有权重矩阵的一部分，计算自己负责的输出激活，然后把激活拼接起来得到完整输出。

```
FSDP:
  权重平时分片
  计算前 all-gather 成完整权重
  每张设备执行完整matmul

TP:
  权重分配后不重新拼完整
  每张设备直接用自己的权重 shard 做一部分 matmul
```

考虑 x (B, D) W (D, Dff) 目的计算 y = xW

Column Parallel: 按输出维度切:  

把W列切开 W = [W0,W1...] 形状: W(i) (D, Dff/N_TP)

每个设备拿到完整的 x, 计算 y(i) = x W(i) 形状: (B, Dff/N_TP)

然后沿着最后一个维度 concat

因此， xW = all-gather({xW(i)})


Row Parallel: 按输入维度切:

把W的行切开 W = [W0,W1...]T 形状: W(i) (D/N_TP, Dff)

输入维度被切开，x也得切开 x = [x0,x1...]T 形状: x(i) (B, D/N_TP)

然后计算 yi = x(i) W(i) 形状: (B, Dff)

然后 all-reduce({yi}) 得到最终输出 y = sum(yi)

**为什么FFN要把两种分片配在一起**:

SwiGLU FFN是 
```

x1 = xW1
X2 = xW2
z = f(x1) * x2
Y = zW3
```

讲义选择 W1\ W2 按照 column parallel， W3 按照 row parallel

W1，W2 按照列切，每个rank得到的 x1, x2 形状都是 B x Dff/N_TP, 于是本地就能计算 z，不需要all-gather

W3 按照row切

W3 (DFF/N_TP, D)  Z 刚好是(B , DFF/N_TP) ，所以可以直接算，然后所有rank的结果相加。最后使用一次all-reduce

**为什么中间不需要all-gather**:

column parallel后，要把输出 shards all-gather

但是这下一层W3 是 row parallel，它本来就只需要对应的activation shard


**forward阶段**

rank 0 拿走 w1 w2 的前半列，w3的上半行。 rank 1 拿走 w1 w2 的后半列，w3的下半行。


**backward怎么推**:

给定完整上游梯度 (B,D) 

由于forward最后的y在所有rank上完整复制，所以每个rank都有完整的dy.

先经过W3,然后SwiuGLU门控，得到dx1, dx2, dz,最后计算W1 W2的梯度。

这里全程不需要通信，因为它们对应的就是 shard的完整梯度

**计算输入梯度**:

每张设备只能得到一部分贡献，需要对 dx进行all-reduce

TP的权重梯度不需要同步，因为他们并不是计算同一个参数的不同局部贡献，而是在计算不同参数的shard的梯度。

**计算量 accounting**:

每张卡只保存 1/N_TP的权重，计算量也缩小为 1/N_TP

Forward  6BDDff / N_TP FLOPs

backward 12BDDff / N_TP FLOPs

**通信量**:

TP通信的是activation,不是整个权重或者权重梯度

forward 的 all-reduce tensor:  y (B,D)  

FP16的大小: S = 2BD bytes

Ring all-reduce 需要 2(N_TP-1)/N_TP * S / W seconds

则
Tcomm fwd = 2(N_TP-1)/N_TP * 2BD / W
Tcomm bwd = 2(N_TP-1)/N_TP * 2BD / W

**边界计算**:

Tcompute,fwd = 6BDDff / N_TP C
通信
Tcomm,fwd = 4(N_TP-1)/N_TP * BD / W

得到 Ntp <= 1 + 3DffW / (2C)

backward
Ntp <= 1 + 3DffW / (C)

所以 forward 通常更容易先成为 TP 的通信瓶颈

| 方式   | 切什么        | 通信什么                     |
| ---- | ---------- | ------------------------ |
| DP   | batch      | weight gradients         |
| FSDP | batch和模型状态 | weights、weight gradients |
| TP   | 单个权重矩阵的维度  | activations              |

Tensor Parallelism 不像 FSDP 那样在计算前恢复完整权重，而是让各 rank 直接对权重 shard 执行部分矩阵乘法；通过把 column-parallel 层和 row-parallel 层配对，可以让中间 activation 一直保持分片，只在 FFN 的入口/出口附近进行少量 all-reduce。

