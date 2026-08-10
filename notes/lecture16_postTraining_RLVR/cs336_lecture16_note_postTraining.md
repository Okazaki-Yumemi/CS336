# RLVR = Reinforcement Learning with Verifiable Rewards

```
Question
   ↓
LM 自己生成一整段 reasoning
   ↓
最终答案 = 42
   ↓
程序检查答案
   ↓
correct → reward = 1
wrong   → reward = 0
```

coding 也类似：

```
模型写程序
   ↓
跑 unit tests
   ↓
pass → 1
fail → 0
```

所以和经典 RLHF 最大的变化甚至不是 RL algorithm，而是 reward source.

```
RLHF:
human preference → reward model → reward

RLVR:
ground truth / compiler / unit test / verifier → reward
```
后者的 reward 通常便宜、客观、可以大规模自动生成。DeepSeek-R1 这类工作正是证明了在数学、代码等可验证任务上，只利用这种结果 reward 进行 RL，也能显著增强 reasoning。


# 先把语言模型彻底看成一个 RL policy

以前 A1 我们写

$$ P_\theta(y_t|x,y_{<t}) $$

到了RL里面只是换个名字

$$ \pi_\theta(a_t|s_t) $$

```
RL                        Language Model
------------------------------------------------
state s_t          =      prompt + 已经生成的 tokens
action a_t         =      下一个 token
policy πθ          =      LM
trajectory τ       =      整个 response
reward R           =      最终答案是否正确
```

RLVR的目标就是让模型自己生成答案，然后提高那些高 reward 答案出现的概率

# 那一个只有 0/1 的最终 reward，怎么训练几百个 token？
假设题目是：

17 × 23 = ?

模型 rollout 四次：

y1: reasoning A ... 391     reward = 1
y2: reasoning B ... 381     reward = 0
y3: reasoning C ... 391     reward = 1
y4: reasoning D ... 401     reward = 0

我们希望：

y1 ↑ probability
y3 ↑ probability

y2 ↓ probability
y4 ↓ probability

Policy Gradient的关机就是如果这一整条 trajectory 最后成功了，就提高组成这条 trajectory的token的概率

但这里也暴露了一个很大的缺点：

RLVR 并不知道“哪一步 reasoning 是关键的”。

# 为什么不是直接乘 reward，而要搞 Advantage？

variance太大，于是RL 里经典做法是减一个 baseline

PPO 通常会额外训练一个 $V_\phi(s_t)$ 作为critic来估计baseline

但是对 LLM 来说，又养一个和 LM 差不多大的 value model，很贵。

于是 GRPO 出场。

# GRPO

对同一道题sample G个response,得到R1, R2, ..., RG

然后不用value network，而是直接计算这一组的平均 reward


然后约等于
$$ A_i = \frac {R_i - R}{std(R)+\epsilon} $$

现在语义就是

```
比同伴做得好 → positive advantage → 概率 ↑
比同伴做得差 → negative advantage → 概率 ↓
```

GRPO 最初由 DeepSeekMath 提出，其关键动机之一就是去掉 PPO 所需的额外 value model，从而降低 RL 训练的资源开销。

这样就有一个现象了，RLVR如果全错 or 全对，都学不到什么东西，所以RLVR最喜欢的是处理处在模型能力边界附近的题目。

```
取一个 math prompt x
          ↓
当前 policy sample G 个答案
          ↓
verifier 给每个答案 R_i
          ↓
同组 reward 做 normalization
          ↓
得到 advantage A_i
          ↓
重新算这些 response 的 token log-prob
          ↓
positive advantage:
    increase response probability

negative advantage:
    decrease response probability
          ↓
importance ratio + clipping
防止一步走太远
          ↓
optimizer.step()
          ↓
再 rollout
```

