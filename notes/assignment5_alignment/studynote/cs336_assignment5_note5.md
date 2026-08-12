# 6. Off-policy RL

Our experiments and algorithms so far have been fully on-policy, meaning that every gradient estimator used samples drawn directly from the model being updated.

In this section, we'll explore off-policy RL, which takes multiple training steps per inference batch and hast the potential to speed up training, at the cost of potential instability and increased and increased algorithmic complexity.

## 6.1 Importance reweighting

In the vanilla version of policy gradients, we sample a batch of responses, and then take a single large-batch gradient step on these responses.

But we might hope for the model to train faster if we split the batch into a series of minibatches,and then take one training step per minibatch.

This approach is called "off-policy RL", in contrast with our original approach of "on-policy RL" where we take one train step per inference batch.

The reason this approach is called "off-policy" is that after the first minibatch, the current policy now differs from the policy we used during inference. The samples are now "off-policy" or "stale": they are not from the current policy, for which we would like to compute a policy gradient.

In notation,if we denote the inference policy as $\pi_0$, and the current policy as $\pi_\theta$,we have the following:

$$E_{x \sim \rho}E_{y \sim \pi_0}[r(y|x) \nabla_\theta \log \pi_\theta(y|x)] != E_{x \sim \rho}E_{y \sim \pi_\theta}[r(y|x) \nabla_\theta \log \pi_\theta(y|x)]$$

左边是 naive off-policy estimator， 其中的y是从旧的策略采样的。

off-policy的本质就是我们拿到的输入数据是很昂贵的，我们不希望每次都得重新拿到输入的数据，然后仅仅训练一步。我们真正想要做的是在拿到一批数据之后，尽可能多地训练模型。off-policy就是这样，每个minibatch都会step一次，但是问题也来了，原来的policy-gradient 就失效了，因为我们拿到的y是从旧的策略采样的。也就是上面的这个公式。


核心技巧:

概率论里面的问题:
> 想要求p下面的expectation,但是只有Q里的样本，怎么办

$$ E_{y ~P}[f(y)] = E_{y ~Q}[\frac{P(y)}{Q(y)} f(y)] $$

这里设 P = $\pi_\theta$， Q = $\pi_0$， 
 
$$ w(y) = \frac{\pi_\theta(y|x)}{\pi_0(y|x)}$$
就是importance weight
所以，正确的 off-policy gradient estimator 是

$$ E_{y \sim \pi_0} [\frac{\pi_\theta(y|x)}{\pi_0(y|x)} \nabla_\theta \log \pi_\theta(y|x) ] $$

经过weighting之后，在expectation上就等价于从 $\pi_\theta$ 采样的gradient estimator了。

在语言模型内，y是一整个sequence， 我们需要

$$ w(y) = \frac{\pi_\theta(y|x)}{\pi_0(y|x)}$$

但是语言模型的sequence probability 是一个累乘。

所以

$$ \frac{\pi_\theta(y|x)}{\pi_0(y|x)} = \prod_{t=1}^L \frac{\pi_\theta(y_t|x, y_{<t})}{\pi_0(y_t|x, y_{<t})} $$

sequence importance weight = 所有 token importance ratios 的乘积.

假设每个token只有一点点区别，都会被累乘放大，导致最终的importance weight非常大或者非常小，导致gradient estimator的方差非常大，就容易带来variance explosion的问题。

所以我们最后得到的是两种方法

**方法A: 完全不reweight**:直接就从$\pi_0$ 采样，没有任何variance爆炸的可能，但是是我们拿到的gradient estimator是biased的，不能保证收敛到最优解。

**方法B: 完整的sequence importance reweighting**: 优点是correct expectation, 缺点是 variance 巨大

old_log_probs 到底是什么, 在取样之后，我们要保留之前的log_probs，然后对同一条旧的response再forward当前的模型，就得到新的log_probs，然后就可以计算importance weight了。


## 6.2 PPO/ GRPO-style importance reweighting and clipping

### 6.2.1 Token-level reweighting

PPO/GRPO 并没用6.1那个理论完全正确的 sequence-level importance sampling, 而是做了一个有bias，但是variance小得多的token-level estimator

PPO/GRPO的token-level 公式是

$$ r(y) \sum_{t=1}^L w_t \nabla_\theta \log \pi_\theta(y_t|y_{<t})$$

```
sequence-level:

token 1 grad ─┐
token 2 grad ─┼─ × (w1 w2 w3 ... wL)
token 3 grad ─┤
...           │
token L grad ─┘


token-level:

token 1 grad × w1
token 2 grad × w2
token 3 grad × w3
...
token L grad × wL
```

上一6.1中最大的问题是

$$ \prod_{t=1}^L w_t$$

哪怕每个w = 1.01, 都会放大特别多。

而token-level estimator 每一项只出现 wt, 不再有长度 L 个 ratio 的乘积, 所以handout指出这些 importance reweighting terms 不再随着 response length 指数增长，variance 会显著降低.

这个算法就相当于只会在你要的那一步把 $\pi_\theta$ 和 $\pi_0$ 的差异修正

一个具体的例子:

假设当前模型已经学会了一种新 reasoning pattern：
```
"... therefore we should factor the quadratic ..."
```

旧模型并不熟悉这种 reasoning 风格。
当前 policy 在 timestep t 产生：
```
"factor"
```
这个 token 本来可能非常好，因为当前 policy 后面能够继续
```
factor
→ find roots
→ substitute
→ correct answer
```

但 token-level surrogate 评估它的时候：
```
factor
→ OLD policy 接管
→ old policy 不知道怎么延续
→ reasoning drift
→ wrong answer
```

handout problem中那个题目是让我们考虑 reweighting的范围,例如不采用那么夸张的 sequence-level importance weight, 而是只在某个范围内做 reweighting, 例如只在前几个 token 做 reweighting, 后面的 token 不再做 reweighting, 这样就可以避免后面 reasoning drift 的问题。

```
0-token correction:
no reweighting

1-token correction:
PPO/GRPO token-level

2-token correction:
handout pairwise estimator

...

L-token correction:
full sequence importance sampling
```



### 6.2.2 Clipping

为什么要clip?

现在token-level importance ratio是

$$ w_t = \frac{\pi_\theta(y_t | x, y_{<t})}{\pi_0(y_t | x, y_{<t})}$$

- wt = 1 current 和 old 很接近
- wt >> 1 current policy 很喜欢这个 token, old policy 不喜欢
- wt << 1 current policy 不喜欢这个 token, old policy 很喜欢

如果不断复用一套旧的rollout， 那么 $\pi_\theta$ 会逐渐远离 $\pi_0$,wt可能越来越极端

所以如果一个token的 ratio已经偏离1太多，就不要继续无限推动它。

Handout 因此把ratio限制在了 [1-ε, 1+ε] 的范围内，超过这个范围的ratio就被clip掉了。

PPO用的也不是简单的clipped ration，其用的是

$$ min(Aw_t, A clip(w_t, 1-\epsilon, 1+\epsilon))$$

PPO 想限制的不是：

>“任何 w 离 1 太远都不准动。”

而是：

>如果 policy 已经朝着我们想要的方向移动得太远，就停止继续获得收益；如果它反而朝错误方向跑了，还必须保留梯度把它拉回来。

举个例子:


假设现在 A = +1， epsilon = 0.2

假如 w = 1.1， clip之后得到的还是1.1，如果w=1.5,就停在1.2, 也就是1+epsilon

假如 A = 1， w = 0.5 证明这个action明明是好的，但是当前policy却不喜欢它，这种情况必阻止，所以

min(Aw, A clip(w, 1-ε, 1+ε)) = min(0.5, 0.8) = 0.5, 并不会因为截断就不去压制它了。

A < 0 类似，假如 A = -1, w = 1.5, clip之后得到的是 -1.5 要狠狠压制

所以PPO clipping 其实是非对称的

```
A > 0（好 action）：

w 太小                    w 正常                  w 太大
← wrong direction                                  desired direction →
  不 clip                  不 clip                  CLIP
                                                     ↑
                                              好得过头才限制


A < 0（坏 action）：

w 太小                    w 正常                  w 太大
← desired direction                                  wrong direction →
  CLIP                     不 clip                  不 clip
   ↑
坏 action 已经压够了
```


**Problem: Off-policy gradient with token-level reweighting**:

Update your method compute_policy_gradient to support grpo or noclip 
old_log_probs denotes the per-token log probabilities under the model used to generate the rollouts, and you’ll have to add a few lines to your training script to compute these. cliprange denotes the clipping strength parameter 𝜀.


## 6.3 GSPO

GRPO level是每个token还有自己的importance ratio，

而GSPO认为这种token-level太局部，于是它重新回到sequence-level

完整的sequence ratio本来是

$$ w(y) = \prod_{t=1}^L \rho_t $$

但是我们知道这玩意的variance太恐怖，所以GSPO直接做了一键式去

$$ s = (\prod_{t=1}^L \rho_t)^{1/L} $$

变成几何平均,这个地方 L 是 response token数量，不能包括prompt token数量。

```py
def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    
    
    # === shaping ===
    if raw_rewards_or_advantages.ndim == 1:
        # (B,) -> (B,1)
        advantages = raw_rewards_or_advantages.unsqueeze(-1)
    else:
        advantages = raw_rewards_or_advantages
    
    
    
    if importance_reweighting_method == "none":
        per_token_loss = - advantages * policy_log_probs
    
    elif importance_reweighting_method == "noclip":
        
        if old_log_probs == None:
            raise ValueError
        
        ratio = torch.exp(policy_log_probs - old_log_probs)

        objective = advantages * ratio

        per_token_loss = - objective
    
    elif importance_reweighting_method == "grpo":
    
        if old_log_probs == None:
            raise ValueError
                
        ratio = torch.exp(policy_log_probs - old_log_probs)

        unclipped_objective = advantages * ratio
        
        if cliprange == None:
            raise ValueError
        
        clipped_ratio = torch.clamp(ratio, 1-cliprange, 1+cliprange)

        clipped_objective = advantages * clipped_ratio
        
        objective = torch.minimum(
            unclipped_objective,
            clipped_objective,
        )
    
        per_token_loss = - objective
        
    elif importance_reweighting_method == "gspo":
        
        if old_log_probs == None:
            raise ValueError
        if cliprange == None:
            raise ValueError
        if response_mask == None:
            raise ValueError
        
        log_ratio = policy_log_probs - old_log_probs
        
        masked_log_ratio = log_ratio * response_mask
        
        response_length = response_mask.sum(dim = 1, keepdim= True)
        
        mean_log_ratio = masked_log_ratio.sum(dim=1, keepdim=True) / response_length
        
        sequence_ratio = torch.exp(mean_log_ratio)
        
        unclipped_objective = advantages * sequence_ratio
        
        clipped_ratio = torch.clamp(sequence_ratio,1-cliprange,1+cliprange)
        
        clipped_objective = advantages * clipped_ratio
        
        objective = torch.minimum(
            unclipped_objective,
            clipped_objective,
        )
        
        per_token_loss = - objective.expand_as(policy_log_probs)
```