# Training / 训练

```py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from cs336_basics.bpe import train_bpe
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.model import Linear
from cs336_basics.model import Embedding
from cs336_basics.model import RMSNorm
from cs336_basics.model import SwiGLU
from cs336_basics.model import RoPE
from cs336_basics.model import softmax
from cs336_basics.model import scaled_dot_product_attention
from cs336_basics.model import MultiHeadSelfAttention
from cs336_basics.model import TransformerBlock
from cs336_basics.model import TransformerLM
from cs336_basics.model import cross_entropy_loss
from cs336_basics.model import AdamW
from cs336_basics.model import cosine_lr_schedule
from cs336_basics.model import gradient_clipping
from cs336_basics.model import data_loader
from cs336_basics.model import save_checkpoint
from cs336_basics.model import load_checkpoint



def evaluate(
    model: TransformerLM,
    validation_data: np.ndarray,
    *,
    batch_size: int,
    context_length: int,
    device: str,
    num_batches: int
) -> float:
    """在若干随机验证 batch 上面计算平均 loss"""
    
    model.eval()
    
    losses: list[float] = []
    
    with torch.no_grad():
        for _ in range(num_batches):
            inputs, targets = data_loader(
                validation_data,
                batch_size = batch_size,
                context_length = context_length,
                device = device
            )
            
            logits = model(inputs)
            loss = cross_entropy_loss(logits, targets)
            
            losses.append(loss.item())
            
    model.train()
    return sum(losses) / len(losses)

def train(args: argparse.Namespace) -> None:
    """ 训练模型 """
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device(args.device)
    
    # 对 .npy 文件进行内存映射, 不一次性读入全部数据
    train_data = np.load(args.train_data, mmap_mode='r')
    validation_data = np.load(args.validation_data, mmap_mode='r')
    
    model = TransformerLM(
        d_model = args.d_model,
        num_heads = args.num_heads,
        d_ff = args.d_ff,
        max_seq_len = args.context_length,
        theta = args.rope_theta,
        vocab_size = args.vocab_size,
        num_layers = args.num_layers,
        device = device,
        dtype = torch.float32,
        context_length = args.context_length
    )
            
    optimizer = AdamW(
        model.parameters(),
        lr = args.max_lr,
        betas = (args.beta1, args.beta2),
        weight_decay = args.weight_decay,
        eps = args.eps
    )
    
    start_step = 0
    
    if args.resume_from is not None:
        start_step = load_checkpoint( 
            model,
            optimizer,
            args.resume_from,
        )
        
        print(f"Resume from step {start_step}")
    
    model.train()
    
    for step in range(start_step, args.max_steps):
        # 计算当前学习率
        current_lr = cosine_lr_schedule(
            it = step,
            max_learning_rate = args.max_lr,
            min_learning_rate = args.min_lr,
            warmup_steps = args.warmup_steps,
            cosine_cycle_iters = args.cosine_cycle_iters
        )
        
        # 优化器可能包含多个parameter group, 因此 全部更新
        for group in optimizer.param_groups:
            group['lr'] = current_lr
        
        # 随机采样输入和 next-token targets
        inputs, targets = data_loader(
            train_data,
            batch_size = args.batch_size,
            context_length = args.context_length,
            device = args.device,
        )
        
        # 清除上一轮梯度
        optimizer.zero_grad(set_to_none = True)
        
        # 前向传播
        logits = model(inputs)
        
        # logits: (B,T,V)
        # targets: (B,T)
        
        loss = cross_entropy_loss(logits, targets)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        gradient_clipping(
            model.parameters(),
            max_l2_norm = args.max_grad_norm,
        )
        
        # AdamW 更新
        optimizer.step()
        
        completed_step = step + 1
        
        if completed_step % args.log_interval == 0:
            print(
                f"step={completed_step:6d}",
                f"train_loss={loss.item():.4f}", 
                f"lr={current_lr:.6e}"
            )
        if completed_step % args.eval_interval == 0:
            validation_loss = evaluate(
                model,
                validation_data,
                batch_size = args.batch_size,
                context_length = args.context_length,
                device = args.device,
                num_batches = args.eval_batches,
            )
            
            print(
                f"step={completed_step:6d}",
                f"validation_loss={validation_loss:.4f}"
            )
        
        if completed_step % args.save_interval == 0:
            checkpoint_dir = Path(args.checkpoint_dir)
            checkpoint_dir.mkdir(parents = True, exist_ok = True)
            
            checkpoint_path = (
                checkpoint_dir / f"checkpoint_step_{completed_step}.pt"
            )
    
    #训练结束之后再保存一次最终状态
    
    final_path = Path(args.checkpoint_dir) / "checkpoint_final.pt"
    final_path.parent.mkdir(parents = True, exist_ok = True)
    
    save_checkpoint(
        model,
        optimizer,
        args.max_steps,
        final_path,
    )
    
    print(f"Training complete. Final checkpoint: {final_path}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    
    # 数据
    parser.add_argument("--train_data", type= str, required = True)
    parser.add_argument("--validation-data", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--resume-from", type=str, default=None)

    # 模型
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=768)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)

    # 优化器
    parser.add_argument("--max-lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    # 训练
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)

```

代码的主要部分其实就在
evaluate 函数里面, 主要是对模型进行验证, 计算平均 loss.
```py

model.eval() # 切换到验证模式, 关闭 dropout 等训练特有的操作

losses: list[float] = [] # 用于存储每个验证 batch 的 loss

with torch.no_grad(): # 不计算梯度, 节省显存和计算
    for _ in range(num_batches):
        inputs, targets = data_loader(
            validation_data,
            batch_size = batch_size,
            context_length = context_length,
            device = device
        )  # 随机采样一个 batch 的输入和目标
        
        logits = model(inputs) # 前向传播得到 logits
        loss = cross_entropy_loss(logits, targets) # 计算 loss
        
        losses.append(loss.item()) # 存储 loss

model.train() # 切换回训练模式
return sum(losses) / len(losses) # 返回平均 loss
```

Train 函数里面主要是训练循环, 每个 step 里面会做以下几件事:
```py

torch.manual_seed(args.seed) # 设置随机种子, 保证实验可复现
np.random.seed(args.seed) # 设置 numpy 的随机种子

device = torch.device(args.device) # 设置设备, CPU 或 GPU

# 对 .npy 文件进行内存映射, 不一次性读入全部数据
train_data = np.load(args.train_data, mmap_mode='r') # 训练数据
validation_data = np.load(args.validation_data, mmap_mode='r') # 验证数据

model = TransformerLM(
    d_model = args.d_model,
    num_heads = args.num_heads,
    d_ff = args.d_ff,
    max_seq_len = args.context_length,
    theta = args.rope_theta,
    vocab_size = args.vocab_size,
    num_layers = args.num_layers,
    device = device,
    dtype = torch.float32,
    context_length = args.context_length
) # 初始化模型

optimizer = AdamW(
    model.parameters(),
    lr = args.max_lr,
    betas = (args.beta1, args.beta2),
    weight_decay = args.weight_decay,
    eps = args.eps
) # 初始化优化器

start_step = 0 # 记录训练步数

if args.resume_from is not None:
    start_step = load_checkpoint( 
        model,
        optimizer,
        args.resume_from,
    ) # 如果指定了 checkpoint, 则加载模型和优化器状态
        
    print(f"Resume from step {start_step}")

model.train() # 切换到训练模式

for step in range(start_step, args.max_steps): # 训练循环
    # 计算当前学习率
    current_lr = cosine_lr_schedule(
        it = step,
        max_learning_rate = args.max_lr,
        min_learning_rate = args.min_lr,
        warmup_steps = args.warmup_steps,
        cosine_cycle_iters = args.cosine_cycle_iters
    ) # 学习率调度
    
    # 优化器可能包含多个parameter group, 因此 全部更新
    for group in optimizer.param_groups:
        group['lr'] = current_lr # 更新学习率
        
    # 随机采样输入和 next-token targets
    inputs, targets = data_loader(
        train_data,
        batch_size = args.batch_size,
        context_length = args.context_length,
        device = args.device,
    ) # 随机采样一个 batch 的输入和目标
    
    # 清除上一轮梯度
    optimizer.zero_grad(set_to_none = True) # 清除梯度
    
    # 前向传播
    logits = model(inputs) # 前向传播得到 logits
    
    # logits: (B,T,V)
    # targets: (B,T)
    
    loss = cross_entropy_loss(logits, targets) # 计算 loss
    
    # 反向传播
    loss.backward() # 反向传播计算梯度
    
    # 梯度裁剪
    gradient_clipping(
        model.parameters(),
        max_l2_norm = args.max_grad_norm,
    ) # 梯度裁剪
    
    # AdamW 更新
    optimizer.step() # 更新参数

    completed_step = step + 1 # 记录完成的步数

    if completed_step % args.log_interval == 0:
        print(
            f"step={completed_step:6d}",
            f"train_loss={loss.item():.4f}", 
            f"lr={current_lr:.6e}"
        ) # 打印训练信息
    if completed_step % args.eval_interval == 0:
        validation_loss = evaluate(
            model,
            validation_data,
            batch_size = args.batch_size,
            context_length = args.context_length,
            device = args.device,
            num_batches = args.eval_batches,
        ) # 计算验证 loss

        print(
            f"step={completed_step:6d}",
            f"validation_loss={validation_loss:.4f}"
        ) # 打印验证信息
    
    if completed_step % args.save_interval == 0:
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_dir.mkdir(parents = True, exist_ok = True) # 创建 checkpoint 目录

        checkpoint_path = (
            checkpoint_dir / f"checkpoint_step_{completed_step}.pt"
        ) # checkpoint 文件路径

final_path = Path(args.checkpoint_dir) / "checkpoint_final.pt"
final_path.parent.mkdir(parents = True, exist_ok = True) # 创建 checkpoint 目录

save_checkpoint(
    model,
    optimizer,
    args.max_steps,
    final_path,
) # 保存最终模型

```

然后parse_args 函数主要是解析命令行参数, 这里就不赘述了, 主要是设置训练的超参数和模型参数.


# Decode / 文本

- 接受Prompt
- 自回归地逐 token 生成
- 遇到 `<|endoftext|>`
- 支持最大生成长度
- 支持 temperature
- 支持 top-k / nucleus sampling

只取最后一个位置的logits

input_ids: (1, T)

模型输出

logits:(1,T,V)

- 1 batch size
- T 数列长度
- v 词表大小

生成下个token只需要

next_token_logits = logits[:, -1, :]  # (1, V)

# 最基础的采样

先做softmax

然后按照概率抽样

# Temperature

Temperature 修改 softmax 前的 logits

代码层面的操作就是
```
next_token_logits -> 除以 temperature -> softmax
```

**Temperature 小于 1**:
例如:
```
原始 logits: [3, 2, 1]
temperature = 0.5
缩放后 logits: [6, 4, 2]
```

最高概率 token 更容易被选中

**Temperature 大于 1**:

```
原始 logits: [3, 2, 1]
temperature = 2
缩放后 logits: [1.5, 1.0 , 0.5]
```

# Top-p 是什么

保留累计概率达到阈值p的最小token集合

例如 排序后

| Token |   概率 | 累计概率 |
| ----- | ---: | ---: |
| A     | 0.40 | 0.40 |
| B     | 0.25 | 0.65 |
| C     | 0.15 | 0.80 |
| D     | 0.10 | 0.90 |
| E     | 0.05 | 0.95 |
| 其他    | 0.05 | 1.00 |

p = 0.9
就会保留
A, B, C, D 四个token

# Top-p 的张量操作流程

对于概率 probs: (V,)

1. 排序
2. 计算cumulative sum
3. 找出累计概率超过p 后的尾部token
4. 将尾部概率设为 0
5. 重新归一化
6. multinomial 采样
7. 把排序后的索引映射回原词表 ID

# 完整decode流程

```
prompt 字符串
    ↓ tokenizer.encode
prompt_ids: list[int]
    ↓ 转成 Tensor
generated_ids: (1, prompt_length)

循环最多 max_new_tokens 次：
    ↓
截取当前 context window
    ↓
model(current_input)
    ↓
logits[:, -1, :]
    ↓
除以 temperature
    ↓
softmax
    ↓
可选 top-p 截断和重新归一化
    ↓
torch.multinomial
    ↓
next_token_id
    ↓
如果是 EOS：停止
    ↓
拼接到 generated_ids

generated_ids
    ↓ tokenizer.decode
最终字符串
```

```py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from cs336_basics.bpe import train_bpe
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.model import Linear
from cs336_basics.model import Embedding
from cs336_basics.model import RMSNorm
from cs336_basics.model import SwiGLU
from cs336_basics.model import RoPE
from cs336_basics.model import softmax
from cs336_basics.model import scaled_dot_product_attention
from cs336_basics.model import MultiHeadSelfAttention
from cs336_basics.model import TransformerBlock
from cs336_basics.model import TransformerLM
from cs336_basics.model import cross_entropy_loss
from cs336_basics.model import AdamW
from cs336_basics.model import cosine_lr_schedule
from cs336_basics.model import gradient_clipping
from cs336_basics.model import data_loader
from cs336_basics.model import save_checkpoint
from cs336_basics.model import load_checkpoint

def generate(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    context_length: int,
    temperature: float = 1.0,
    top_p : float = 1.0,
    eos_token_id: int | None = None,
    device: str = "cuda",
) -> str:
    """根据给定的 prompt 生成文本"""
    
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")

    if not 0 < top_p <= 1.0:
        raise ValueError("top_p must be in the range (0, 1]")
    
    
    
    model.eval()
    
    # prompt 编码
    input_ids = tokenizer.encode(prompt)
    # 空值拦截
    if len(input_ids) == 0:
        raise ValueError("prompt is empty after encoding")
    
    # 转为 tensor 并移动到指定设备
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)  # shape: (1, seq_len)
    # 生成的 token id 列表
    generated_ids = input_ids.tolist()[0]
    
    #循环最多生成 max_new_tokens 个 token
    for _ in range(max_new_tokens):
        # 如果当前输入长度超过 context_length，则截断
        if input_ids.size(1) > context_length:
            input_ids = input_ids[:, -context_length:]
            
        # 前向传播得到 logits
        with torch.no_grad():
            model_output = model(input_ids)
            logits = model_output[0, -1, :]  # 取最后一个 token 的 logits
            
        #应用温度缩放
        logits = logits / temperature
        #softmax 得到概率分布
        probs = softmax(logits,-1)
        
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(
                probs,
                descending = True,
            )
            
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            remove_mask = cumulative_probs > top_p
            
            # 将 mask 右移一位, 保留第一个跨过 top_p 的token
            remove_mask[1:] = remove_mask[:-1].clone()
            remove_mask[0] = False
            
            sorted_probs = sorted_probs.masked_fill(
                remove_mask,
                0.0,
            )
            
            sorted_probs = sorted_probs / sorted_probs.sum()
            
            #这里抽到的是排序后列表种的位置
            sample_rank = torch.multinomial(
                sorted_probs,
                num_samples=1,
            )
            
            # 映射为回原始词表 ID
            next_token_id = sorted_indices[sample_rank]
        
        else:
            
            # 不做top-p 截断, 直接在原词表分布上采样
            next_token_id = torch.multinomial(probs, num_samples=1)
        
        token_id = next_token_id.item()
        
        
        if eos_token_id is not None and token_id == eos_token_id:
            break
    
        # 将采样得到的 token id 添加到生成的 token id 列表中
        generated_ids.append(token_id)
        # 将采样得到的 token id 添加到输入中，以便下一次迭代
        input_ids = torch.cat([input_ids, next_token_id.view(1,1)], dim=1)
    # 从生成的 token id 列表中解码为字符串
    return tokenizer.decode(generated_ids)
```

