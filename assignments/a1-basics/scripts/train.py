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