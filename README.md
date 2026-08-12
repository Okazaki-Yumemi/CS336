# CS336-2026

> Stanford **CS336: Language Modeling from Scratch, Spring 2026** 个人自学仓库。

本仓库记录我完整学习 Stanford CS336 的过程，包括课程资料、Assignment starter code、个人实现、实验记录、Debug 过程以及学习笔记。

- Course: [Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)
- Repository: [Okazaki-Yumemi/CS336](https://github.com/Okazaki-Yumemi/CS336)

> [!NOTE]
> 本仓库是个人自学记录，并非 Stanford 官方课程仓库。  
> 部分需要大规模 GPU / B200 资源的实验由于本地计算资源限制，没有完整复现；相关算法、代码路径、实验设计与分析仍进行了学习和实现。

---

## 学习状态

**CS336 Spring 2026 主线学习已完成。**

| Assignment | Topic | Status |
| --- | --- | --- |
| A1 | Basics | ✅ 完成 |
| A2 | Systems | ✅ 主线完成 |
| A3 | Scaling Laws | ✅ 主线完成，部分大规模实验未运行 |
| A4 | Data | ✅ 完成 |
| A5 | Alignment | ✅ 完成 |

整个学习过程大致覆盖了现代 Language Model 从训练前到训练后的完整技术链：

```text
Raw Text
   ↓
Tokenization
   ↓
Transformer
   ↓
Language Modeling
   ↓
Optimization & Training
   ↓
GPU / Distributed Systems
   ↓
Scaling Laws
   ↓
Data Processing
   ↓
Evaluation & Inference
   ↓
Post-training
   ↓
RL / Alignment
```

---

## Repository Structure

```text
CS336/
├── assignments/
│   ├── a1-basics/
│   ├── a2-systems/
│   ├── a3-scaling/
│   ├── a4-data/
│   └── a5-alignment/
├── lectures/
├── notes/
├── .gitignore
└── README.md
```

### `assignments/`

保存各次 Assignment 的：

- starter code
- 个人实现
- tests
- experiment scripts
- implementation records

### `lectures/`

保存课程相关资料，例如：

- lecture slides
- webpages
- images
- handouts
- 其他课程材料

### `notes/`

保存我的学习笔记，包括：

- Lecture 知识整理
- Assignment 实现思路
- 数学推导
- Debug 过程
- 实验分析
- 测试结果解释
- 对不同算法和系统设计的理解

笔记既按 Assignment 整理，也包含单独的 Lecture 学习记录。

---

# Assignment 1: Basics

**Status: ✅ Completed**

A1 从最底层开始构建一个 Language Model，主要学习和实现：

### Tokenization

- Unicode 与 UTF-8
- byte-level vocabulary
- GPT-2 style regex pre-tokenization
- Byte Pair Encoding (BPE)
- special token handling
- tokenizer encode / decode

### Transformer

- Linear
- Embedding
- RMSNorm
- SwiGLU
- Rotary Positional Embedding (RoPE)
- Self-Attention
- Transformer Block
- Causal Language Model

### Training

- Cross Entropy
- AdamW
- learning-rate scheduling
- gradient clipping
- checkpointing
- training loop
- text generation

### Profiling

- runtime benchmarking
- memory usage
- model profiling

A1 是整门课程的基础：从 raw text 开始，最终真正构建并训练一个 Transformer Language Model。

---

# Assignment 2: Systems

**Status: ✅ Main Track Completed**

A2 主要研究：

> 如何让 Language Model training 真正在 GPU 和分布式系统上高效运行。

主要内容包括：

- Profiling and Benchmarking
- GPU memory analysis
- mixed precision
- GPU kernels
- Triton
- Distributed Data Parallel (DDP)
- communication primitives
- gradient synchronization
- optimizer state sharding
- Fully Sharded Data Parallel (FSDP)
- parallelism strategy analysis

这一部分让我开始从单纯的“模型算法”转向理解：

```text
Model FLOPs
Memory
Communication
Synchronization
Parallelism
```

之间的系统级 trade-off。

---

# Assignment 3: Scaling

**Status: ✅ Main Track Completed**

A3 主要围绕 **Scaling Laws** 展开。

主要学习：

- model size / data / compute 之间的关系
- compute-optimal training
- power-law fitting
- experimental design
- scaling-law parameter estimation
- 根据小规模实验预测大规模训练结果
- compute budget allocation

部分大规模实验由于计算资源限制没有完整运行，因此这一 Assignment 的重点放在：

- 理论理解
- 实验方法
- 代码实现
- scaling law fitting
- 结果分析

核心问题是：

> 在有限 compute budget 下，模型应该多大、数据应该多少，以及如何利用小规模实验预测更大规模训练的行为？

---

# Assignment 4: Data

**Status: ✅ Completed**

A4 关注 Language Model training 中经常被低估、但极其重要的一部分：

> **Data**

主要涉及：

- web data processing
- text extraction
- filtering
- language identification
- quality filtering
- personally identifiable information processing
- deduplication
- exact / approximate matching
- MinHash
- data pipeline

这一部分让我进一步认识到：

```text
Better Model
    +
Better Systems
    +
Better Data
```

三者共同决定最终 Language Model 的质量。

模型 architecture 固定之后，数据质量、数据分布以及 preprocessing pipeline 本身仍然可以显著影响最终训练结果。

---

# Assignment 5: Alignment

**Status: ✅ Completed**

A5 进入 Language Model 的 **Post-training / Alignment**。

这一部分从 prompting 开始，一路进入现代 reasoning model 中常见的 Reinforcement Learning 方法。

主要内容包括：

### Prompting & Post-training

- prompting
- supervised fine-tuning background
- RLHF background
- Reinforcement Learning with Verifiable Rewards (RLVR)

### Policy Gradient

- language model as a policy
- trajectory
- reward / return
- REINFORCE
- policy gradient
- baseline
- advantage estimation
- variance reduction

### GRPO Family

- Group Relative Policy Optimization (GRPO)
- group-normalized rewards
- advantage normalization
- sequence normalization
- Dr. GRPO
- Rejection Sampling Fine-Tuning / RFT
- MaxRL

### Off-policy RL

- behavior policy / current policy
- stale rollout
- importance sampling
- sequence-level importance reweighting
- token-level importance reweighting
- bias-variance trade-off
- PPO-style clipping
- GRPO clipping
- GSPO
- sequence-level geometric-mean importance ratio

这一部分最终让我把许多看似不同的 RL estimator 放到了同一个框架下：

```text
REINFORCE
    ↓
Baseline / Advantage
    ↓
GRPO
    ↓
Normalization Choices
    ├── Dr. GRPO
    ├── RFT
    └── MaxRL
    ↓
Off-policy Reuse
    ↓
Importance Reweighting
    ↓
Clipping
    ├── GRPO
    └── GSPO
```

这些方法本质上都在解决同一个问题：

> 如何利用有限、昂贵并且高方差的 rollout，构造一个足够准确、稳定且高效的 policy-gradient estimator？

---

# Environment

每个 Assignment 基本都是独立的 Python project，拥有自己的：

- `pyproject.toml`
- dependencies
- environment configuration

项目主要使用 [`uv`](https://docs.astral.sh/uv/) 进行 Python 环境和依赖管理。

例如：

```bash
cd assignments/a1-basics

uv sync

uv run pytest
```

运行单独测试文件：

```bash
uv run pytest tests/test_train_bpe.py -vv
```

运行指定测试：

```bash
uv run pytest -k test_train_bpe
```

不同 Assignment 的依赖可能不同，因此建议分别在对应目录中使用 `uv` 管理环境。

---

# Notes

详细学习记录位于：

```text
notes/
```

我的笔记并不只记录最后的实现结果，而更关注：

- 一个问题应该如何拆解
- 数学公式如何对应到实际代码
- 为什么某个算法这样设计
- 应该选择什么数据结构
- Tensor shape 如何流动
- 某个 bug 为什么出现
- 如何根据 test failure 定位问题
- 不同 estimator 的 bias / variance
- 不同 system design 的 compute / memory / communication trade-off
- 实验究竟在验证什么 hypothesis

因此这个仓库的目的并不只是保存“最终能运行的代码”，也希望保存：

> **从不知道，到理解，再到真正实现出来的过程。**

---

# What I Learned

CS336 最重要的收获之一，是让我不再把 Large Language Model 看成单独的 Transformer architecture。

一个真正的 Language Model system 是多个层次共同构成的：

```text
                   Language Model
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
     Algorithm         Systems           Data
        │                │                │
 Transformer        GPU Kernels       Filtering
 Optimization       Parallelism       Deduplication
 Tokenization       Communication     Data Mixture
        │                │                │
        └────────────────┼────────────────┘
                         ▼
                     Training
                         │
                         ▼
                    Evaluation
                         │
                         ▼
                    Post-training
                         │
                         ▼
                 Alignment / RL
```

课程前半部分回答：

> **一个 Language Model 是如何从零开始训练出来的？**

课程中段回答：

> **如何让它训练得更快、更大、更高效？**

课程后半部分回答：

> **应该使用什么数据，以及如何在预训练之后进一步改善模型行为？**

最终，这门课把 Tokenization、Transformer、Optimization、GPU Systems、Scaling Laws、Data 和 Reinforcement Learning 串成了一套相对完整的 Language Modeling 技术体系。

---

# About This Repository

这是一个个人自学仓库，而不是课程答案仓库。

其中代码主要用于：

- 学习
- 实现
- Debug
- 实验
- 复习
- 记录学习过程

部分目录包含 Stanford 提供的 Assignment starter code 和课程材料。

课程原始材料、Assignment specification 以及相关教学内容的版权归其原作者和 Stanford University 所有。

本仓库并非 Stanford 官方课程仓库。

个人实现代码仅代表我的学习过程，不应被其他学生直接复制或作为自己的课程作业提交。

---

## Final Status

> **Stanford CS336 Spring 2026 — Main Track Completed**

后续可能继续补充：

- 全课程复习与总结
- 笔记整理
- 代码重构
- 未运行的大规模实验
- 相关论文阅读
- Language Modeling / LLM Systems / Post-training 相关学习