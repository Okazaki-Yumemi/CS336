# Overview
Some design decisions are simply not (yet) justifiable and just come from experimentation.
Example: Noam Shazeer paper that introduced SwiGLU

## The bitter lesson
**accuracy = efficiency x resources**

## Introduction to language models

### Pre-neural (before 2010s)
- Language model to measeure thre entropy of English

### Neural ingredients (2010s)
- Long-Short Term Memory
- First neural language model
- Sequence-to-sequence modeling
- Adam optimizer
- Attention mechanism
- Transformer architecture
- mixture of experts (2017)
- model parallelism (2018)

### Early foundations (2018-2020)
- ELMo
- BERT
- Googles T5

### Embracing scaling
- OpenAI GPT-2
- Scaling laws
- OpenAI GPT-3
- Google PaLM
- Deepmind Chinchilla

### Open models
- EleutherAI open datasets
- Meta's OPT
- HuggingFace's BLOOM

- Meta's LLaMA
- Mistral's models
- Deepseek
- Qwen
- Minimax
- GLM
- Xiaomi MIMO

**Ideas from open models enable us to teach CS336**

what is a language model?
- 2018 BERT: something you fine-tune
- 2020 GPT-3: something you prompt
- 2022 ChatGPT: something you talk to
- 2026 agents: something that acts autonomously

# course logistics

- 5 assignments

- Implement locally to test for correctness , then run on cluster for benchmarking
- No scaffolding code , but we provide unit tests and adapter interfaces to help you check correctness and benchmark your code

## 不同的部分
### basics:
Goal: be able to train a basic language model
Components: tokenization, model architecture , training

### Tokenization:

popular tokenization methods: BPE

Efficiency lens
- Reduce context length
- Adaptive computation

The dream: tokenizer-free models

Refinements:
- Activation functions ReLU, SwiGLU
- Positional encodings:RoPE, sinusoidal
- Normalization: LayerNorm,RMSNorm
- Attention: Full,sparse
- Recurrence/ state-space / linear attention
- MLP:dense, mixture of experts
- Shape:

### Training:
- Loss functions: cross-entropy, contrastive, RLHF
- Optimizers: Adam, Adafactor, Lion
- Initialization: Xavier, Kaiming, NTK
- Learning rate schedules: cosine, linear warmup, one-cycle
- Regularization: dropout, weight decay, label smoothing
- Batch size: Critical batch size, gradient accumulation, micro-batching
- MoE specific: load balancing, expert dropout, capacity factor

# Assignment 1
- Implement BPE tokenizer
- Implement Transformer , cross-entropy loss, AdamW optimizer , training loop
- Do resource accounting
- Train on TinyStories and OpenWebText
- Leaderboard: minimize OpenWebText perplexity

**High-level principles**:
- Expressivity
- Stability
- Efficiency

# Assignment 2
systems

- Resource accounting: memory and compute characteristics of a model

Kernel
- Kernel is a function that runs on GPU
- When using pyTorch , each primitive operation is a kernel
- Can write custom kernels to make GPUS go brrr
- Principle: organize computation to minimize data movement
- Naive:read HBM,compute A,write HBM,read HBM,compute B,write HBM
- Fused: read HBM,compute A and B,write HBM
- Strategies: operator fusion , tiling
- Warp divergence,memory coalescing , bank conflicts
- Write kernels in CUDA/Triton

Parallelism

Inference:
Goal: generate tokens given a prompt
Inference is also needed for reinforcement learning,test-time compute , evaluation

- Prefill(similar to training): compute all attention states for the prompt
- Decode: generate tokens one at a time, compute attention states for each new token

Methods to speed up decoding
- Use cheaper model
- Speculative decoding
- Systems optimizations: kernel fusion, quantization, memory management

### Works
- Implement a fused RMSNorm kernel in Triton
- Implement distributed data parallel training
- Implement optimizer state sharding
- Benchmark and profile the implementations

# Assignment 3
scaling-laws

- optimize the scaling targeting a larger scale using smaller scale experiments
- Predict the loss at the scale before actually running  the experiment

TLDR: 计算量大约是模型大小的20倍

Assignment 3
- We define a training API
- Submit "training jobs"
- Fit scaling laws to the data points
- Submit extrapolated hyperparameters and loss predictions

# Assignment 4
data.

## Evaluation
What is the purpose of evaluation?
- Internal: guide model development
- External: Measure absolute quality of a real use case

Examples of evaluations:
1. Perplexity: ideally run on private documents ont on Internet
2. Advanced use cases:GPQA,HLE,SWE-Bench,Terminal-Bench
LMs are general purpose,require a diverse set of evaluations!

Data curation
- Data dose not just fall from the sky
- Sources:Webpages crawled form the Internet, books, code, Wikipedia, academic papers, social media, etc.
- Appeal to fair use to train on copyrighted data
- Might have to license data
- Raw data is HTML,PDF,Directories(not text)

Data Processing
- Transformation:convert to text, remove boilerplate, deduplicate, filter
- Filtering: keep high-quality data, remove low-quality data, remove duplicates, remove toxic content, remove personally identifiable information
- Deduplication: save compute,avoid memorization
- Data mixing:how much to upweight/downweight
- Rewriting / synthetic data

Types of data:
- Pretraining data: large and diverse
- Mid-training data:high-quality, 
- Post-training data:supervised fine-tuning

assignment 4
- Convert Common Crawl HTML to text
- Train classifiers to filter for quality, toxicity, and PII
- Deduplications using minhash and simhash
- Leaderboard: minimize perplexity given token budget

# Assignment 5
alignment

weak supervision

Basic template:
1. Generate responses from the model
2. Score responses with a human
3. Update the model to produce better responses

Algorithms:
- Proximal Policy Optimization (PPO)
- Direct Preference Optimization (DPO)
- Group Relative Preference Optimization (GRPO)

Challenges:
- RL algorithms are unstable and hard to tune
- At scale,this requires a lot of new infrastructure and tooling
- Constantly trading off systems efficiency and op-policyness

Assignment 5

- Implement Dpo and GRPO

Remember: it's all about efficiency
- Resources: data + hardware
- How do you train the best model with the least resources?

- Systems: clearly about efficiency
- Tokenization: working with raw bytes is elegant but compute-inefficient 
- Model architecture
- Data filtering
- Scaling laws