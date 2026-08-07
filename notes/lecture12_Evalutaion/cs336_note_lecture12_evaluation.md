# Evaluation

**Core challenge** -> concrete metric

Maybe a model is good if it does well on benchmarks...

![alt text](image.png)

Maybe a model is good if people simply like to pay for it...


## perplexity

测量困惑度.

GPT-2 开展了标准集训练，测试集评估的新范式

- True distribution is t, model is p
- BBest possible perplexity is H(t) obtained iff p = t
- if p = t , then solve all the tasks
- So by pushing down on perplexity,we will eventually  reach AGI

## exam_benchmarks

**Massive Multitask Language Understanding (MMLU)**
- 57 subjects,multiple choice
- Collected by graduate students from Wikipedia, textbooks, and online courses...
![alt text](image-1.png)

**Graduate level google proof Q＆A (GPQA)**

过去很难，博士出的题，非专家基本上难以答对。

![alt text](image-2.png)

**Humanity's last exam (HLE)**

![alt text](image-3.png)

![alt text](image-4.png)

这个考试目前mythos也只有64%

## chat_benchmarks

**Arena ai**
- Randin person 
- 给两个ai回复，让人选一个更好的

- Define model: p(A wins against B) = 1 / (1 + 10^((ELO(B) - ELO(A)) / 400))

**AlpacaEval**

用大语言模型当裁判

**WildBench**:

![alt text](image-5.png)

## Agentic benchmarks

- Previously: evaluate what LMs say
- Now: evaluate what LMs do

Agent = language model + agent scaffold


**SWEBench**

给一个任务和issue description，然后交一个PR

![alt text](image-6.png)

**Terminal Bench**
![alt text](image-7.png)

**CyBench**
![alt text](image-8.png)

- 40 Capture the Flag tasks
- Use first-solve time a measure of performance

**MLEBench**

- 75 kaggle competitions

Agent scaffolds 很重要
![alt text](image-9.png)

- Explicit planning
- Hierarchical delegation
- Persistent memory
- Extreme context engineering


## Pure_reasoning_benchmarks

ARC-AGI 1 \ 2 \ 3 代通过规律考察推理能力

例如ARC-AGI 3
![alt text](image-10.png)

交互式的解谜游戏

- Goal is to disentangle reasoning from knowledge
- Constrained to human reasoning
- Clearly exposes gaps in current agents.

## Safety benchmarks

**HarmBench**

**AIR-BENCH**

Jailbreaking
- Language models are trained to refuse harmful instructions
- Greedy coordinate gradient, automatically optimizes prompts to bypass safety
- Transfers from open-weight models to closed models.


## validity

**Train-test overlap**

- Don't train on your test set.
- Pre-foundation models (ImageNet,SQuAD) well defined train-test split


Route 2: encourage reporting norms

Route 3: use fresh evals
- LiveCodeBench, UncheatableEval: scrape new webpages
- Timestamps aren't sate due to copying either

Route 4: use private evals
- Companies use internal code bases that are not on the Internet

**Dataset Quality**

![alt text](image-11.png)

