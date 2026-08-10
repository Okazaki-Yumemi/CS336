# 同一个 base model，只改变 prompt，模型的解题行为会发生多大变化？

# 三个实验对象

question_only 基本就是：

{question} Please put your final answer within \boxed{}.

没有明确告诉模型怎么 reasoning，只要求最后把答案放进 \boxed{}。
---

r1_zero 则明显在诱导 reasoning：

A conversation between User and Assistant...
...
User: {question}
Assistant: <think>

也就是 prompt 已经替模型打开了 <think>，希望模型接着生成：
reasoning ...
</think>
<answer>72</answer>

---

而 three-shot 版本则在当前问题前面放三个完整的：

question
→ reasoning
→ answer


# Reward:

staff 已经给了：

cs336_alignment.drgrpo_grader.r1_zero_reward_fn
cs336_alignment.drgrpo_grader.question_only_reward_fn

两个 R1 prompt：

r1_zero
r1_zero_three_shot

用：r1_zero_reward_fn
question_only 用：question_only_reward_fn



```py
from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn


def load_gsm8k():
    """
    Load the GSM8K dataset.
    Returns:
        A list of dictionaries, each containing a 'question' and 'answer'.
    """
    import json

    # Assuming the dataset is stored in a JSONL file named 'test.jsonl'
    dataset_path = "data/gsm8k/test.jsonl"
    
    data = []
    with open(dataset_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    return data

def load_prompt():
    """
    Load the prompt template from a text file.
    Returns:
        A string containing the prompt template.
    """
    question_only_path = "cs336_alignment/prompts/question_only.prompt"
    r1_zero_3_shot_path = "cs336_alignment/prompts/r1_zero_three_shot_gsm8k.prompt"
    r1_zero_path = "cs336_alignment/prompts/r1_zero.prompt"
    
    with open(question_only_path, 'r') as f:
        question_only_prompt = f.read()
    
    with open(r1_zero_3_shot_path, 'r') as f:
        r1_zero_3_shot_prompt = f.read()
    
    with open(r1_zero_path, 'r') as f:
        r1_zero_prompt = f.read()
    
    return question_only_prompt, r1_zero_3_shot_prompt, r1_zero_prompt

# 用vLLM测试

from cs336_alignment.vllm_utils import VLLMServer

data = load_gsm8k()
question_only_prompt, r1_zero_3_shot_prompt, r1_zero_prompt = load_prompt()

example = data[0]

prompt = r1_zero_3_shot_prompt.format(
    question=example['question']
)

server = VLLMServer(
    model_id="allenai/OLMo-2-0425-1B",
    gpu = 0,
    gpu_memory_utilization=0.7,
)

server.start()

sampling_params = {
    "temperature": 1.0,
    "max_tokens": 512,
    "n":1,
    "seed":0,
    "stop":["</answer>"],
    "include_stop_str_in_output": True
}

completions = server.generate_completions(
    prompts = [prompt],
    sampling_params = sampling_params
)

completion = completions[0]
ground_truth = example['answer'].split("####")[-1].strip()

reward = r1_zero_reward_fn(completion.text, ground_truth)

print("===============================")
print("TEXT:", repr(completion.text))
print("TOKEN IDS:", completion.token_ids)
print("FINISH REASON:", completion.finish_reason)
print("===============================")

print("REWARD:", reward)
print("===============================")

print("Question:")
print(example['question'])

print("Ground TRUTH")
print(ground_truth)

print("MODEL:")
print(completion.text)
print("===============================")
```


# A5 Prompting Baseline：初步实验记录

## 1. 实验目的

在正式运行整个 GSM8K benchmark 之前，先使用同一道题对三种 prompt 做单样本 smoke test，观察 `OLMo-2-0425-1B` 这一 base model 在不同 prompting 策略下的行为差异。

测试的三种 prompt：

1. `question_only`
2. `r1_zero`
3. `r1_zero_three_shot`

生成参数保持一致：

- temperature = 1.0
- max_tokens = 512
- n = 1
- seed = 0

对于 `r1_zero` 和 `r1_zero_three_shot`，使用 `</answer>` 作为 stop string。

测试题：

> Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?

正确答案：

```text
18
```
正确推理应为：
```
16 - 3 - 4 = 9 eggs
9 × $2 = $18
```

## 2. question_only

模型输出：
```
TEXT: ''
TOKEN IDS: [100257]
FINISH REASON: stop
```
即模型直接生成结束 token，没有尝试回答问题。

观察

这说明对于一个未经 instruction tuning 的 base model，仅仅提供：

question + "Please put your final answer within \boxed{}."

并不能保证模型进入“回答用户问题”的行为模式。

Base LM 本质上仍然是在执行 next-token prediction，而不是天然具有：

用户提出问题 → 必须回答问题

这种 assistant 行为。

在这个样本上，question_only 基本属于：

format_reward = 0
answer_reward = 0

即 Category 3。

## 3. Zero-shot r1_zero

加入 R1-Zero 风格 prompt，并以：

Assistant: <think>

作为生成起点后，模型开始尝试进行 chain-of-thought reasoning。

模型输出大致为：
```
Janet’s ducks lay 16 eggs per day...
</think> <answer>
Janet’s mode of payment is through a dollar store.
She has daily sales of 16 - 3 - 4 = 9 eggs each day.
She has 10 days worth of sales ...
</answer>
```
Reward：
```
format_reward = 1.0
answer_reward = 0.0
reward = 0.0
```
即 Category 2。

观察

相比 question_only，r1_zero 已经显著改变了模型行为：

question_only
→ 直接 EOS

r1_zero
→ 开始 reasoning
→ 正确使用 </think> <answer> ... </answer>

模型甚至正确计算出了中间结果：

16 - 3 - 4 = 9

但随后推理发生漂移，没有继续得到：

9 × 2 = 18

反而生成了与题意无关的“10 days worth of sales”等内容。

因此该样本是一个真实的模型错误，而不是 grader parsing error。

## 4. Three-shot r1_zero_three_shot

加入三个完整 reasoning demonstrations 后，模型输出：
```
Every day, she eats 6 ducks (3 + 4)
and has 10 ducks left (16 - 6).

Even though she only sells a portion at the market,
the total is 60 ducks.

60 ducks * $2 per duck = $120 every day.

</think> <answer> $120 </answer>
```
Reward：
```
format_reward = 1.0
answer_reward = 0.0
reward = 0.0
```
仍然属于 Category 2。

主要错误

该回答出现了多层 reasoning failure：

1. 3 + 4 = 6：基础算术错误，实际应为 7。
2. 将题目中的 eggs 错误地改成了 ducks。
3. 使用错误的 16 - 6 = 10 继续推理。
4. 无依据地产生了 60 ducks。
5. 最后正确计算了 60 × 2 = 120，但输入数字本身已经完全错误。

观察

Few-shot prompt 确实能够让模型稳定地模仿：

<think> reasoning </think>
<answer> answer </answer>

这一行为模式，但“学会输出 reasoning 的形式”并不意味着“获得了可靠的 reasoning 能力”。

换句话说：

> Format adherence ≠ reasoning correctness.








# batched test


```py
from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn


def load_gsm8k():
    """
    Load the GSM8K dataset.
    Returns:
        A list of dictionaries, each containing a 'question' and 'answer'.
    """
    import json

    # Assuming the dataset is stored in a JSONL file named 'test.jsonl'
    dataset_path = "data/gsm8k/test.jsonl"
    
    data = []
    with open(dataset_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    return data

def load_prompt():
    """
    Load the prompt template from a text file.
    Returns:
        A string containing the prompt template.
    """
    question_only_path = "cs336_alignment/prompts/question_only.prompt"
    r1_zero_3_shot_path = "cs336_alignment/prompts/r1_zero_three_shot_gsm8k.prompt"
    r1_zero_path = "cs336_alignment/prompts/r1_zero.prompt"
    
    with open(question_only_path, 'r') as f:
        question_only_prompt = f.read()
    
    with open(r1_zero_3_shot_path, 'r') as f:
        r1_zero_3_shot_prompt = f.read()
    
    with open(r1_zero_path, 'r') as f:
        r1_zero_prompt = f.read()
    
    return question_only_prompt, r1_zero_3_shot_prompt, r1_zero_prompt

# 用vLLM测试



def evaluate_prompt(
    server,
    examples,
    prompt_template,
    reward_fn,
    user_r1_stop
):
    prompts = []
    ground_truths = []
    
    # examples转换为prompts 和 ground_truths
    for example in examples:
        prompt = prompt_template.format(
            question=example['question']
        )
        prompts.append(prompt)
        ground_truths.append(example['answer'].split("####")[-1].strip())
    
    sampling_params = {
        "temperature": 1.0,
        "max_tokens": 512,
        "n":1,
        "seed":0,
    }
    if user_r1_stop:
        sampling_params["stop"] = ["</answer>"]
        sampling_params["include_stop_str_in_output"] = True
        
    # 一次多个response
    completions = server.generate_completions(
        prompts = prompts,
        sampling_params = sampling_params,
        batch_size = 4,
    )
    
    records = []
    
    category_1 = 0
    category_2 = 0
    category_3 = 0
    
    for example,ground_truth,completion in zip(
        examples,
        ground_truths,
        completions,
    ):
        reward = reward_fn(completion.text, ground_truth)
        
        format_reward = reward["format_reward"]
        answer_reward = reward["answer_reward"]
        
        if format_reward == 1 and answer_reward == 1:
            category = 1
            category_1 += 1
        elif format_reward == 1 and answer_reward == 0:
            category = 2
            category_2 += 1
        elif format_reward == 0 and answer_reward == 0:
            category = 3
            category_3 += 1
        else:
            category = -1
            
        records.append({
            "question": example['question'],
            "ground_truth": ground_truth,
            "completion": completion.text,
            "format_reward": format_reward,
            "answer_reward": answer_reward,
            "category": category,
        })
        
    stats = {
        "total": len(examples),
        "category_1": category_1,
        "category_2": category_2,
        "category_3": category_3,
        "accuracy": category_1 / len(examples) if len(examples) > 0 else 0.0,
    }
    
    return records, stats



from cs336_alignment.vllm_utils import VLLMServer
def main():
    data = load_gsm8k()
    
    question_only_prompt, r1_zero_3_shot_prompt, r1_zero_prompt = load_prompt()

    
    server = VLLMServer(
        model_id="allenai/OLMo-2-0425-1B",
        gpu= 0,
        gpu_memory_utilization= 0.7,
    )
    
    server.start()
    
    examples = data[:20]
    
    records, stats = evaluate_prompt(
        server=server,
        examples=examples,
        prompt_template=question_only_prompt,
        reward_fn=question_only_reward_fn,
        user_r1_stop=False,
    )
    print(records)
    print("===============================")
    print(stats)
    
if __name__ == "__main__":
    main()
```



结果

```bash
question only:{'total': 20, 'category_1': 0, 'category_2': 4, 'category_3': 16, 'accuracy': 0.0}

r1 zero {'total': 20, 'category_1': 0, 'category_2': 14, 'category_3': 6, 'accuracy': 0.0}

3 shot  {'total': 20, 'category_1': 0, 'category_2': 19, 'category_3': 1, 'accuracy': 0.0}
```

1B 模型太小了，无法在 20 个样本上得到任何正确答案。

prompt 越明确，尤其加入 demonstrations 后，模型越能学会“应该怎么回答”这个输出协议。

