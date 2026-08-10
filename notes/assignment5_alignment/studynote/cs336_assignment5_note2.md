# 3.Prompting

Our first step in applying our pretrained base model to downstream is to prompt it. Base models are pretrained on a wide range of bahaviors,and prompting is a lightweight way to shape its behavoir toward task-solving modes. Later in the assignment,we'll also see that the prompt choice shapes RL dynamics and exploration.

The most basic way to prompt the model is to provide the question, and sample from its next token distribution to produce the answer; we will call this strategy to the r1_zero prompt,which includes both the question and instructions for the model to do chain-of-thought reasoning.


## 3.1 Using vLLM for inference

To generate responses from the model, we will need an inference engine. Implementing an inference engine is outside the scope of this assignment,so we will use the vLLM inference engine,which implements a variety of optimizations including fast CUDA kernels, PagedAttention for efficient attention KV caching, and so on. Code to start a vLLM server and produce generations has been provided in cs336_alignment/vllm_utils.py, which has the following interface:

```py

@dataclass
class VLLMCompletion:
    text: str
    token_ids: list[int]
    finish_reason: str | None


@dataclass
class VLLMServer:
    model_id: str
    gpu: int = 0
    seed: int = 0
    gpu_memory_utilization: float = 0.9

def start(self) -> None: ...

def generate_completion(
    self,
    prompts: list[str],
    sampling_params: dict,
    batch_size: int | None = None,
) -> list[VLLMCompletion]: ...
```

## 3.2 Zero-shot, few-shot, and chain-of-thought prompting

Deepseek R1-Zero prompt:

```
A conversation between User and Assistant. The User asks a question, and the Assistant solves 
it. The Assistant first thinks about the reasoning process in the mind and then provides the 
User with the answer. The reasoning process is enclosed within <think> </think> and the 
answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning 
process here </think> <answer> answer here </answer>.
User: {question}
Assistant: <think>
```

In this prompt, question refers to some question that we insert. The expectation is that the model plays the role of the assistant, and starts generating the thinking process, and then generates a final symbolic answer within the answer tags,like <answer> 4x + 10 </answer>. The purpose of having the model generate tags like <answer> </answer> is so that we can easily parse the model's output and compare it against a ground truth answer, and so that we can stop response generation when we see the closing answer tag </answer>.

Another prompting approach, called few-shot prompting, is to prepend a few question-answer pairs before prompting the model with the actual question. A few-shot version of the r1_zero prompt looks like the following:

```
A conversation between User and Assistant. The User asks a question, and the Assistant solves 
it. The Assistant first thinks about the reasoning process in the mind and then provides the 
User with the answer. The reasoning process is enclosed within <think> </think> and the 
answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning 
process here </think> <answer> answer here </answer>.
User: {question-1}
Assistant: <think> {reasoning-1} </think> <answer> {answer-1} </answer>
User: {question-2}
Assistant: <think> {reasoning-2} </think> <answer> {answer-2} </answer>
User: {question-3}
Assistant: <think> {reasoning-3} </think> <answer> {answer-3} </answer>
User: {question}
Assistant: <think>
```
Few-shot prompting improves the model’s performance by giving it a few examples of the task it is meant to solve.

Finally,as a baseline,we will include the question_only prompt, available at `cs336_alignment/prompts/question_only.prompt`

```
{question} Please put your final answer within \\boxed{{}}
```

## 3.3 Grading function

Once the model generates a response, we need to check whether it is correct. Our math problems include ground truth answers, but the model can provide a correct answer in many ways: for example,in can answer <answer> 1/2 </answer> or `The answer is 0.5`. So to properly grade model outputs,we need an answer parsing function that takes as input the model's output and a known ground truth, and returns a boolean indicating whether the model's output is correct.

For our experiments, we will use a fast and fairly accurate answer parser used in recent work on reasoning RL . For the r1_zero prompts, this reward function is implemented at cs336_alignment.drgrpo_grader.r1_zero_reward_fn. The question_only prompt does not ask the model to use <think> and <answer> tags, so it should instead be evaluated with cs336_alignment.drgrpo_grader.question_only_reward_fn from the same grader file. 


## 3.4 Experiments

**Generation hyperparameters**:  When generating responses we will sanple with temperature 1.0, top p 1.0, max generation length 512. The r1_zero prompts ask the model to end its answer with the string </answer>, so when using those prompts we can direct vLLM to stop when the model outputs this string:

```py
sampling_params['stop'] = ['</answer>']
sampling_params['include_stop_str_in_output'] = True
```


**Problem Run OLMo-2-0425-1B on GSM8K**:

(a) Write a script to evaluate OLMo-2-0425-1B performance on GSM8K with zero-shot question_only, zero-shot r1_zero, and few-shot r1_zero_three_shot prompts.

Then, run your script and observe the outputs. For each prompt, how many model generations fall into each of the following categories: 
(1) correct with both format and correctness reward 1, 
(2) format reward 1 and correctness reward 0, 
(3) format reward 0 and correctness reward 0? 
Observing at least ten examples of category 2, how many model outputs are actually correct but just not parsed properly? What about category 3?

(b)  Observing the model outputs, characterize the model’s behavior with each prompt. For example, if we want the model to answer the question, is it enough to just provide the question, or does the model exhibit other behaviors besides just answering the question? How do the zero-shot r1_zero and few-shot r1_zero_three_shot prompts shape the model’s behavior?