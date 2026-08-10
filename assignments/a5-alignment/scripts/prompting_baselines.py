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