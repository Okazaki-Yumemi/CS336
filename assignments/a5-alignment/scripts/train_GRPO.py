import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.vllm_utils import VLLMServer
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

# 按你自己的文件名修改
from cs336_alignment.Grpo_train_step import grpo_train_step


# ============================================================
# 1. Configuration
# ============================================================

MODEL_ID = "allenai/OLMo-2-0425-1B"

TRAIN_DEVICE = "cuda:0"
VLLM_GPU = 1

SEED = 0

GROUP_SIZE = 8

# 正式 assignment 是：
# rollout_batch_size = 256 responses
# group_size = 8
# -> 32 prompts per rollout step
PROMPTS_PER_STEP = 32

GRADIENT_ACCUMULATION_STEPS = 32

LEARNING_RATE = 1e-5
MAX_GRAD_NORM = 1.0

NUM_STEPS = 200

TRAIN_FILE = "data/gsm8k/train.jsonl"
PROMPT_FILE = "cs336_alignment/prompts/r1_zero.prompt"


# ============================================================
# 2. Small utility functions
# ============================================================

def load_jsonl(path: str) -> list[dict]: # 加载jsonline文件，变成列表
    examples = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    return examples


def extract_gsm8k_ground_truth(answer: str) -> str:
    """
    GSM8K answer normally looks like:

        some rationale ...
        #### 42

    The reward function wants only "42".
    """
    return answer.rsplit("####", 1)[-1].strip()


def tensor_to_number(x):
    if torch.is_tensor(x):
        return x.detach().float().cpu().item()
    return x


# ============================================================
# 3. Load dataset + prompt
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)

train_examples = load_jsonl(TRAIN_FILE)

prompt_template = Path(PROMPT_FILE).read_text(
    encoding="utf-8"
)


# ============================================================
# 4. Load HuggingFace TRAINING policy
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

policy = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
)

policy = policy.to(TRAIN_DEVICE) # type: ignore
policy.train()


optimizer = torch.optim.AdamW(
    policy.parameters(),
    lr=LEARNING_RATE,
    betas=(0.9, 0.95),
    weight_decay=0.0,
)

optimizer.zero_grad(set_to_none=True)


# ============================================================
# 5. Start vLLM INFERENCE policy
# ============================================================

vllm_server = VLLMServer(
    model_id=MODEL_ID,
    gpu=VLLM_GPU,
    seed=SEED,
    gpu_memory_utilization=0.9,
)

vllm_server.start()

# 创建：
#
#   training GPU 0  <----NCCL---->  inference GPU 1
#
vllm_server.init_weight_sync(
    policy_device=TRAIN_DEVICE,
)


# ============================================================
# 6. Sampling configuration
# ============================================================

sampling_params = {
    "temperature": 1.0,
    "max_tokens": 512,

    # 最重要：
    # 每个 prompt 采 GROUP_SIZE 条 response
    "n": GROUP_SIZE,

    "seed": SEED,

    # r1_zero 会以 </answer> 结束
    "stop": ["</answer>"],
    "include_stop_str_in_output": True,
}


# ============================================================
# 7. RL training loop
# ============================================================

try:

    for step in range(NUM_STEPS):

        # ----------------------------------------------------
        # A. Sample PROMPTS_PER_STEP problems
        # ----------------------------------------------------

        examples = random.sample(
            train_examples,
            PROMPTS_PER_STEP,
        )

        prompts = [
            prompt_template.format(
                question=example["question"]
            )
            for example in examples
        ]

        ground_truths = [
            extract_gsm8k_ground_truth(
                example["answer"]
            )
            for example in examples
        ]


        # ----------------------------------------------------
        # B. Sync current HF policy -> vLLM
        # ----------------------------------------------------

        vllm_server.sync_policy_weights(policy)


        # ----------------------------------------------------
        # C. vLLM generates rollouts
        # ----------------------------------------------------

        completions = vllm_server.generate_completions(
            prompts=prompts,
            sampling_params={
                **sampling_params,

                # 每个 rollout step 改一下 seed
                "seed": SEED + step,
            },

            # 注意：
            # 这是 vLLM HTTP request batching，
            # 不是 GRPO training microbatch size。
            batch_size=8,
        )

        rollout_responses = [
            completion.text
            for completion in completions
        ]


        # ----------------------------------------------------
        # D. Convert:
        #
        # prompts:
        #   [p1, p2, ...]
        #
        # into:
        #   [p1,p1,...8x, p2,p2,...8x]
        #
        # to align with rollouts
        # ----------------------------------------------------

        repeated_prompts = [
            prompt
            for prompt in prompts
            for _ in range(GROUP_SIZE)
        ]

        repeated_ground_truths = [
            gt
            for gt in ground_truths
            for _ in range(GROUP_SIZE)
        ]


        expected_num_rollouts = (
            PROMPTS_PER_STEP * GROUP_SIZE
        )

        assert len(rollout_responses) == expected_num_rollouts
        assert len(repeated_prompts) == expected_num_rollouts
        assert len(repeated_ground_truths) == expected_num_rollouts


        # ----------------------------------------------------
        # E. One GRPO optimizer update
        # ----------------------------------------------------

        loss, metadata = grpo_train_step(
            model=policy,
            tokenizer=tokenizer,
            optimizer=optimizer,

            gradient_accumulation_steps=(
                GRADIENT_ACCUMULATION_STEPS
            ),
            max_grad_norm=MAX_GRAD_NORM,

            reward_fn=r1_zero_reward_fn,

            repeated_prompts=repeated_prompts,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths,

            group_size=GROUP_SIZE,

            baseline="mean",
            advantage_normalizer="std",

            importance_reweighting_method="none",

            loss_normalization="sequence",
        )


        # ----------------------------------------------------
        # F. Logging
        # ----------------------------------------------------

        printable_metadata = {
            key: tensor_to_number(value)
            for key, value in metadata.items()
        }

        print(
            f"step={step:03d}",
            f"loss={tensor_to_number(loss):.6f}",
            printable_metadata,
        )


        # Qualitative inspection
        if step % 10 == 0:
            print("\nQUESTION:")
            print(examples[0]["question"])

            print("\nROLLOUT:")
            print(rollout_responses[0])

            print("\nGROUND TRUTH:")
            print(ground_truths[0])

            print("=" * 80)


finally:
    vllm_server.stop()