import numpy as np
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
eos_id = tokenizer.eos_token_id

input_path = "data/filtered/sample.txt"
output_path = "data/filtered/train.bin"

num_docs = 0
num_tokens = 0

with open(input_path, "r", encoding="utf-8") as f_in, \
     open(output_path, "wb") as f_out:

    for line in f_in:
        text = line.strip()
        if not text:
            continue

        ids = tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        ids.append(eos_id)

        np.array(ids, dtype=np.uint16).tofile(f_out)

        num_docs += 1
        num_tokens += len(ids)

print("documents:", num_docs)
print("tokens:", num_tokens)