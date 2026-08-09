# Filter data for language modeling

Now that we've implemented a variety of primitives for filtering web crawl data, let's put it to use and generate some language modeling training data.

本节我们的目的是用刚刚我们实现的各种过滤原语来过滤网络爬取的数据，并生成一些用于语言建模的训练数据。

例子

```py

import concurrent.futures
import os

from tqdm import tqdm

def process_single_wet_file(input_path: str, output_path: str):
    # Implement the logic to process a single WET file and filter the data
    pass
  
# Set up the executor

num_cpus = len(os.sched_getaffinity(0))  # Get the number of available CPUs
executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus)
wet_filepaths = ["a.warc.wet.gz", "b.warc.wet.gz", "c.warc.wet.gz"]  # List of WET file paths to process
output_dir = "/path/to/output_directory"  # Directory to save the filtered output files

futures = []

for wet_filepath in wet_filepaths:

  wet_filename = str(pathlib.Path(wet_filepath).name)
  future = executor.submit(
    process_single_wet_file,
    wet_filepath,
    os.path.join(output_dir, wet_filename),
  )

  #Store the future
  futures.append(future)

# Iterate over the completed futures as they finish, using a progress bar
# to keep track of progress

for future in tqdm(
  concurrent.futures.as_completed(futures),
  total = len(wet_filepaths)
):
  output_path = future.result()  # Get the result of the completed future
  print(f"Processed and saved filtered data to: {output_path}")  # Print the output path of the processed file
```

We also suggest using the FastWARC library to iterate over records in each WET file, and the tldextractlibrary to extract domains from URLs for filtering. In particular, these classes may be helpful


**Problem: Filter data for language modeling**:

(a) Write a script to filter language modeling data from a collection of Common Crawl WET files (located under /shared-data/english-wet-data). You are free to apply any of the primitives we’ve implemented in earlier parts of the assignment, and you’re also free to explore other filters and methods for generating data (e.g., filtering based on n-gram language model perplexity). Your goal is to produce data that, when trained on, minimizes the perplexity on the C4 100 domains subset of the Paloma benchmark


```py
from fastwarc.warc import ArchiveIterator , WarcRecordType
import gzip
from cs336_data.html2text import extract_text_from_html_bytes
from cs336_data.languageIdentification import identify_language
from cs336_data.Pii import mask_emails,mask_phone_numbers,mask_ips
from cs336_data.harmful import identify_nsfw, identify_hate
from cs336_data.quality import gopher_quality_filter
from cs336_data.quality_classifier import classify_quality
from cs336_data.exactlinededup import exact_line_dedup
from cs336_data.minHash_LSH_dedup import dedup

def process_single_wet_file(input_path: str, output_path: str):
    stats = {
        "total": 0,
        "gopher_rejected": 0,
        "kept": 0,
        "emails_masked": 0,
        "phones_masked": 0,
        "ips_masked": 0,
        "nsfw_rejected": 0,
        "toxic_rejected": 0,
        "language_identified": 0,
    }
    
    with gzip.open(input_path, 'rb') as f_in, open(output_path, 'w', encoding = "utf-8") as f_out:
        
        for record in ArchiveIterator(f_in):
            if record.record_type != WarcRecordType.conversion:
                continue
            
            stats["total"] += 1
            
            text = record.reader.read().decode('utf-8', errors='replace')
            
            # Apply gopher quality filter
            if not gopher_quality_filter(text):
                stats["gopher_rejected"] += 1
                continue
            
            # Identify language
            language, confidence = identify_language(text)
            if language != "en" and confidence > 0.7:
                stats["language_identified"] += 1
                continue

            # NSFW
            label, score = identify_nsfw(text)
            if label == "nsfw" and score > 0.9:
                stats["nsfw_rejected"] += 1
                continue
            
            # toxic
            label, score = identify_hate(text)
            if label == "toxic" and score > 0.9:
                stats["toxic_rejected"] += 1
                continue            
            

            # Mask PII
            text, emails_masked = mask_emails(text)
            stats["emails_masked"] += emails_masked

            text, phones_masked = mask_phone_numbers(text)
            stats["phones_masked"] += phones_masked

            text, ips_masked = mask_ips(text)
            stats["ips_masked"] += ips_masked
            
            text = " ".join(text.split())
            
            f_out.write(text + "\n")
            stats["kept"] += 1
            
    return stats

if __name__ == "__main__":
    stats = process_single_wet_file(
        "local-shared-data/CC/example.warc.wet.gz",
        "data/filtered/sample.txt"
    )
    print("Processing complete. Stats:", stats)
```

结果
```bash
Processing complete. Stats: {
'total': 19637, 
'gopher_rejected': 3884, 
'kept': 7579, 
'emails_masked': 3525, 
'phones_masked': 4608, 
'ips_masked': 233, 
'nsfw_rejected': 34, 
'toxic_rejected': 16, 
'language_identified': 8124}
```

Tokenize:

```py

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
```


结果

```bash
(cs336-data) (base) soyo@localhost:~/projects/CS336-2026/assignments/a4-data$ uv run scripts/serialize_data.py 
Token indices sequence length is longer than the specified maximum sequence length for this model (1364 > 1024). Running this sequence through the model will result in indexing errors
documents: 7579
tokens: 13190339
```