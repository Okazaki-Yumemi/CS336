# 这个笔记是记录我完成bpe的过程,逐步记录我的实现

# Step1 观察代码框架

`adapters.py`

```py
def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    return train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        **kwargs,
    )
```
本次实验我们要在adapters.py 里面把它更新为我们写的函数，adapters.py 里面的函数会被调用，在
`test_train_bpe` 里面被调用，检查如下

```py
def test_train_bpe_speed():
    """
    Ensure that BPE training is relatively efficient by measuring training
    time on this small dataset and throwing an error if it takes more than 1.5 seconds.
    This is a pretty generous upper-bound, it takes 0.38 seconds with the
    reference implementation on my laptop. In contrast, the toy implementation
    takes around 3 seconds.
    """
    input_path = FIXTURES_PATH / "corpus.en"
    start_time = time.time()
    _, _ = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )
    end_time = time.time()
    assert end_time - start_time < 1.5


def test_train_bpe():
    input_path = FIXTURES_PATH / "corpus.en"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )

    # Path to the reference tokenizer vocab and merges
    reference_vocab_path = FIXTURES_PATH / "train-bpe-reference-vocab.json"
    reference_merges_path = FIXTURES_PATH / "train-bpe-reference-merges.txt"

    # Compare the learned merges to the expected output merges
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(reference_merges_path, encoding="utf-8") as f:
        gpt2_reference_merges = [tuple(line.rstrip().split(" ")) for line in f]
        reference_merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_reference_merges
        ]
    assert merges == reference_merges

    # Compare the vocab to the expected output vocab
    with open(reference_vocab_path, encoding="utf-8") as f:
        gpt2_reference_vocab = json.load(f)
        reference_vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_reference_vocab.items()
        }
    # Rather than checking that the vocabs exactly match (since they could
    # have been constructed differently), we'll make sure that the vocab keys and values match
    assert set(vocab.keys()) == set(reference_vocab.keys())
    assert set(vocab.values()) == set(reference_vocab.values())


def test_train_bpe_special_tokens(snapshot):
    """
    Ensure that the special tokens are added to the vocabulary and not
    merged with other tokens.
    """
    input_path = FIXTURES_PATH / "tinystories_sample_5M.txt"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=1000,
        special_tokens=["<|endoftext|>"],
    )

    # Check that the special token is not in the vocab
    vocabs_without_specials = [word for word in vocab.values() if word != b"<|endoftext|>"]
    for word_bytes in vocabs_without_specials:
        assert b"<|" not in word_bytes

    snapshot.assert_match(
        {
            "vocab_keys": set(vocab.keys()),
            "vocab_values": set(vocab.values()),
            "merges": merges,
        },
    )
```
`test_train_bpe_speed()` 函数会调用run_train_bpe()，检查运行时间是否小于1.5秒，`test_train_bpe()` 函数会调用run_train_bpe()，检查返回的vocab和merges是否和参考文件一致，`test_train_bpe_special_tokens()` 函数会调用run_train_bpe()，检查特殊token是否被正确处理。

以上为基本代码的观察，从现在开始我们确认了我们要完成的一个
`bpe.py` 文件的基本框架

# Step2 开始完成bpe.py 文件

`bpe.py` 文件是从空文件彻底开始空空如也开始的

我们首先确定我们要些什么函数
经过思考，我们确认了以下四个函数

```py

def train_bpe(
  input_path: str | os.PathLike,
  vocab_size: int,
  special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]
  return vocab, merges
"""
作为主要的函数，读取文件，统计词频,然后实现bpe训练迭代,返回词表和merge记录 
"""
```
```py
def split_on_special_tokens(
  text: str,
  special_tokens: list[str],
) -> list[str]

"""
这个函数按照正则表达式，对文本进行pre-tokenization, 按照特殊token进行切分,返回切分后的token列表
"""
```
```py
def merge_pair_in_sequence(
  sequence: tuple[bytes, ...],
  pair: tuple[bytes, bytes],
) -> tuple[bytes, ...]
"""
输入之前的tokenized pretoken sequence, 和要合并的pair，
返回合并后的sequence
例：(b"a", b"b", b"c", b"d", b"e"), (b"c", b"d") -> (b"a", b"b", b"cd", b"e")
"""
```
```py
def count_pairs(
  tokenized_pretokens: dict[tuple[bytes, ...], int]
) -> dict[tuple[bytes, bytes], int]
"""
输入tokenized_pretokens 的序列，自动计数返回pair_counts, 统计每个pair出现的次数
"""
```

# Step3 先写辅助函数

```py

def split_on_special_tokens(
  text: str,
  special_tokens: list[str],
) -> list[str]:
  if not special_tokens:
    return [text] # 没有special tokens, 返回原始文本作为唯一的token
  else:
    sorted_special_tokens = sorted(special_tokens, key=len, reverse=True)
    # 降序排序特殊token, 避免短的token匹配到长的token的一部分

    pattern = '|'.join(re.escape(token) for token in sorted_special_tokens)
    # 正则表达式匹配特殊token, 使用re.escape() 转义特殊字符

    return [segment for segment in re.split(pattern, text) if segment]
```

```py

def merge_pair_in_sequence(
    sequence: tuple[bytes, ...],
    pair: tuple[bytes, bytes],
) -> tuple[bytes, ...]:

  merged_sequence : list[bytes] = []

  i = 0
  while i < len(sequence):
    if i < len(sequence) - 1 and sequence[i] == pair[0] and sequence[i + 1] == pair[1]:
      merged_sequence.append(b"".join(pair))  # 合并pair
      i += 2 # 跳过下一个token, 因为已经合并了
    else:
      merged_sequence.append(sequence[i])  # 保留原始token
      i += 1 # 继续下一个token
  return tuple(merged_sequence)  # 返回合并后的序列
```

```py

def count_pairs(
    tokenized_pretokens: dict[tuple[bytes, ...], int],
) -> dict[tuple[bytes, bytes], int]:

  pair_counts: dict[tuple[bytes, bytes], int] = {}

  for tokenized_pretoken , count in tokenized_pretokens.items():
    for i in range(len(tokenized_pretoken) - 1):
      pair = (tokenized_pretoken[i], tokenized_pretoken[i + 1])
      if pair in pair_counts:
        pair_counts[pair] += count
      else:
        pair_counts[pair] = count
  return pair_counts
```

# Step4 完成train_bpe函数

```py
def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes,bytes]]]:

  #初始化vocab和merges
  vocab = {i: bytes([i]) for i in range(256)}  # 初始vocab为所有单字节字符
  vocab.update(
    {
      i + 256: token.encode("utf-8") for i, token in enumerate(special_tokens)
    }
  )
  merges: list[tuple[bytes, bytes]] = []

  #打开文件
  with open(input_path, "r", encoding="utf-8") as f:
    text = f.read()

    # 拿到segments, 先按照特殊token切分
    segments = split_on_special_tokens(text, special_tokens)

    # 定义每个pretoken的计数器
    pretoken_counts: dict[str, int] = {}

    for segment in segments:
      # 对每个segment 进行 pre-tokenization
      pre_tokens = regex.finditer(PAT, segment)
      for pre_token in pre_tokens:
        pretoken_str = pre_token.group()
        if pretoken_str in pretoken_counts:
          pretoken_counts[pretoken_str] += 1
        else:
          pretoken_counts[pretoken_str] = 1
    
    # pretoken_counts 长这样-> {'hello': 5, 'world': 3}

    # 将pretoken_counts 转换为 tokenized_pretokens, 以便后续统计pair
    tokenized_pretokens: dict[tuple[bytes, ...], int] = {}

    for pretoken_str, count in pretoken_counts.items():
      pre_token_bytes = tuple(bytes([b]) for b in pretoken_str.encode("utf-8"))
      tokenized_pretokens[pre_token_bytes] = count

    # tokenized_pretokens 长这样 # {
    #     (b"h", b"e", b"l", b"l", b"o"): 5,
    #     (b"w", b"o", b"r", b"l", b"d"): 3,
    # }

    # 开始统计pair_counts, 并进行迭代合并
    pair_counts = count_pairs(tokenized_pretokens)

    #迭代
    while len(vocab) < vocab_size:
      if not pair_counts:
        break # 没有合并的了
      else:
        best_pair = max(pair_counts, key = lambda pair:(pair_counts[pair], pair)) # 找到出现次数最多的pair 如果出现次数相同, 按照字典序排序
        merges.append(best_pair) # 记录合并的pair

        # 更新vocab
        new_token = b"".join(best_pair)
        vocab[len(vocab)] = new_token
        # 更新tokenized_pretokens  
        updated_tokenized_pretokens: dict[tuple[bytes, ...], int] = {}

        for old_sequence, count in tokenized_pretokens.items():
          new_sequence = merge_pair_in_sequence(old_sequence, best_pair)
          if new_sequence in updated_tokenized_pretokens:
            updated_tokenized_pretokens[new_sequence] += count
          else:
            updated_tokenized_pretokens[new_sequence] = count
        
        tokenized_pretokens = updated_tokenized_pretokens
        pair_counts = count_pairs(tokenized_pretokens) # 重新统计pair_counts
  return vocab, merges
```


```bash
uv run pytest tests/test_train_bpe.py -vv

================================================================================ test session starts ================================================================================
platform linux -- Python 3.13.12, pytest-9.0.2, pluggy-1.6.0 -- /home/soyo/projects/CS336-2026/assignments/a1-basics/.venv/bin/python
cachedir: .pytest_cache
rootdir: /home/soyo/projects/CS336-2026/assignments/a1-basics
configfile: pyproject.toml
plugins: timeout-2.4.0, jaxtyping-0.3.9
collected 3 items                                                                                                                                                                   

tests/test_train_bpe.py::test_train_bpe_speed PASSED
tests/test_train_bpe.py::test_train_bpe PASSED
tests/test_train_bpe.py::test_train_bpe_special_tokens PASSED

3 passed in 7.19s 
```