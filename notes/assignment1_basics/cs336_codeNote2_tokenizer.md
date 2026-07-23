# 本次作业对应handout P11的 Problem(tokenizer)

作业要求

Implement a Tokenizer class that, given a vocabulary and a list of merges, encodes text into integer IDs and decodes integer IDs into text.

Your tokenizer should also support user-provided special tokens 

作业要求的接口

```python

def __init__(self,vocab,merges,special_tokens=None)

```

- vocab: dict[int,bytes]
- merges: list[tuple[bytes,bytes]]
- special_tokens: list[str] | None = None

```python
def from_files(cls,vocab_filepath,merges_filepath,special_tokens=None)
```

从序列化的vocab和merges文件中加载tokenizer

- vocab_filepath: str
- merges_filepath: str
- special_tokens: list[str] | None = None

```python
def encode(self,text:str)->list[int]
```

Encode an input text into a sequence of token IDs

```py
def encode_iterable(self,iterable:Iterable[str])-> Iterator[int]
```

Given an iterable of strings, return a generator that lazily yields token IDs

```py

def decode(self,ids:list[int])->str
```
decode a sequence of token IDs into text.

# Step1. 观察测试

测试分为五组

### 1. 基础可逆性

```
test_roundtrip_empty
test_roundtrip_single_character
test_roundtrip_single_unicode_character
test_roundtrip_ascii_string
test_roundtrip_unicode_string
```

至少要求
```py
decode(encode(text)) == text
```

### 2. 必须与 tiktoken完全一致

```
test_empty_matches_tiktoken
test_single_character_matches_tiktoken
test_unicode_string_matches_tiktoken
...
test_tinystories_matches_tiktoken
```

测试要求Token ID序列也必须和tokenizer想通过

### 3.Special token 边界情况

```
test_unicode_string_with_special_tokens
test_overlapping_special_tokens
test_encode_special_token_trailing_newlines
test_encode_special_token_double_newline_non_whitespace
```

overlapping special tokens的情况是指，两个special token的前缀是相同的，比如`<|special|>`和`<|special2|>`，在这种情况下，应该优先匹配最长的那个。

### 4. 多语言和真实预料
```py
test_address_...
test_german_...
test_tinystories_...
```
用于检查标点、数字、空格、Unicode 和大段文本。


### 5.流式和内存
```py
test_encode_iterable_tinystories_sample_roundtrip
test_encode_iterable_tinystories_matches_tiktoken
test_encode_iterable_memory_usage
test_encode_memory_usage
```

- encode_iterable 必须惰性产生 ID；
- 结果还要与参考 tokenizer 一致；
- 不能简单把整个 iterable 拼成一个巨大字符串；
- 普通 encode 也不能制造明显失控的中间对象。

# Step2. 开始实现

创建文件
`tokenizer.py`
先放接口骨架

# Step3. Tokenizer骨架

```py
class Tokenizer:
  def __init__(
    self,
    vocab: dict[int , bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
  )：
    self.vocab = vocab

    #反向词表，方便后面encode从 byte到数字
    self.token_to_id = {token: id for id , token in vocab.items()}

    self.merges = merges
    # rank merge， 越小越优先
    self.merge_ranks = {pair: rank for rank pair in enumerate(merges)}

    self.special_tokens = special_tokens if special_tokens is not None else []
    self.special_token_to_id = {
      token: self.token_to_id[token.encode("utf-8")]
      for token in self.special_tokens
    }

```



# Step4. decode

decode是最简单的，只需要避开我们先前提到的 "一个一个解码" 的问题，就能比较好的完成。

具体的做法是把整个输入的token id 都先解码进去，然后把剩下的工作交给decode


```py

def decode(self, ids:list[int]) -> str :
    byte_stream = b"".join(self.vocab[id] for id in ids)

    return byte_stream.decode("utf-8",errors = "replace")
```
把 id 里面的东西对应到词表里面去，因为词表是 key = int , value = bytes 的dict。

# Step5.准备一些辅助函数

首先是正则表达式分句的函数，因为我们输入的text里面必须想办法把句子用special token分割，然后special token在本处还需要保留，不能就是说和之前bpe一样直接扔掉

```py

def _split_on_special_token(self,text:str ,)->list[str]:
  """
  根据正则表达式进行pretokenization,返回list,里面是分割后的segment
  在这个地方，special_token会单独作为一个元素
  """
  if not self.special_tokens:
    return [text]
  else:
    sorted_special_tokens = sort(self.special_tokens, key = len , reverse = True)

    escaped_tokens = [
      regex.escape(token)
      for token in sorted_special_tokens
    ]
    pattern = f"({'|'.join(escaped_tokens)})"
    return [segment for segment in regex.split(pattern, text) if segment]
```

第二个辅助函数是用来在split之后，对bytes进行词表还原的
因为我们把一个字转换成bytes之后得到的都是最碎的utf-8单元，我们得给它拼接起来，按照merge规则

```py
def _encode_pretoken(self, pretoken:str) -> list[int]:
  """
  BPE合并，接受一个pretoken，把它转换成词表里面的list，同时merge合并
  """
  byte_tokens = tuple(bytes([byte]) for byte in pretoken.encode("utf-8"))

  while True:
    pairs = [(byte_tokens[i] , byte_tokens[i+1]) for i in range(len(byte_tokens) - 1)]

    if not pairs:
      break
    
    merged_candidates = [(pair , self.merge_ranks.get(pair,float("inf"))) for pair in pairs]

    merged_candidates.sort(key = lambda x:x[1])
    # 没的合并了
    if merged_candidates[0][1] == float("inf"):
      break
    
    best_pair = merged_candidates[0][0]

    new_byte_tokens = []

    i = 0

    while i < len(byte_tokens):
      if i < len(byte_tokens) - 1 and (byte_tokens[i],byte_token[i+1])== best_pair:
        new_byte_tokens.append(b"".join(best_pair))
        i += 2
      else:
        new_byte_tokens.append(byte_tokens[i])
        i += 1
    byte_tokens = tuple(new_byte_tokens)
  
  return [self.token_to_id[token] for token in byte_tokens]
```

# Step6 完成encode循环
这个encode我是逐步逐步写的，所以有些地方按理来说可能合并，不过按照思维分开了


```py

def encode(
  self,
  text:str
)-> list[int]:
  #如果空，返回滚木
  if not text:
    return []
  else:
    if len(text) == 1:
      return self._encode_pretoken(text)
    else:
      #先对数据做处理
      segments = self._split_on_special_tokens(self)

      token_ids = list[int] = []
      for segment in segments:
        if segment in self.special_tokens:
          token_ids.append(self.special_token_to_id[segment])
          continue
        
        pre_tokens = regex.finditer(PAT,segment)
        for pre_token in pretokens:
          pre_token_str = pre_token.group()
          token_ids.extend(self._encode_pretoken(pre_token_str))
      
      return token_ids
```


```py

def encode_iterable(
  self,
  iterable: Iterable[str],
) -> Iterator[int]:
  for chunk in iterable:
      yield from self.encode(chunk)

@classmethod
def from_files(
  cls,
  vocab_filepath: str,
  merges_filepath: str,
  special_tokens: list[str] | None = None,
) -> "Tokenizer":
  #用pickle实现
  with open(vocab_filepath,"rb") as vocab_file:
    vocab = pickle.load(vocab_file)
  with open(merges_filepath,"rb") as merges_file:
    merges = pickle.load(merges_file)
        
  return cls(
    vocab = vocab,
        merges = merges,
        special_tokens = special_tokens
    )

```

测试均通过，除Memory test预测失败.

```bash

uv run pytest tests/test_tokenizer.py


test session starts 
platform linux -- Python 3.13.12, pytest-9.0.2, pluggy-1.6.0
rootdir: /home/soyo/projects/CS336-2026/assignments/a1-basics
configfile: pyproject.toml
plugins: timeout-2.4.0, jaxtyping-0.3.9
collected 25 items                                                                     

tests/test_tokenizer.py::test_roundtrip_empty PASSED
tests/test_tokenizer.py::test_empty_matches_tiktoken PASSED
tests/test_tokenizer.py::test_roundtrip_single_character PASSED
tests/test_tokenizer.py::test_single_character_matches_tiktoken PASSED
tests/test_tokenizer.py::test_roundtrip_single_unicode_character PASSED
tests/test_tokenizer.py::test_single_unicode_character_matches_tiktoken PASSED
tests/test_tokenizer.py::test_roundtrip_ascii_string PASSED
tests/test_tokenizer.py::test_ascii_string_matches_tiktoken PASSED
tests/test_tokenizer.py::test_roundtrip_unicode_string PASSED
tests/test_tokenizer.py::test_unicode_string_matches_tiktoken PASSED
tests/test_tokenizer.py::test_roundtrip_unicode_string_with_special_tokens PASSED
tests/test_tokenizer.py::test_unicode_string_with_special_tokens_matches_tiktoken PASSED
tests/test_tokenizer.py::test_overlapping_special_tokens PASSED
tests/test_tokenizer.py::test_address_roundtrip PASSED
tests/test_tokenizer.py::test_address_matches_tiktoken PASSED
tests/test_tokenizer.py::test_german_roundtrip PASSED
tests/test_tokenizer.py::test_german_matches_tiktoken PASSED
tests/test_tokenizer.py::test_tinystories_sample_roundtrip PASSED
tests/test_tokenizer.py::test_tinystories_matches_tiktoken PASSED
tests/test_tokenizer.py::test_encode_special_token_trailing_newlines PASSED
tests/test_tokenizer.py::test_encode_special_token_double_newline_non_whitespace PASSED
tests/test_tokenizer.py::test_encode_iterable_tinystories_sample_roundtrip PASSED
tests/test_tokenizer.py::test_encode_iterable_tinystories_matches_tiktoken PASSED
tests/test_tokenizer.py::test_encode_iterable_memory_usage PASSED
tests/test_tokenizer.py::test_encode_memory_usage XFAIL (Tokenizer.encode is
expected to take more m...)

24 passed, 1 xfailed in 10.11s 
```