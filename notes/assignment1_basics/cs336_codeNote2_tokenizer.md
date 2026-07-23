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

