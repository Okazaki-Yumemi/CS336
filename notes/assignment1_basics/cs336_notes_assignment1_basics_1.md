## 2.1 The Unicode standard

Unicode is a text encoding standard that maps characters to integer code points. As of Unicode 17.0 
(released in September 2025), the standard defines 159,801 characters across 172 scripts. For example, the 
character “s” has the code point 115 (typically notated as U+0073, where U+ is a conventional prefix and 
0073 is 115 in hexadecimal), and the character “牛” has the code point 29275. In Python, you can use the 
ord() function to convert a single Unicode character into its integer representation. The chr() function 
converts an integer Unicode code point into a string with the corresponding character.

```python
>>> ord('牛')
29275
>>> chr(29275)
'牛'
```

## Problem: Understanding Unicode

(a) What Unicode character does chr(0) return?

>chr(0) represents the null character, which acts as a string terminator and can cause hidden or truncated data in programming and databases.

(b) How does this character’s string representation (__repr__()) differ from its printed 
representation?

> __repr__() provides a detailed, unambiguous string for developers, while the printed representation (__str__()) gives a user-friendly, readable string.

(c) What happens when this character occurs in text? It may be helpful to play around with the 
following in your Python interpreter and see if it matches your expectations

```py

>>> chr(0)
>>> print(chr(0))
>>> "this is a test" + chr(0) + "string"
>>> print("this is a test" + chr(0) + "string")
```

上面四行的输出结果如下：

```py
>>> chr(0)
'\x00'
>>> print(chr(0))

>>> "this is a test" + chr(0) + "string"
'this is a test\x00string'
>>> print("this is a test" + chr(0) + "string")
this is a teststring
```

## 2.2 Unicode Encodings

Unicode characters can be represented in different ways using various encodings. The most common encodings are UTF-8, UTF-16, and UTF-32. Each encoding has its own way of representing Unicode code points as sequences of bytes.

```py

>>> test_string = "hello! こんにちは!"
>>> utf8_encode = test_string.encode('utf-8')
>>> print(utf8_encode)
b'hello! \xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf!'
>>> print(type(utf8_encode))
<class 'bytes'>
>>> list(utf8_encode)
[104, 101, 108, 108, 111, 33, 32, 227, 129, 147, 227, 130, 147, 227, 129, 171, 227, 129, 161, 227, 129, 175, 33]
>>> print(len(test_string))
13
>>> print(len(utf8_encode))
23
>>> print(utf8_encode.decode('utf-8'))
hello! こんにちは!
```

通过把Unicode转换为 sequence of bytes,我们可以 taking a sequence of code points 然后把它们转换为 a sequence of byte values.

The 256-length byte vocabulary 十分方便管理。我们也不用担心 out-of-vocabulary tokens,因为我们知道任何输入都会被表达为 a sequence of integers from 0 to 255.

### Problem: Unicode Encodings(3 points)

(a) What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than 
UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various 
input strings

> 因为 UTF-8 is more space-efficient for texts that are primarily in the ASCII range, as it uses only one byte for these characters. UTF-16 and UTF-32 use more bytes for the same characters, which can lead to larger file sizes and increased memory usage. Additionally, UTF-8 is backward compatible with ASCII, making it easier to handle legacy systems and data.

(b) Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string 
into a Unicode string. Why is this function incorrect? Provide an example of an input byte 
string that yields incorrect results.
```py
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
return "".join([bytes([b]).decode("utf-8") for b in bytestring])
>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
```

> 这个 function is incorrect because it attempts to decode each byte individually, which can lead to errors when dealing with multi-byte characters in UTF-8. For example, the character "こ" (U+3053) is represented by the byte sequence `\xe3\x81\x93` in UTF-8. If we pass this byte sequence to the function, it will try to decode each byte separately, resulting in a `UnicodeDecodeError` or incorrect characters.

(c) Give a two-byte sequence that does not decode to any Unicode character(s)

> A two-byte sequence that does not decode to any valid Unicode character is `b'\xC0\xAF'`. This sequence is invalid in UTF-8 because it represents an overlong encoding, which is not allowed in the UTF-8 standard.

## 2.3 Subword Tokenization

Subword tokenization 是在 charater-level 和 word-level中的一个折中。 它在 tardes off a large vocabulary size for better compression of the input byte sequence.

For example, if the byte sequence `b'the'` often occurs in our raw text training data,assigning it an entry in the vocabulary would reduce this 3-token sequence to a single token.

为了寻找subword tokens， 于是BPE就诞生了

## 2.4 BPE Tokenizer Training
Three main steps.

1. Vocabulary initialization: initial vocabulary is the set of all bytes. initial vocabulary is of size 256.
2. Pre-tokenization: count how often btyes occur next to each other in your text and begin merging them starting with the most frequent pair of bytes.
But there is a problem: first it's too slow. and directly merging bytes across the corpus may result in tokens that differ only in punctuation.

To avoid this,we pre-tokenize the corpus. 

最简单的pre-tokenization方法就是 splitting on whitespace. like `s.split()`

Most modern tokenizers use a regex-based pre-tokenizer.
```py
>>> PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
>>> import regex as re
>>> re.findall(PAT,"some text that i'll pre-tokenize")
['some', ' text',' that', ' i', "'ll", ' pre', '-', 'tokenize']
```

3. Compute BPE merges: 我们一般不考虑 在pre-token之间出现的pair. 然后在计算merge的时候，deterministically break ties in pair frequency by prefering the lexicographically greater pair.(这个词的意思是: 字典序更大的pair)

For example,如果  (“A”, “B”), (“A”, “C”), (“B”, “ZZ”), and (“BA”, “A”) 都出现了，

我们选择 ("BA", "A") 作为merge pair, 因为它是字典序最大的pair. (先比较第一个，第一个相同再比较第二个)
```py
>>> max([("A", "B"), ("A", "C"), ("B", "ZZ"), ("BA", "A")])
('BA', 'A')
```

**Example(bpe_example)：BPE Training example**:
corpus consisting of the following text
```
low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
```
and the vocabulary has a special token <|endoftext|>

**Vocabulary**
We initiallize our vocabulary with our special token <|endoftext|> and all the 256 bytes

**Pre-tokenization**
为了方便假设其为空格分界
我们可以得到
{low: 5 , lower: 2, widest: 3, newest: 6}

用 `dict[tuple[bytes,...],int]` 来标识. eg. `dict[("l","o","w"),5]` 表示 "low" 出现了5次

**Merges**
We first look at every successive pair of bytes and sum the frequency of the words where they appear.

{lo: 7, ow: 7, we: 8, er: 2, wi: 3, id: 3, de: 3, es: 9, st: 9, ne: 6, ew: 6}

第一步，我们选择('s','t')作为merge pair, 因为它出现了9次，且是字典序最大的pair. 然后我们将所有的('s','t')替换为('st')，得到新的词频统计：

{(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,e,st): 3, (n,e,w,e,st): 6}.

第二轮中，我们发现 (e,st)是频率最高的，然后合并
 {(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,est): 3, (n,e,w,est): 6}

## 2.5 Experimenting with BPE tokenizer Training

**parallelizing pre-tokenization**:

speed up pre-tokenization by parallelizing your code with built-in library multiprocessing.

分块. Ensuring your chunk boundaries occur at the begining of a special token.

**Removing special tokens before pre-tokenization**:
Before running pre-tokenization with the regex pattern, you should strip out all special tokens from your corpus.

This can be done using `re.split` with `"|".join(special_tokens)` as the delimiter

**Optimizing the merging step**:

Naive implementation of BPE is slow because for every merge , it iterates over all byte pairs to identify the most frequent pair.

However, the only pair counts that change after each merge are those that overlap with the merged pair.

BPE training speed can be improved by indexxing the counts of all pairs and incrementally updating these counts,rather than explicitly iterating over each pair of bytes to count pair frequencies.

**Problem (train_bpe):BPE Tokenizer Training**:

**Deliverable**: Write a function that,given a path to an input text file,trains a BPE tokenizer.

**Input**:

input_path: str, Path to a text file with BPE tokenizer training data.
vocab_size: int A positive integer that defines the maximum final vocabulary size
special_tokens: list[str] A list of strings to add to the vocabulary. During training, treat them as hard boundaries that prevent merges across their spans

**Output**:
vocab: dict[int,bytes] Tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes(token bytes)

merges: list[tuple[bytes,bytes]] A list of BPE merges produced from training. Each list item is a tuple of bytes (<token1> , <token2>) , representing that <token1> was merged with <token2>.
The merges should be ordered by order of creation

To test, first implement the test adapter at adapters.run_train_bpe

Then run `uv run pytest tests/test_train_bpe.py` to run the tests.





## 2.6 BPE tokenizer: Encoding and Decoding

在之前我们写好了 BPE 的 训练算法，我们可以得到 vocab 词表和 merges列表，现在我们将开始写BPE tokenizer that loads a provided vocabulary and list of merges and uses them to encode and decode text to / from token IDs

### 2.6.1 Encoding text

**Step 1 : Pre-tokenize** 先用UTF-8 来pre-token，和训练过程一样。

**Step 2 : Apply the merges**: 按照merges列表的顺序，依次将pre-tokenized的tokens进行merge，直到没有更多的merges可以应用。


**Example**:
句子 'the cat ate'

vocabulary is {0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'}

learned merges are [(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')].

首先pre-tokenizer 会把句子变成
['the','cat','ate']

第一个pre-token会被变成 [b't', b'h', b'e']

然后我们运用merge列表，把第一个pre-token变成[b'th', b'e']
然后下一次合并是[b'th',b'e'] 第一个pre-token变成[b'the']

其对应的token ID是9

这样的规则执行下来，cat会变成 [b' c', b'a', b't'],也就是[7, 1, 5]

ate变成[b'at', b'e'],也就是[10, 3]

所以一整个句子变成了
[9, 7, 1, 5, 10, 3]

###  解码时候的一个细节，无效 UTF-8

单个token不一定能够独立变成一个字符,例如 "牛" 的 UTF-8 字节是

e7 89 9b

要解码应该这么处理

```py
byte_stream = b"".join(vocab[token_id] for token_id in token_ids)

text = byte_stream.decode("utf-8",errors="replace")
```

special token单独放进vocab还不够，必须作为一整个整体进入vocab，拿到token ID


### Decoding

Decoding简单很多，直接把token id查表拼接输出就好了


### 2.6 还有一个重点，流式编码

```py
encode_iterable(self,iterable:Iterable[str]) -> Iterator[int]:
```

可以分块读取，但是必须保留边界附近尚未确定完整的文本，不能把任意输入chunk作为天然的tokenizer边界
