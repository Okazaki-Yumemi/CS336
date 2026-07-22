# Tokenization:

## Intro_to_tokenization
raw text is generally represented as Unicode strings

a language model places a probability distribution over sequences of tokens 

so we need to convert the raw text into a sequence of tokens

A **tokenizer** is a class that implements the encode and decode methods

## Observations
- A word and its preceding space are part of the same token
- A word at the beginning and in the middle are represented differently

eg.
```py
tokenizer = get_gpt5_tokenizer()

string = "Hello, world!"

indices = tokenizer.encode(string)

reconstructed_string = tokenizer.decode(indices)

assert string == reconstructed_string
```

Compression ratio:number of bytes per token

## Character_tokenizer

字符级tokenizer

最简单的方法就是将每个字符作为一个token
eg

```py

assert ord("a") == 97
assert ord("b") == 98

# it can be converted back via chr.

assert chr(97) == "a"
assert chr(98) == "b"
```

那么字符级的分词器:
```py

tokenizer = CharacterTokenizer()
string = "Hello, world!"
indices = tokenizer.encode(string) #call ord
reconstructed_string = tokenizer.decode(indices) #call chr
assert string == reconstructed_string
```
- Problem 1: this is a very large vocabulary
- Problem 2: many characters are quite rare , which is inefficient of the vocabulary

## Byte_tokenizer

Unicode decoding UTF-8

```py
assert bytes("a", "utf-8") == b"a"
```

```py
tokenizer = ByteTokenizer()
string = "Hello, world!"
indices = tokenizer.encode(string) 
reconstructed_string = tokenizer.decode(indices)
assert string == reconstructed_string
```

The vocabulary is nice and small, a byte can represent 256 values

## word_tokenizer

Another appraoch is to split strings into words

```py

string = "I'll say supercalifragilisticexpialidocious!"

chunks = regex.findall(r"\w+|.", string)
```
each token is meaningful

vocabulary size = number of distinct chunks in the training corpus

- Many words are rare and the model won't learn much about them
- this does't obviously provide a fixed vocabulary size
- New words we haven't seen during training get a special token (like <unk>)

## BPE_tokenizer

Basic idea: train the tokenizer on raw text to constuct a vocabulary tailored to the data


-> start with each byte as a token , and successively merge the most common pairs of adajacent tokens

```py

def train_bpe(string: str, num_merges: int) -> BPETokenizerParams:  # @inspect string, @inspect num_merges
    text("Start with the list of bytes of `string`.")
    indices = list(map(int, string.encode("utf-8")))  # @inspect indices
    merges: dict[tuple[int, int], int] = {}  # index1, index2 => merged index
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}  # index -> bytes

    for i in range(num_merges):
        # Count the number of occurrences of each pair of tokens
        counts = count_adjacent_pairs(indices)  # @inspect counts @stepover

        # Find the most common pair
        pair = max(counts, key=counts.get)  # @inspect pair

        # Merge that pair
        new_index = 256 + i  # @inspect new_index
        merges[pair] = new_index  # @inspect merges
        vocab[new_index] = vocab[pair[0]] + vocab[pair[1]]  # @inspect vocab
        indices = merge(indices, pair, new_index)  # @inspect indices @stepover

    compression_ratio = get_compression_ratio(string, indices)  # @inspect compression_ratio

    return BPETokenizerParams(vocab=vocab, merges=merges)

```

例如 t和 h 组合成th之后，在词表的256+0=256的位置，th的字节表示为b'th'

反复合并，序列会越来越短，词表越来越大


**Using the tokenizer**:

```py

class BPETokenizer(Tokenizer):
    """BPE tokenizer given a set of merges and a vocabulary."""
    def __init__(self, params: BPETokenizerParams):
        self.params = params

    def encode(self, string: str) -> list[int]:
        indices = list(map(int, string.encode("utf-8")))  # @inspect indices
        # Note: this is a very slow implementation
        for pair, new_index in self.params.merges.items():  # @inspect pair, @inspect new_index
            indices = merge(indices, pair, new_index)  # @stepover
        return indices

    def decode(self, indices: list[int]) -> str:
        bytes_list = list(map(self.params.vocab.get, indices))  # @inspect bytes_list
        string = b"".join(bytes_list).decode("utf-8")  # @inspect string
        return string
```

using
```py

tokenizer = BPETokenizer(params)
string = "Hello, world!"
indices = tokenizer.encode(string)
reconstructed_string = tokenizer.decode(indices)
assert string == reconstructed_string
```

In assignment 1 , you will go beyond this in following ways:
- encode() currently loops over all merges, Only loop over merges that matter
- Detect and preserve special tokens like <unk> and <pad>
- Use pre-tokenization
- Try to make the implementation as fast as possible

