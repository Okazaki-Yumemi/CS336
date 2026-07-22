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