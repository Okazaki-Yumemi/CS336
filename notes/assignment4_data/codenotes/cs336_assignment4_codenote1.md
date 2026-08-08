# HTML to text conversion 

# (a) 先实现 bytes → plain text

```
html_bytes
    ↓
优先尝试 UTF-8 decode
    ↓
成功？
 ┌──┴──┐
是     否
│      ↓
│   detect_encoding(...)
│      ↓
│   用检测出的 encoding decode
└──┬───┘
   ↓
Unicode HTML string
   ↓
extract_plain_text(...)
   ↓
plain text
```

```py
def extract_text_from_html_bytes(
    html_bytes: bytes
) -> str:
    # 检测decoding
    try:
        html_text = html_bytes.decode('utf-8')
    except UnicodeDecodeError as e:
        # 如果解码失败，尝试检测编码
        encoding = detect_encoding(html_bytes)
        html_text = html_bytes.decode(encoding, errors='replace')
    plain_text = extract_plain_text(html_text)
    return plain_text
```
即可，没啥难的


```bash
===== test session starts =====
platform linux -- Python 3.12.3, pytest-8.3.5, pluggy-1.5.0
rootdir: /home/soyo/projects/CS336-2026/assignments/a4-data
configfile: pyproject.toml
plugins: anyio-4.13.0, timeout-2.4.0, jaxtyping-0.3.2, furu-0.0.31, hydra-core-1.3.2
collected 21 items / 20 deselected / 1 selected                                                                                                                                                                                       

tests/test_extract.py::test_extract_text_from_html_bytes PASSED
===== 1 passed, 20 deselected in 0.03s =====
```



update:

```py
# raw HTML bytes -> Unicode HTML -> plain text
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding

from fastwarc.warc import ArchiveIterator, WarcRecordType

def extract_text_from_html_bytes(
    html_bytes: bytes
) -> str:
    # 检测decoding
    try:
        html_text = html_bytes.decode('utf-8')
    except UnicodeDecodeError as e:
        # 如果解码失败，尝试检测编码
        encoding = detect_encoding(html_bytes)
        html_text = html_bytes.decode(encoding, errors='replace')
    plain_text = extract_plain_text(html_text)
    return plain_text

warc_path = "local-shared-data/CC/example.warc.gz"

with open(warc_path, 'rb') as f:
    for record in ArchiveIterator(f):
        if record.record_type == WarcRecordType.response:
            html_bytes = record.reader.read()
            text = extract_text_from_html_bytes(html_bytes)
            print(text)
``` 

扫出来的结果就不放了，全是黄色网站。
