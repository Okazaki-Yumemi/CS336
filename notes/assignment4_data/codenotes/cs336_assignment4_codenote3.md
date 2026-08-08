```py
import re
EMAIL_PATTERN = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')

def mask_emails(text: str) -> tuple[str, int]:
    mask_text, count = re.subn(EMAIL_PATTERN, '|||EMAIL_ADDRESS|||', text)
    return mask_text, count

```

```py
#eg.
# 2831823829
#(283)-182-3829
#(283) 182 3829
##283-182-3829
PHONE_NUMBER_PATTERN = re.compile(
    r'(?<!\d)(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)'
)

def mask_phone_numbers(text: str) -> tuple[str, int]:
    mask_text, count = re.subn(PHONE_NUMBER_PATTERN, '|||PHONE_NUMBER|||', text)
    return mask_text, count
```

(?<!\d) 前面不能是数字

(?:\(\d{3}\)|\d{3}):

(?: A | B ) 匹配 A或者匹配 B

A = \(\d{3}\)  匹配 (283)

```
\(       字面意义的 (
\d{3}    三个数字
\)       字面意义的 )
```

B = \d{3} 匹配 283

[-.\s]? 匹配 - 或者 . 或者 空格，出现0次或者1次

最后
\d{3}[-.\s]?\d{4} 匹配 182-3829 或者 182.3829 或者 182 3829

(?!\d) 后面不能是数字


```py
## IPv4 masking

octet = r'(?:25[0-5]|2[0-4]\d|1?\d?\d)'
IPV4_PATTERN = re.compile(r'\b' + octet + r'\.' + octet + r'\.' + octet + r'\.' + octet + r'\b')

def mask_ips(text: str) -> tuple[str, int]:
    mask_text, count = re.subn(IPV4_PATTERN, '|||IP_ADDRESS|||', text)
    return mask_text, count
```

测试均通过。

