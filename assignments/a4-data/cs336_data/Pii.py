import re
EMAIL_PATTERN = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')

def mask_emails(text: str) -> tuple[str, int]:
    mask_text, count = re.subn(EMAIL_PATTERN, '|||EMAIL_ADDRESS|||', text)
    return mask_text, count

#eg.
# 2831823829
#(283)-182-3829
#(283) 182 3829
#283-182-3829
PHONE_NUMBER_PATTERN = re.compile(
    r'(?<!\d)(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)'
)

def mask_phone_numbers(text: str) -> tuple[str, int]:
    mask_text, count = re.subn(PHONE_NUMBER_PATTERN, '|||PHONE_NUMBER|||', text)
    return mask_text, count


## IPv4 masking

octet = r'(?:25[0-5]|2[0-4]\d|1?\d?\d)'
IPV4_PATTERN = re.compile(r'\b' + octet + r'\.' + octet + r'\.' + octet + r'\.' + octet + r'\b')

def mask_ips(text: str) -> tuple[str, int]:
    mask_text, count = re.subn(IPV4_PATTERN, '|||IP_ADDRESS|||', text)
    return mask_text, count