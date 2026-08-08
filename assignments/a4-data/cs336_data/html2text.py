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