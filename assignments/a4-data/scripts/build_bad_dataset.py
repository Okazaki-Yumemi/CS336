from fastwarc.warc import ArchiveIterator, WarcRecordType
from cs336_data.html2text import extract_text_from_html_bytes
from cs336_data.languageIdentification import identify_language
from cs336_data.quality import gopher_quality_filter

data_path = "local-shared-data/CC/example.warc.gz"

target_path = "data/bad/cc_negative.txt"

target = 68
count = 0
with open(data_path, 'rb') as f:
    with open(target_path, 'w') as out_file:
        for record in ArchiveIterator(f):
            if record.record_type == WarcRecordType.response:
                text = extract_text_from_html_bytes(record.reader.read())
                if text is None:
                    continue
                one_line_text = " ".join(text.split())
                
                language, confidence = identify_language(one_line_text)
                if language != "en" or confidence < 0.7:
                    continue
                
                if len(one_line_text) < 50:
                    continue
                
                out_file.write("__label__cc " + one_line_text + "\n")
                count += 1
                
                if count == target:
                    break
            
