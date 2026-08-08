data_path = "data/quality/wiki_refs.warc.gz"
from fastwarc.warc import ArchiveIterator, WarcRecordType
from cs336_data.html2text import extract_text_from_html_bytes
from cs336_data.languageIdentification import identify_language
from cs336_data.quality import gopher_quality_filter


with open(data_path, 'rb') as f:
    total = 0
    extracted = 0
    english = 0
    quality = 0
    
    with open("data/quality/wiki_positive.txt", 'w') as pos_file:
        for record in ArchiveIterator(f):
            total += 1
            
            html_bytes = record.reader.read()
            try:
                text = extract_text_from_html_bytes(html_bytes)
            except Exception as e:
                continue
            
            one_line_text = " ".join(text.split())
            
            extracted += 1
            
            language, confidence = identify_language(one_line_text)
            
            if language != "en" or confidence < 0.7:
                continue
            
            english += 1
            
            if gopher_quality_filter(one_line_text) is False:
                continue
            
            quality += 1
            
            pos_file.write("__label__wiki " + one_line_text + "\n")

print(f"Total records: {total}")
print(f"Extracted text: {extracted}")
print(f"English text: {english}")
print(f"High quality text: {quality}")

