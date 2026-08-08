from cs336_data.html2text import extract_text_from_html_bytes
from fastwarc.warc import ArchiveIterator, WarcRecordType
import fasttext


model = fasttext.load_model("local-shared-data/classifiers/lid.176.bin")

def identify_language(text: str) -> tuple[str, float]:
    text = text.replace("\n", " ")
    label, confidence = model.predict(text)
    return label[0].replace("__label__", ""), float(confidence[0]) # type: ignore

def main():
    
    warc_path = "local-shared-data/CC/example.warc.gz"
    
    with open(warc_path, 'rb') as f:
        for record in ArchiveIterator(f):
            if record.record_type == WarcRecordType.response:
                html_bytes = record.reader.read()
                text = extract_text_from_html_bytes(html_bytes)
                language, confidence = identify_language(text)
                print(f"Language: {language}, Confidence: {confidence}")



if __name__ == "__main__":
    main()
    