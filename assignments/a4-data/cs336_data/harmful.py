import fasttext
from cs336_data.html2text import extract_text_from_html_bytes
from fastwarc.warc import ArchiveIterator, WarcRecordType

model_nsfw = fasttext.load_model("local-shared-data/classifiers/dolma_fasttext_nsfw_jigsaw_model.bin")

model_hate = fasttext.load_model("local-shared-data/classifiers/dolma_fasttext_hatespeech_jigsaw_model.bin")


def identify_nsfw(text: str) -> tuple[str, float]:
    text = text.replace("\n", " ")
    label, confidence = model_nsfw.predict(text)
    return label[0].replace("__label__", ""), float(confidence[0]) # type: ignore

def identify_hate(text: str) -> tuple[str, float]:
    text = text.replace("\n", " ")
    label, confidence = model_hate.predict(text)
    return label[0].replace("__label__", ""), float(confidence[0]) # type: ignore


def main():

    warc_path = "local-shared-data/CC/example.warc.gz"

    with open(warc_path, 'rb') as f:
        for record in ArchiveIterator(f):
            if record.record_type == WarcRecordType.response:
                html_bytes = record.reader.read()
                text = extract_text_from_html_bytes(html_bytes)
                nsfw_label, nsfw_confidence = identify_nsfw(text)
                hate_label, hate_confidence = identify_hate(text)
                print(f"NSFW: {nsfw_label}, Confidence: {nsfw_confidence}")
                print(f"Hate Speech: {hate_label}, Confidence: {hate_confidence}")

if __name__ == "__main__":
    main()