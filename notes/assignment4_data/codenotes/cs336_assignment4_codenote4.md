```py
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
```




```bash

NSFW: non-nsfw, Confidence: 1.0000096559524536
Hate Speech: non-toxic, Confidence: 1.000009536743164
NSFW: non-nsfw, Confidence: 0.9978517293930054
Hate Speech: non-toxic, Confidence: 1.0000044107437134
NSFW: non-nsfw, Confidence: 1.0000100135803223
Hate Speech: non-toxic, Confidence: 1.0000100135803223
NSFW: non-nsfw, Confidence: 0.999592125415802
Hate Speech: non-toxic, Confidence: 0.9997073411941528
NSFW: non-nsfw, Confidence: 1.000006914138794
Hate Speech: non-toxic, Confidence: 1.0000028610229492
NSFW: nsfw, Confidence: 0.9981629252433777
Hate Speech: toxic, Confidence: 0.999985933303833
NSFW: nsfw, Confidence: 0.9999978542327881
Hate Speech: toxic, Confidence: 1.0000098943710327
NSFW: non-nsfw, Confidence: 0.9996693134307861
Hate Speech: non-toxic, Confidence: 0.9999959468841553
NSFW: non-nsfw, Confidence: 0.9998847246170044
Hate Speech: non-toxic, Confidence: 0.9996370077133179
NSFW: non-nsfw, Confidence: 0.9995582699775696
Hate Speech: non-toxic, Confidence: 0.9993264675140381
NSFW: non-nsfw, Confidence: 0.9986296892166138
Hate Speech: non-toxic, Confidence: 0.9969630241394043
```


测试均通过