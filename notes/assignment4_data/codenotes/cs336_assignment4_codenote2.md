# language identification.

```py
import fasttext


model = fasttext.load_model("local-shared-data/classifiers/lid.176.bin")

def identify_language(text: str) -> tuple[str, float]:
    text = text.replace("\n", " ")
    label, confidence = model.predict(text)
    return label[0].replace("__label__", ""), float(confidence[0]) # type: ignore

def main():
    
    print(identify_language("欢迎来到我们的网站"))



if __name__ == "__main__":
    main()
```

```bash
==== test session starts ====
platform linux -- Python 3.12.3, pytest-8.3.5, pluggy-1.5.0
rootdir: /home/soyo/projects/CS336-2026/assignments/a4-data
configfile: pyproject.toml
plugins: anyio-4.13.0, timeout-2.4.0, jaxtyping-0.3.2, furu-0.0.31, hydra-core-1.3.2
collected 21 items / 19 deselected / 2 selected                                                                                                                                                                                       

tests/test_langid.py::test_identify_language_english PASSED
tests/test_langid.py::test_identify_language_chinese_simplified PASSED

==== 2 passed, 19 deselected in 0.13s ====
```

一开始忘了加上 `text = text.replace("\n", " ")`，导致测试用例失败。因为 fastText 的语言识别模型在训练时没有考虑换行符，所以输入中有换行符会导致识别结果不准确。


(c)

```
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
```




```bash
(cs336-data) (base) soyo@localhost:~/projects/CS336-2026/assignments/a4-data$ uv run cs336_data/languageIdentification.py 
Language: zh, Confidence: 0.9608736634254456
Language: zh, Confidence: 0.9902801513671875
Language: zh, Confidence: 0.9242303967475891
Language: zh, Confidence: 0.9970689415931702
Language: zh, Confidence: 0.9923343062400818
Language: zh, Confidence: 0.9676010608673096
Language: zh, Confidence: 0.9651224613189697
Language: en, Confidence: 0.7118707299232483
Language: ru, Confidence: 0.994608461856842
Language: ru, Confidence: 0.9790343046188354
Language: de, Confidence: 0.9181560277938843
Language: zh, Confidence: 0.993803858757019
Language: zh, Confidence: 0.8872329592704773
Language: el, Confidence: 0.9986318349838257
Language: en, Confidence: 0.8817716836929321
Language: zh, Confidence: 0.8872559070587158
Language: zh, Confidence: 0.9762156009674072
Language: zh, Confidence: 0.9676953554153442
Language: en, Confidence: 0.11044929921627045
Language: en, Confidence: 0.9568921327590942
Language: en, Confidence: 0.11044929921627045
Language: ja, Confidence: 0.9922927021980286
Language: zh, Confidence: 0.9486950635910034
Language: zh, Confidence: 0.9809586405754089
Language: nl, Confidence: 0.8178791403770447
Language: zh, Confidence: 0.9890671372413635
Language: zh, Confidence: 0.9070631265640259
Language: zh, Confidence: 0.9882130026817322
Language: zh, Confidence: 0.9910210967063904
Language: zh, Confidence: 0.6539320349693298
Language: zh, Confidence: 0.5781425833702087
Language: ru, Confidence: 0.9842963814735413
Language: ru, Confidence: 0.9899467825889587
Language: en, Confidence: 0.11044929921627045
Language: zh, Confidence: 0.9861137270927429
Language: zh, Confidence: 0.8912011981010437
Language: zh, Confidence: 0.9541097283363342
Language: zh, Confidence: 0.985442578792572
Language: zh, Confidence: 0.9876883625984192
Language: zh, Confidence: 0.974861741065979
Language: zh, Confidence: 0.9778426289558411
Language: ru, Confidence: 0.9729216694831848
Language: zh, Confidence: 0.9601818323135376
Language: ru, Confidence: 0.9998847246170044
Language: en, Confidence: 0.9218418598175049
Language: de, Confidence: 0.37443867325782776
Language: ko, Confidence: 0.9302152991294861
Language: en, Confidence: 0.8041372299194336
Language: en, Confidence: 0.11044929921627045
Language: en, Confidence: 0.926743745803833
Language: tr, Confidence: 0.9926760196685791
Language: en, Confidence: 0.11044929921627045
Language: zh, Confidence: 0.9960695505142212
Language: en, Confidence: 0.36531123518943787
Language: en, Confidence: 0.9453595876693726
Language: en, Confidence: 0.11044929921627045
Language: id, Confidence: 0.8264915943145752
Language: zh, Confidence: 0.9797539114952087
Language: zh, Confidence: 0.9934940934181213
Language: fr, Confidence: 0.9906957149505615
```