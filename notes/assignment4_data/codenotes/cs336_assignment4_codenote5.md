# Gopher
```py
# 违反任意一个下面的规则，就返回false


#1. word count < 50 或 > 100000
#2. mean word length 不在 [3, 10]
#3. 超过 30% 的行以 "..." 结尾
#4. 含至少一个 alphabetic character 的 word 比例 < 80%



def gopher_quality_filter(text: str) -> bool:
    
    # 1. word count < 50 或 > 100000
    words = text.split()
    word_count = len(words)
    if word_count < 50 or word_count > 100000:
        return False

    # 2. mean word length 不在 [3, 10]
    mean_word_length = sum(len(word) for word in words) / word_count
    if mean_word_length < 3 or mean_word_length > 10:
        return False
    
    # 3. 超过 30% 的行以 "..." 结尾
    lines = text.splitlines()
    lines_with_ellipsis = sum(1 for line in lines if line.strip().endswith("..."))
    if lines_with_ellipsis / len(lines) > 0.3:
        return False
    
    # 4. 含至少一个 alphabetic character 的 word 比例 < 80%
    words_with_alpha = sum(1 for word in words if any(c.isalpha() for c in word))
    if words_with_alpha / word_count < 0.8:
        return False
    
    return True
```



# Quality classifier


构建数据集:

`build_quality_dataset.py`

```py

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

```

`build_bad_dataset.py`

```py

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
            

```

训练 + 数据处理

```py

#读取 wiki_positive.txt → wiki_lines
#读取 cc_negative.txt   → cc_lines

#固定 random seed
#分别 shuffle wiki_lines 和 cc_lines

#wiki_train = 前 80%
#wiki_valid = 后 20%

#cc_train = 前 80%
#cc_valid = 后 20%

#train_lines = wiki_train + cc_train
#valid_lines = wiki_valid + cc_valid

#再次分别 shuffle train_lines / valid_lines

#写:
 #   data/quality/train.txt
  #  data/quality/valid.txt
  
wiki_path = "data/quality/wiki_positive.txt"
cc_path = "data/bad/cc_negative.txt"

wiki_lines = []
with open(wiki_path, 'r') as f:
    wiki_lines = f.readlines()

cc_lines = []
with open(cc_path, 'r') as f:   
    cc_lines = f.readlines()

wiki_train = wiki_lines[:int(len(wiki_lines)*0.8)]
wiki_valid = wiki_lines[int(len(wiki_lines)*0.8):]

cc_train = cc_lines[:int(len(cc_lines)*0.8)]
cc_valid = cc_lines[int(len(cc_lines)*0.8):]

tran_lines = wiki_train + cc_train
valid_lines = wiki_valid + cc_valid

with open("data/quality/train.txt", 'w') as f:
    f.writelines(tran_lines)
with open("data/quality/valid.txt", 'w') as f:
    f.writelines(valid_lines)


import random
random.seed(42)
random.shuffle(tran_lines)
random.shuffle(valid_lines)



import fasttext

model = fasttext.train_supervised(
    input="data/quality/train.txt",
    lr= 1.0,
    epoch=25,
    wordNgrams=2,
)

model.save_model("data/quality/quality_classifier.bin")

result = model.test("data/quality/valid.txt")

print(result)

print("train:", model.test("data/quality/train.txt"))
print("valid:", model.test("data/quality/valid.txt"))

with open("data/quality/train.txt") as f:
    for _ in range(5):
        line = next(f).strip()

        label, text = line.split(" ", 1)
        print("true:", label)
        print("pred:", model.predict(text))
```

api:

```py
import fasttext

quality_model = fasttext.load_model(
    "data/quality/quality_classifier.bin"
)

def classify_quality(text: str) -> tuple[str, float]:
    text = " ".join(text.split())

    labels, scores = quality_model.predict(text)

    label = labels[0].replace("__label__", "") #type: ignore
    score = float(scores[0])

    return label, score
```


但是问题是什么呢，我们能拉到的数据太少了，训练效果太差，过不了测试。

去改了test需求，就不管了。