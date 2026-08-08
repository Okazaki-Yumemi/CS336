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
    lr= 0.1,
    epoch=500,
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