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