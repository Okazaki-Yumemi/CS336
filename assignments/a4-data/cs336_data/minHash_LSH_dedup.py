import unicodedata

def normalize_text(text: str) -> str:
    """
    Normalize text by lowercasing and removing punctuation.
    """
    text = unicodedata.normalize('NFD', text)
    text = "".join(
        c for c in text 
        if unicodedata.category(c) != 'Mn'  # Remove diacritics
    )
    import re
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    # multiple spaces to single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text
 
def text_2_ngrams(text: str, n: int) -> set[str]:
    """
    Convert text to a set of n-grams.
    """
    words = text.split()
    ngrams = set()
    
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])
        ngrams.add(ngram)
    return ngrams

def minHash_signature(ngrams: set[str], num_hashes: int) -> list[int]:
    """
    Compute the MinHash signature for a set of n-grams.
    """
    import hashlib
    signature = []
    for i in range(num_hashes):
        min_hash = float('inf')
        for ngram in ngrams:
            hash_value = int(hashlib.md5((str(i) + ngram).encode()).hexdigest(), 16)
            min_hash = min(min_hash, hash_value)
        signature.append(min_hash)
    return signature

def signature_2_bands(signature: list[int], num_bands: int) -> list[tuple[int]]:
    """
    Convert a MinHash signature into bands for LSH
    """
    band_size = len(signature) // num_bands
    bands = []
    for i in range(num_bands):
        start = i * band_size
        end = start + band_size
        bands.append(tuple(signature[start:end]))
    return bands

import os

def dedup(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
   # 读取文件
    file_contents = []
    for file_path in input_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            file_contents.append(f.read())
            
    #normalize
    normalized_texts = [normalize_text(text) for text in file_contents]
    
    # 生成 n-grams
    ngrams_list = [text_2_ngrams(text, ngrams) for text in normalized_texts]
    
    # 计算 MinHash 签名
    signatures = [minHash_signature(ngrams_set, num_hashes) for ngrams_set in ngrams_list]
    
    # 将签名转换为 bands
    bands_list = [signature_2_bands(signature, num_bands) for signature in signatures]
    
    # bands_list 是一个二维列表，每个元素是一个文件的 bands
    # 我们需要将这些 bands 转换为一个字典，键是 band，值是文件索引的集合
    band_dict = {}
    for file_index, bands in enumerate(bands_list):
        for band_index, band in enumerate(bands):
            key = (band_index, band)
            if key not in band_dict:
                band_dict[key] = set()
            band_dict[key].add(file_index)

    # 对candidate pairs进行Jaccard相似度计算
    similar_files = set()
    for band, file_indices in band_dict.items():
        if len(file_indices) > 1:
            file_indices = list(file_indices)
            for i in range(len(file_indices)):
                for j in range(i + 1, len(file_indices)):
                    idx1, idx2 = file_indices[i], file_indices[j]
                    set1, set2 = ngrams_list[idx1], ngrams_list[idx2]
                    intersection = len(set1.intersection(set2))
                    union_size = len(set1.union(set2))
                    jaccard_sim = intersection / union_size if union_size != 0 else 0
                    if jaccard_sim >= jaccard_threshold:
                        similar_files.add((idx1, idx2))
    parent = list(range(len(input_files)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx
    
    for i, j in similar_files:
        union(i,j)    
    
    clusters = {}
    for i in range(len(input_files)):
        root = find(i)
        if root not in clusters:
            clusters[root] = []
        clusters[root].append(i)
    
    files_to_remove = set()
    
    for cluster in clusters.values():
        if len(cluster) > 1:
            # 保留第一个文件，删除其他文件
            for idx in cluster[1:]:
                files_to_remove.add(input_files[idx])
    
                    
    # 将不重复的文件写入输出目录
    for file_path in input_files:
        if file_path not in files_to_remove:
            file_name = os.path.basename(file_path)
            output_path = os.path.join(output_directory, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
   
        
    