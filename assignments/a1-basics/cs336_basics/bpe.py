import os
import re
import regex

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes,bytes]]]:
    
    vocab = {i: bytes([i]) for i in range(256)}
    vocab.update(
        {
            i + 256: token.encode("utf-8") 
            for i, token in enumerate(special_tokens)
        }
    )
    merges: list[tuple[bytes, bytes]] = []
    
    with open(input_path, encoding="utf-8") as f:
        text = f.read()
        # 拿到segment
        segments = split_on_special_tokens(text, special_tokens)
        
        # 先定义
        pretoken_counts: dict[str, int] = {}
            
        for segment in segments:
            # 对每个segment进行pre-tokenization
            pre_tokens = regex.finditer(PAT, segment)
            for pre_token in pre_tokens:
                pre_token_str = pre_token.group()
                if pre_token_str in pretoken_counts:
                    pretoken_counts[pre_token_str] += 1
                else:
                    pretoken_counts[pre_token_str] = 1
        
        # pretoken_counts 长这样-> {'hello': 5, 'world': 3}
        
        # 我们现在拿到了pretoken_counts ， 然后我们得给它转换成 tokenized_pretokens ， 用 utf8编码
        tokenized_pretokens: dict[tuple[bytes, ...], int] = {}
        # tokenized_pretokens 长这样 # {
        #     (b"h", b"e", b"l", b"l", b"o"): 5,
        #     (b"w", b"o", b"r", b"l", b"d"): 3,
        # }
        for pre_token_str, count in pretoken_counts.items():
            pre_token_bytes = tuple(bytes([b]) for b in pre_token_str.encode("utf-8"))
            tokenized_pretokens[pre_token_bytes] = count
        
        # 现在我们拿到了 tokenized_pretokens,开始统计初始的pair_counts
        
        pair_counts: dict[tuple[bytes, bytes], int] = {}
        
        # 用更新好的函数
        pair_counts = count_pairs(tokenized_pretokens)
        
        # 开始迭代，从pair_counts中找到出现次数最多的pair，进行合并，直到vocab_size达到要求
        while len(vocab) < vocab_size:
            if not pair_counts:
                break # 空值拦截，提前退出
            else:
                # 选出出现次数最多的pair，如果次数相同，选出字典序最大的pair
                best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
                # best_pair 合并
                merges.append(best_pair)
                # 更新vocab
                new_token = b"".join(best_pair)
                vocab[len(vocab)] = new_token
                # 更新pre_token序列
                # 我们首先要对 tokenized_pretokens 进行更新，合并best_pair
                updated_tokenized_pretokens: dict[tuple[bytes, ...], int] = {}
                
                for old_sequence, count in tokenized_pretokens.items():
                    new_sequence = merge_pair_in_sequence(old_sequence,best_pair)
                    if new_sequence in updated_tokenized_pretokens:
                        updated_tokenized_pretokens[new_sequence] += count
                    else:
                        updated_tokenized_pretokens[new_sequence] = count
                tokenized_pretokens = updated_tokenized_pretokens
                # 现在要更新pair_counts，我选择单独写一个函数
                # 更新pair_counts，然后别的也更新了，完成！
                pair_counts = count_pairs(tokenized_pretokens)
        
        return vocab, merges
    
def split_on_special_tokens(
    text: str,
    special_tokens: list[str],
) -> list[str]:
    """
    根据正则表达式，进行pretokenization，返回一个list，里面是分割后的segment
    """
    # special token为空列表时，直接返回原始文本
    if not special_tokens:
        return [text]
    else:
        # 对每个special token用re.escape进行转义，然后用'|'连接成一个正则表达式
        # special token按照长度从长到短排序，避免短的token匹配到长的token的一部分
        sorted_special_tokens = sorted(special_tokens, key=len, reverse=True)
        pattern = '|'.join(re.escape(token) for token in sorted_special_tokens)
        # 使用re.split()函数根据正则表达式进行分割
        # 返回时过滤空字符串
        return [segment for segment in re.split(pattern, text) if segment]
    
def merge_pair_in_sequence(
    sequence: tuple[bytes, ...],
    pair: tuple[bytes, bytes],
) -> tuple[bytes, ...]:
    """
    输入之前的tokenized pretoken sequence，和要合并的pair，返回合并后的sequence
    例子: (b"a", b"b", b"c", b"d", b"e"), (b"c", b"d") -> (b"a", b"b", b"cd", b"e")
    
    """
    merged_sequence: list[bytes] = []
    i = 0
    while i < len(sequence):
        if i < len(sequence) - 1 and sequence[i] == pair[0] and sequence[i + 1] == pair[1]:
            merged_sequence.append(b"".join(pair))
            i += 2 # 指针跳过已经合并的下一个token
        else:
            #非合并token,直接添加到merged_sequence中
            merged_sequence.append(sequence[i])    
            i += 1
    
    return tuple(merged_sequence)

def count_pairs(
    tokenized_pretokens: dict[tuple[bytes, ...], int],
) -> dict[tuple[bytes, bytes], int]:
    """
    输入tokenized_pretokens 的序列，自动计数，返回pair_counts类型的字典
    """
    
    pair_counts: dict[tuple[bytes, bytes], int] = {}
    
    for tokenized_pretoken , count in tokenized_pretokens.items():
        for i in range(len(tokenized_pretoken) - 1):
            pair = (tokenized_pretoken[i], tokenized_pretoken[i + 1])
            if pair in pair_counts:
                pair_counts[pair] += count
            else:
                pair_counts[pair] = count
    return pair_counts
    