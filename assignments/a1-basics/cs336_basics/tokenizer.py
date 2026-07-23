from collections.abc import Iterable,Iterator
import regex
import pickle


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int , bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        #开始保存信息
        self.vocab = vocab
        # 测试vocab形状:
        # {0: b"!", 1: b'"', 2: b"#", ...}
        #反向词表
        self.token_to_id = {token: id for id, token in vocab.items()}
        
        self.merges = merges
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
        
        self.special_tokens = special_tokens if special_tokens is not None else []
        self.special_token_to_id = {
            token: self.token_to_id[token.encode("utf-8")]
            for token in self.special_tokens
        }
        
    
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        #用pickle实现
        with open(vocab_filepath,"rb") as vocab_file:
            vocab = pickle.load(vocab_file)
        with open(merges_filepath,"rb") as merges_file:
            merges = pickle.load(merges_file)
        
        return cls(
            vocab = vocab,
            merges = merges,
            special_tokens = special_tokens
        )
        
    
    def encode(
        self, 
        text: str,
    ) -> list[int]:
        # 如果是空字符串，返回空列表
        if not text:
            return []
        else:
            if len(text) == 1:
                return self._encode_pretoken(text)  
            else:
                # special_token是空的,可以直接用 GPT-2 regex pre-tokenization处理
                segments = self._split_on_special_tokens(text)
                
                token_ids: list[int] = []
                for segment in segments:
                    # 如果segment是special_token,直接返回对应的id
                    if segment in self.special_tokens:
                        token_ids.append(self.special_token_to_id[segment])
                        continue
                    
                    
                    pre_tokens = regex.finditer(PAT, segment)
                    for pre_token in pre_tokens:
                        pre_token_str = pre_token.group()
                        token_ids.extend(self._encode_pretoken(pre_token_str))
                return token_ids


    def encode_iterable(
        self,
        iterable: Iterable[str],
    ) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)
    
    def decode(
        self,
        ids:list[int],
    ) -> str:
        
        byte_stream = b"".join(self.vocab[id] for id in ids)

        return byte_stream.decode("utf-8",errors="replace")
    
    
    def _encode_pretoken(self, pretoken: str) -> list[int]:
        """
        这个函数是BPE合并代码的核心部分，它将一个预分词的字符串（pretoken）转换为对应的token id列表。
        输入一个pretoken, 内部完成 UTF-8单字节序列，然后反复merge_rank合并
        """
        byte_tokens = tuple(bytes([byte]) for byte in pretoken.encode("utf-8"))
        
        while True:
            pairs = [(byte_tokens[i] , byte_tokens[i + 1]) for i in range(len(byte_tokens) - 1)]
            
            if not pairs:
                break
            
            merged_candidates = [(pair, self.merge_ranks.get(pair, float("inf"))) for pair in pairs]
            
            merged_candidates.sort(key=lambda x: x[1])
            
            if merged_candidates[0][1] == float("inf"):
                break
            
            best_pair = merged_candidates[0][0]
            
            new_byte_tokens = []
            
            i = 0
            
            while i < len(byte_tokens):
                if i < len(byte_tokens) - 1 and (byte_tokens[i], byte_tokens[i + 1]) == best_pair:
                    new_byte_tokens.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_byte_tokens.append(byte_tokens[i])
                    i += 1
            byte_tokens = tuple(new_byte_tokens)
        
        return [self.token_to_id[token] for token in byte_tokens]
        
    def _split_on_special_tokens(
        self,
        text: str,
    ) -> list[str]:
        """
        根据正则表达式，进行pretokenization，返回一个list，里面是分割后的segment
        """
        if not self.special_tokens:
            return [text]
        else:
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            # 这个地方不同bpe的是，我们需要保留 special token，所以我们要给正则表达式加上括号，
            # 表示捕获组，这样在分割的时候，special token也会被保留下来
            escaped_tokens = [
                regex.escape(token)
                for token in sorted_special_tokens
            ]
            pattern = f"({'|'.join(escaped_tokens)})"
            return [segment for segment in regex.split(pattern, text) if segment]
    
