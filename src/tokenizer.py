## improved version
import regex as re

class RegexTokenizer:
    def __init__(self):
        self.is_trained = False
        self.vocab = {}
        self.merges = {}
        self.special_tokens = {}
        self.inverse_special_tokens = {}
        self.GPT4_SPLIT_PATTERN = re.compile(r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""")

    def register_special_tokens(self, special_tokens: dict[str, int]):
        """Registers tokens like {'[PAD]': 16000, '[SOS]': 16001}"""
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def _get_stats(self, tokens_list: list[list[int]]):
        stats = {}
        for tokens in tokens_list:
            for i in range(len(tokens) - 1):
                temp = tokens[i], tokens[i + 1]
                if temp not in stats:
                    stats[temp] = 0

                stats[temp] += 1

        return stats

    def _merge(self, tokens_list, pair_bytes, new_byte):
        new_tokens_list = []

        for tokens in tokens_list:
            n = len(tokens)
            i = 0

            new_tokens = []
            while i < n - 1:
                temp = tokens[i], tokens[i + 1]
                if temp == pair_bytes:
                    new_tokens.append(new_byte)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            
            if i == n-1:
                new_tokens.append(tokens[i])

            new_tokens_list.append(new_tokens)

        return new_tokens_list
    
    def _add_new_bytes(self, tokens_list, new_byte):
        pair_bytes_count = self._get_stats(tokens_list)
        
        if not pair_bytes_count:
            return None, None
        
        pair_bytes = max(pair_bytes_count, key=pair_bytes_count.get)

        if pair_bytes_count[pair_bytes] == 1:
            return None, None

        new_tokens_list = self._merge(tokens_list, pair_bytes, new_byte)
        return pair_bytes, new_tokens_list

    def _bpe(self, tokens_list: list[list[int]], current_size: int, vocab_size: int, verbose = False):
        merges = {}

        while current_size < vocab_size:
            pair_bytes, temp = self._add_new_bytes(tokens_list, current_size)

            if not temp:
                break

            if verbose:
                print(f"merging {pair_bytes} into a new token {current_size}")

            tokens_list = temp 
            merges[pair_bytes] = current_size
            current_size += 1

        vocab = {idx:bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]

        return tokens_list, merges, vocab

    def train(self, corpus, vocab_size, verbose=False):
        self.is_trained = True 
        text = " ".join(corpus)
        splits = re.findall(self.GPT4_SPLIT_PATTERN, text)
        tokens_splits = [list(map(int, split.encode("utf-8"))) for split in splits]

        new_tokens_list, merges, vocab = self._bpe(tokens_splits, 256, vocab_size, verbose)
        self.merges = merges
        self.vocab = vocab

    def encode(self, text, allowed_special="all"):
        if not self.is_trained:
            return None
        
        if allowed_special == "all" and text in self.special_tokens:
            return [self.special_tokens[text]]
        
        splits = re.findall(self.GPT4_SPLIT_PATTERN, text)
        tokens_splits = [list(map(int, split.encode("utf-8"))) for split in splits]

        # Process merges on a per-chunk basis (Vastly faster inference)
        for i, tokens in enumerate(tokens_splits):
            while len(tokens) >= 2:
                stats = { (tokens[j], tokens[j+1]): j for j in range(len(tokens) - 1) }
                # Find pair that was merged earliest during training
                pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
                if pair not in self.merges:
                    break
                idx = self.merges[pair]
                # In-place chunk substitution optimization loop
                new_tokens = []
                k = 0
                while k < len(tokens) - 1:
                    if (tokens[k], tokens[k+1]) == pair:
                        new_tokens.append(idx)
                        k += 2
                    else:
                        new_tokens.append(tokens[k])
                        k += 1
                if k == len(tokens) - 1:
                    new_tokens.append(tokens[k])
                tokens = new_tokens
            tokens_splits[i] = tokens

        return [t for chunk in tokens_splits for t in chunk]

    def decode(self, ids):
        if not self.is_trained:
            return None
        
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                # Convert the special index back to its string format block representation safely
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"Invalid token ID: {idx}")
                
        return b"".join(part_bytes).decode(encoding="utf-8", errors="replace")
