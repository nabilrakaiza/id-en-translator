import torch
from torch.utils.data import Dataset

def get_corpus_iterator(dataset, lang_key):
    """Yields sentences one by one to prevent memory bloat."""
    for item in dataset:
        yield item["translation"][lang_key]

class TranslationDataset(Dataset):
    def __init__(self, raw_dataset, src_tokenizer, tgt_tokenizer, max_len=128):
        self.dataset = raw_dataset
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len
        
        # Special token indices
        self.pad_idx = 8000
        self.sos_idx = 8001
        self.eos_idx = 8002

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        src_text = item["translation"]["id"]
        tgt_text = item["translation"]["en"]

        # Encode
        src_ids = self.src_tokenizer.encode(src_text)
        tgt_ids = self.tgt_tokenizer.encode(tgt_text)

        # Add SOS and EOS
        src_tokens = [self.sos_idx] + src_ids + [self.eos_idx]
        tgt_tokens = [self.sos_idx] + tgt_ids + [self.eos_idx]

        # Truncate
        src_tokens = src_tokens[:self.max_len]
        tgt_tokens = tgt_tokens[:self.max_len]

        # Pad
        src_padding = self.max_len - len(src_tokens)
        tgt_padding = self.max_len - len(tgt_tokens)

        src_tokens += [self.pad_idx] * src_padding
        tgt_tokens += [self.pad_idx] * tgt_padding

        encoder_input = torch.tensor(src_tokens, dtype=torch.long)
        decoder_input = torch.tensor(tgt_tokens[:-1], dtype=torch.long)
        label = torch.tensor(tgt_tokens[1:], dtype=torch.long)

        # Create padding masks (1 for real tokens, 0 for PAD)
        src_mask = (encoder_input != self.pad_idx).unsqueeze(0).unsqueeze(0).int()
        tgt_pad_mask = (decoder_input != self.pad_idx).unsqueeze(0).unsqueeze(0).int()

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "label": label,
            "src_mask": src_mask,
            "tgt_pad_mask": tgt_pad_mask
        }