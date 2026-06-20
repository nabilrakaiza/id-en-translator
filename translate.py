import torch
import pickle
import os
from src import build_transformer
from src.utils import causal_mask

# Hardcoded constants matching training
VOCAB_SIZE = 16000
SPECIAL_TOKENS = {"[PAD]": 16000, "[SOS]": 16001, "[EOS]": 16002}
TOTAL_VOCAB = VOCAB_SIZE + 3
MAX_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def greedy_decode(model, src, src_mask, max_len, sos_idx, eos_idx):
    """Predicts tokens one by one auto-regressively."""
    memory = model.encode(src, src_mask)

    ys = torch.ones(1, 1).fill_(sos_idx).type_as(src.data)

    for i in range(max_len - 1):
        tgt_mask = causal_mask(ys.size(1)).type_as(src.data)
        out = model.decode(memory, src_mask, ys, tgt_mask)

        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)

        if next_word == eos_idx:
            break

    return ys

def translate(text, model, src_tokenizer, tgt_tokenizer):
    model.eval()

    src_ids = src_tokenizer.encode(text)
    src_tokens = [SPECIAL_TOKENS["[SOS]"]] + src_ids + [SPECIAL_TOKENS["[EOS]"]]
    src_tensor = torch.tensor(src_tokens, dtype=torch.long).unsqueeze(0).to(DEVICE)

    src_mask = (src_tensor != SPECIAL_TOKENS["[PAD]"]).unsqueeze(0).unsqueeze(0).int().to(DEVICE)

    with torch.no_grad():
        tgt_tensor = greedy_decode(
            model, src_tensor, src_mask, MAX_LEN,
            SPECIAL_TOKENS["[SOS]"], SPECIAL_TOKENS["[EOS]"]
        )

    tgt_ids = tgt_tensor.squeeze(0).tolist()
    tgt_ids = [idx for idx in tgt_ids if idx not in SPECIAL_TOKENS.values()]

    return tgt_tokenizer.decode(tgt_ids)

if __name__ == "__main__":
    CHECKPOINT_DIR = "/content/drive/MyDrive/id-en-translator/"

    print("Loading tokenizers...")
    with open(os.path.join(CHECKPOINT_DIR, "src_tokenizer.pkl"), "rb") as f: src_tokenizer = pickle.load(f)
    with open(os.path.join(CHECKPOINT_DIR, "tgt_tokenizer.pkl"), "rb") as f: tgt_tokenizer = pickle.load(f)

    print("Loading model...")
    model = build_transformer(TOTAL_VOCAB, TOTAL_VOCAB)

    checkpoints = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")])
    latest = os.path.join(CHECKPOINT_DIR, checkpoints[-1])
    print(f"Loading checkpoint: {latest}")

    checkpoint = torch.load(latest, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)

    # Test translations
    test_sentences = ["aku suka kamu", "selamat pagi", "terima kasih", "saya tidak mengerti"]
    for text in test_sentences:
        print(f"\nIndonesian: {text}")
        print(f"English:    {translate(text, model, src_tokenizer, tgt_tokenizer)}")