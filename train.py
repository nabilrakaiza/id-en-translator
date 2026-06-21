import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
import pickle
import re

from src import RegexTokenizer, TranslationDataset, get_corpus_iterator, build_transformer
from src.utils import causal_mask, save_checkpoint
from translate import translate

# Hyperparameters
VOCAB_SIZE = 16000
SPECIAL_TOKENS = {"[PAD]": 16000, "[SOS]": 16001, "[EOS]": 16002}
TOTAL_VOCAB = VOCAB_SIZE + 3
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 40
D_MODEL = 512
WARMUP_STEPS = 4000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "/content/drive/MyDrive/id-en-translator/"

def rate(step, d_model=D_MODEL, warmup=WARMUP_STEPS):
    """Transformer learning rate schedule with warmup."""
    if step == 0:
        step = 1
    return (d_model ** -0.5) * min(step ** -0.5, step * warmup ** -1.5)

def main():
    print(f"Using device: {DEVICE}")
    os.makedirs("weights", exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # 1. Load Data
    print("Loading opus_books dataset...")
    raw_dataset = load_dataset("Helsinki-NLP/opus-100", "en-id", split="train[:50000]")

    # 2. Tokenizer Setup
    src_tokenizer_path = os.path.join(CHECKPOINT_DIR, "src_tokenizer.pkl")
    tgt_tokenizer_path = os.path.join(CHECKPOINT_DIR, "tgt_tokenizer.pkl")

    if os.path.exists(src_tokenizer_path) and os.path.exists(tgt_tokenizer_path):
        print("Loading pre-trained tokenizers...")
        with open(src_tokenizer_path, "rb") as f: src_tokenizer = pickle.load(f)
        with open(tgt_tokenizer_path, "rb") as f: tgt_tokenizer = pickle.load(f)
    else:
        print("Training tokenizers (this may take a while)...")
        src_tokenizer = RegexTokenizer()
        src_tokenizer.train(get_corpus_iterator(raw_dataset, "id"), vocab_size=VOCAB_SIZE)
        src_tokenizer.register_special_tokens(SPECIAL_TOKENS)

        tgt_tokenizer = RegexTokenizer()
        tgt_tokenizer.train(get_corpus_iterator(raw_dataset, "en"), vocab_size=VOCAB_SIZE)
        tgt_tokenizer.register_special_tokens(SPECIAL_TOKENS)

        with open(src_tokenizer_path, "wb") as f: pickle.dump(src_tokenizer, f)
        with open(tgt_tokenizer_path, "wb") as f: pickle.dump(tgt_tokenizer, f)

    # 3. Data Loader
    train_dataset = TranslationDataset(raw_dataset, src_tokenizer, tgt_tokenizer, MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 4. Model Setup
    print("Building Transformer...")
    model = build_transformer(TOTAL_VOCAB, TOTAL_VOCAB, N=6, d_model=D_MODEL, d_ff=2048, h=8, dropout=0.1)
    model.to(DEVICE)

    criterion = nn.NLLLoss(ignore_index=SPECIAL_TOKENS["[PAD]"])
    optimizer = torch.optim.Adam(model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=rate)

    # 5. Resume logic
    START_EPOCH = 0
    latest_checkpoint = None

    if os.path.exists(CHECKPOINT_DIR):
        checkpoints = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")])
        if checkpoints:
            latest_checkpoint = os.path.join(CHECKPOINT_DIR, checkpoints[-1])

    if latest_checkpoint:
        print(f"Resuming from {latest_checkpoint}...")
        ckpt = torch.load(latest_checkpoint)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        START_EPOCH = ckpt["epoch"] + 1
        print(f"Resuming from epoch {START_EPOCH}")

    # 6. Training Loop
    test_sentences = ["aku suka kamu", "selamat pagi", "terima kasih"]

    for epoch in range(START_EPOCH, EPOCHS):
        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            encoder_input = batch["encoder_input"].to(DEVICE)
            decoder_input = batch["decoder_input"].to(DEVICE)
            label = batch["label"].to(DEVICE)
            src_mask = batch["src_mask"].to(DEVICE)

            tgt_pad_mask = batch["tgt_pad_mask"].to(DEVICE)
            tgt_causal_mask = causal_mask(decoder_input.size(1)).to(DEVICE)
            tgt_mask = tgt_pad_mask & tgt_causal_mask

            out = model(encoder_input, decoder_input, src_mask, tgt_mask)
            out = out.view(-1, out.size(-1))
            label = label.view(-1)

            loss = criterion(out, label)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Completed | Average Loss: {avg_loss:.4f}")

        # Sanity check translations
        print("Sanity check:")
        for s in test_sentences:
            print(f"  {s} → {translate(s, model, src_tokenizer, tgt_tokenizer)}")

        # Save checkpoint
        new_checkpoint_path = os.path.join(CHECKPOINT_DIR, f"transformer_epoch_{epoch+1}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": avg_loss,
        }, new_checkpoint_path)
        print(f"Checkpoint saved: transformer_epoch_{epoch+1}.pt")

        # Delete old checkpoints, keep only latest
        old_checkpoints = sorted(
            glob.glob(os.path.join(CHECKPOINT_DIR, "transformer_epoch_*.pt")),
            key=lambda x: int(re.search(r"epoch_(\d+)\.pt", x).group(1))
        )
        for old in old_checkpoints[:-1]:
            os.remove(old)
            print(f"Deleted old checkpoint: {os.path.basename(old)}")

if __name__ == "__main__":
    main()