import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
import pickle

from src import RegexTokenizer, TranslationDataset, get_corpus_iterator, build_transformer
from src.utils import causal_mask, save_checkpoint

# Hyperparameters
VOCAB_SIZE = 8000
SPECIAL_TOKENS = {"[PAD]": 8000, "[SOS]": 8001, "[EOS]": 8002}
TOTAL_VOCAB = VOCAB_SIZE + 3
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"Using device: {DEVICE}")
    os.makedirs("weights", exist_ok=True)

    # 1. Load Data
    print("Loading opus_books dataset...")
    raw_dataset = load_dataset("opus_books", "en-id", split="train[:5000]") # Subset for quick testing

    # 2. Tokenizer Setup
    src_tokenizer_path = "weights/src_tokenizer.pkl"
    tgt_tokenizer_path = "weights/tgt_tokenizer.pkl"

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
    model = build_transformer(TOTAL_VOCAB, TOTAL_VOCAB, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=SPECIAL_TOKENS["[PAD]"])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)

    # 5. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            encoder_input = batch["encoder_input"].to(DEVICE)
            decoder_input = batch["decoder_input"].to(DEVICE)
            label = batch["label"].to(DEVICE)
            src_mask = batch["src_mask"].to(DEVICE)
            
            # Combine pad mask and causal look-ahead mask for decoder
            tgt_pad_mask = batch["tgt_pad_mask"].to(DEVICE)
            tgt_causal_mask = causal_mask(decoder_input.size(1)).to(DEVICE)
            tgt_mask = tgt_pad_mask & tgt_causal_mask

            # Forward pass
            out = model(encoder_input, decoder_input, src_mask, tgt_mask)
            
            # Reshape for loss calculation: [batch_size * seq_len, vocab_size]
            out = out.view(-1, out.size(-1))
            label = label.view(-1)
            
            loss = criterion(out, label)
            total_loss += loss.item()

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1} Completed | Average Loss: {total_loss/len(train_loader):.4f}")
        save_checkpoint(model, optimizer, epoch, f"weights/transformer_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    main()