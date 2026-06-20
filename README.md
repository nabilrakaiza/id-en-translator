# ID-EN Translator: Vanilla Transformer & Custom Tokenizer

A from-scratch implementation of an Indonesian-to-English translation machine. This repository features a completely custom, vanilla Transformer architecture (based on *Attention Is All You Need*) and a custom Byte-Pair Encoding (BPE) `RegexTokenizer`.

## Features
* **Vanilla PyTorch Transformer**: Custom implementation of Multi-Head Attention, Positional Encoding, and Encoder/Decoder stacks.
* **Custom RegexTokenizer**: A ground-up BPE tokenizer utilizing GPT-4 style regex splitting.
* **Memory-Efficient Data Pipeline**: Streams data directly from Hugging Face's `datasets` library using generators to prevent memory bloat during tokenizer training.

## Repository Structure
```text
id-en-translator/
├── src/
│   ├── __init__.py
│   ├── tokenizer.py       # Custom RegexTokenizer (BPE)
│   ├── model.py           # Vanilla Transformer components
│   ├── dataset.py         # PyTorch Dataset and iterator logic
│   └── utils.py           # Masking, architecture builder, checkpointing
├── weights/               # Ignored by git; stores .pt models and .pkl tokenizers
│   └── .gitkeep
├── train.py               # Main training loop
├── translate.py           # Inference script with greedy decoding
├── requirements.txt       
├── .gitignore             
└── README.md