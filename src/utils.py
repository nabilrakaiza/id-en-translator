import torch
import torch.nn as nn
import copy
from .model import (
    EncoderDecoder, Encoder, Decoder, EncoderLayer, DecoderLayer, 
    MultiHeadAttention, PositionWiseFeedForward, Embeddings, 
    PositionalEncoding, Generator
)

def build_transformer(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """Constructs the full Transformer model from hyperparameters."""
    c = copy.deepcopy
    attn = MultiHeadAttention(d_model, h, dropout)
    ff = PositionWiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab)
    )
    
    # Initialize parameters with Glorot / fan_avg
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

def causal_mask(size):
    """Creates a mask to hide future tokens in the decoder."""
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.int)
    return subsequent_mask == 0

def save_checkpoint(model, optimizer, epoch, path):
    """Saves model weights and optimizer state."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"Checkpoint saved to {path}")