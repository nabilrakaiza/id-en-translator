import torch
import pickle
from src import build_transformer
from src.utils import causal_mask

# Hardcoded constants matching training
VOCAB_SIZE = 8000
SPECIAL_TOKENS = {"[PAD]": 8000, "[SOS]": 8001, "[EOS]": 8002}
TOTAL_VOCAB = VOCAB_SIZE + 3
MAX_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def greedy_decode(model, src, src_mask, max_len, sos_idx, eos_idx):
    """Predicts tokens one by one auto-regressively."""
    memory = model.encode(src, src_mask)
    
    # Start the decoder with the SOS token
    ys = torch.ones(1, 1).fill_(sos_idx).type_as(src.data)
    
    for i in range(max_len - 1):
        tgt_mask = causal_mask(ys.size(1)).type_as(src.data)
        out = model.decode(memory, src_mask, ys, tgt_mask)
        
        # Pass through the generator to get vocab probabilities
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        
        if next_word == eos_idx:
            break
            
    return ys

def translate(text, model, src_tokenizer, tgt_tokenizer):
    model.eval()
    
    # Encode source text
    src_ids = src_tokenizer.encode(text)
    src_tokens = [SPECIAL_TOKENS["[SOS]"]] + src_ids + [SPECIAL_TOKENS["[EOS]"]]
    src_tensor = torch.tensor(src_tokens, dtype=torch.long).unsqueeze(0).to(DEVICE)
    
    # Create source mask
    src_mask = (src_tensor != SPECIAL_TOKENS["[PAD]"]).unsqueeze(0).unsqueeze(0).int().to(DEVICE)
    
    # Decode
    with torch.no_grad():
        tgt_tensor = greedy_decode(
            model, src_tensor, src_mask, MAX_LEN, 
            SPECIAL_TOKENS["[SOS]"], SPECIAL_TOKENS["[EOS]"]
        )
    
    # Convert token IDs back to text
    tgt_ids = tgt_tensor.squeeze(0).tolist()
    # Remove SOS and EOS for final string
    tgt_ids = [idx for idx in tgt_ids if idx not in SPECIAL_TOKENS.values()]
    
    return tgt_tokenizer.decode(tgt_ids)

if __name__ == "__main__":
    print("Loading tokenizers...")
    with open("weights/src_tokenizer.pkl", "rb") as f: src_tokenizer = pickle.load(f)
    with open("weights/tgt_tokenizer.pkl", "rb") as f: tgt_tokenizer = pickle.load(f)

    print("Loading model...")
    model = build_transformer(TOTAL_VOCAB, TOTAL_VOCAB)
    
    # Load your latest checkpoint here
    checkpoint = torch.load("weights/transformer_epoch_10.pt", map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)

    # Test Translation
    input_text = "Saya suka belajar pembelajaran mesin."
    translation = translate(input_text, model, src_tokenizer, tgt_tokenizer)
    
    print(f"\nIndonesian: {input_text}")
    print(f"English:    {translation}")