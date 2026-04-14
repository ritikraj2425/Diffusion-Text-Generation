import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from dataset import create_dataloader  # Ensure your dataset.py is in the same folder
import math

# --- Architecture Components ---

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class MaskedDiffusionModel(nn.Module):
    def __init__(self, vocab_size, d_model=768, nhead=12, num_layers=12, max_seq_len=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        out = self.transformer(x)
        return self.fc_out(out)

# --- Training Logic ---

def apply_forward_masking(x_0, mask_id, special_ids):
    batch_size, seq_len = x_0.shape
    device = x_0.device
    t = torch.rand(1).item()
    t = max(t, 0.1) # Minimum 10% masking for better learning
    
    rand_probs = torch.rand((batch_size, seq_len), device=device)
    is_special = torch.isin(x_0, torch.tensor(special_ids, device=device))
    is_mask = (rand_probs < t) & (~is_special)
    
    x_t = x_0.clone()
    x_t[is_mask] = mask_id
    return x_t, is_mask, t

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model():
    # 1. SETUP DEVICE (NVIDIA CUDA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on: {device}")

    # 2. LOAD VOCAB
    with open("subword_tokenizer.json", "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
        vocab = vocab_data["model"]["vocab"]
        
    vocab_size = len(vocab)
    mask_id = vocab["[MASK]"]
    special_ids = [vocab["[PAD]"], vocab["[BOS]"], vocab["[EOS]"], vocab["[UNK]"]]

    # 3. INITIALIZE MODEL (MAX POWER VALUES)
    model = MaskedDiffusionModel(
        vocab_size=vocab_size,
        d_model=768, 
        nhead=12, 
        num_layers=12, 
        max_seq_len=128
    ).to(device)

    # PRINT PARAMETER COUNT
    print(f" Model Capacity: {count_parameters(model):,} parameters")

    # 4. OPTIMIZER & DATALOADER
    # Larger batch size for NVIDIA GPUs
    dataloader, _ = create_dataloader("tokenized_data.json", batch_size=64) 
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    
    epochs = 200
    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        total_raw_ce = 0
        
        for x_0 in dataloader:
            x_0 = x_0.to(device)
            optimizer.zero_grad()
            
            x_t, is_mask, t = apply_forward_masking(x_0, mask_id, special_ids)
            logits = model(x_t)
            
            # Loss Calculation
            loss_per_token = F.cross_entropy(logits.view(-1, vocab_size), x_0.view(-1), reduction='none').view_as(x_0)
            masked_loss = (loss_per_token * is_mask.float()).sum() / (is_mask.sum() + 1e-8)
            
            # Diffusion scaling
            scaled_loss = masked_loss * min(1.0 / (t + 1e-5), 5.0)
            
            scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_raw_ce += masked_loss.item()

        avg_error = total_raw_ce / len(dataloader)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | True Error (CE): {avg_error:.4f}")

    torch.save(model.state_dict(), "diffusion_model.pth")
    print("✅ Training complete! Weights saved.")

if __name__ == "__main__":
    train_model()