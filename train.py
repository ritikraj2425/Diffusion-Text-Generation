import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from dataset import create_dataloader
from tokenizer import MAX_SEQ_LENGTH
import math

# ─────────────────────────────────────────────────────────────
#  Model Architecture
# ─────────────────────────────────────────────────────────────

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
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6, max_seq_len=MAX_SEQ_LENGTH):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True, activation="gelu",
            dropout=0.1, norm_first=True   # Pre-norm: more stable training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, src_key_padding_mask=None):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        out = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        return self.fc_out(out)


# ─────────────────────────────────────────────────────────────
#  Guided Masking — only mask the BOT response, not the user prompt
# ─────────────────────────────────────────────────────────────

def find_bot_start(sequence, bot_token_ids):
    """
    Find the position where 'bot' token appears.
    We only mask tokens AFTER this so the model learns:
    given user prompt → predict bot reply.
    """
    seq = sequence.tolist()
    for bid in bot_token_ids:
        for i in range(len(seq)):
            if seq[i] == bid:
                return i + 1  # Start masking after 'bot' token
    return len(seq) // 2  # Fallback: mask second half if 'bot' not found


def apply_guided_masking(x_0, mask_id, pad_id, special_ids, bot_token_ids):
    """
    FIX: Per-sample masking rates instead of per-batch.
    Each sample gets its own random masking rate t, drawn from [0.15, 1.0].
    This gives the model diverse training signals within each batch.
    """
    batch_size, seq_len = x_0.shape
    device = x_0.device

    x_t     = x_0.clone()
    is_mask = torch.zeros_like(x_0, dtype=torch.bool)

    for b in range(batch_size):
        # FIX: Per-sample masking rate (was: single t for entire batch)
        t = torch.rand(1).item()
        t = max(t, 0.15)  # Minimum 15% masking

        bot_start = find_bot_start(x_0[b], bot_token_ids)
        for pos in range(bot_start, seq_len):
            tok = x_0[b, pos].item()
            if tok in special_ids or tok == pad_id:
                continue
            if torch.rand(1).item() < t:
                x_t[b, pos]    = mask_id
                is_mask[b, pos] = True

    return x_t, is_mask


def get_bot_token_ids(vocab):
    """
    Find the token IDs for 'bot' in the BPE vocabulary.
    BPE may represent it as 'bot', 'bot:', or with a space prefix.
    """
    candidates = ['bot', 'bot:', 'Ġbot', ' bot']
    ids = [vocab[t] for t in candidates if t in vocab]
    return ids


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────
#  Validation Loop
# ─────────────────────────────────────────────────────────────

def validate(model, val_loader, vocab_size, mask_id, pad_id, special_ids, bot_token_ids, device):
    """Run validation and return average masked cross-entropy loss."""
    model.eval()
    total_loss = 0
    total_masked = 0
    valid_batches = 0

    with torch.no_grad():
        for x_0 in val_loader:
            x_0 = x_0.to(device)
            pad_mask = (x_0 == pad_id)

            x_t, is_mask = apply_guided_masking(
                x_0, mask_id, pad_id, special_ids, bot_token_ids
            )

            if is_mask.sum() == 0:
                continue

            logits = model(x_t, src_key_padding_mask=pad_mask)

            loss_per_token = F.cross_entropy(
                logits.view(-1, vocab_size),
                x_0.view(-1),
                reduction='none',
                ignore_index=pad_id
            ).view_as(x_0)

            # FIX: Uniform loss weighting (no 1/t scaling)
            masked_loss = (loss_per_token * is_mask.float()).sum() / (is_mask.sum() + 1e-8)

            total_loss    += masked_loss.item()
            total_masked  += is_mask.sum().item()
            valid_batches += 1

    model.train()
    return total_loss / max(valid_batches, 1)


# ─────────────────────────────────────────────────────────────
#  Training Loop
# ─────────────────────────────────────────────────────────────

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    with open("subword_tokenizer.json", "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
        vocab = vocab_data["model"]["vocab"]

    vocab_size  = len(vocab)
    mask_id     = vocab["[MASK]"]
    pad_id      = vocab["[PAD]"]
    special_ids = {vocab["[PAD]"], vocab["[BOS]"], vocab["[EOS]"], vocab["[UNK]"], vocab["[MASK]"]}

    bot_token_ids = get_bot_token_ids(vocab)
    print(f"'bot' token IDs: {bot_token_ids}")
    if not bot_token_ids:
        print("WARNING: 'bot' not found in vocab — falling back to half-masking.")

    # ── Model ──────────────────────────────────────────────
    # Scaled up: d_model 128→256, nhead 4→8, layers 4→6
    model = MaskedDiffusionModel(
        vocab_size=vocab_size,
        d_model=256, nhead=8, num_layers=6,
        max_seq_len=MAX_SEQ_LENGTH
    ).to(device)
    print(f"Parameters: {count_parameters(model):,}")

    # ── Data ───────────────────────────────────────────────
    # Now uses 90/10 train/val split
    train_loader, val_loader, total = create_dataloader(
        "tokenized_data.json", batch_size=32, val_split=0.1
    )

    # ── Optimizer + Schedule ───────────────────────────────
    base_lr       = 5e-4     # Was 1e-3 → 5e-4 (larger model needs smaller LR)
    epochs        = 150      # Was 400 → 150 (more data = fewer epochs)
    warmup_epochs = 10
    warmup_steps  = len(train_loader) * warmup_epochs
    global_step   = 0

    optimizer = AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs, eta_min=1e-5
    )

    best_val_loss  = float('inf')
    best_train_loss = float('inf')
    patience = 0
    max_patience = 20  # Early stopping after 20 epochs without improvement

    print(f"\nTraining for {epochs} epochs (early stop after {max_patience} epochs no improvement)")
    print(f"{'='*70}\n")

    for epoch in range(epochs):
        model.train()
        total_loss    = 0
        total_masked  = 0
        valid_batches = 0

        for x_0 in train_loader:
            x_0 = x_0.to(device)

            # Linear warmup for first warmup_epochs
            if global_step < warmup_steps:
                lr = base_lr * (global_step + 1) / warmup_steps
                for g in optimizer.param_groups:
                    g['lr'] = lr

            pad_mask = (x_0 == pad_id)
            optimizer.zero_grad()

            x_t, is_mask = apply_guided_masking(
                x_0, mask_id, pad_id, special_ids, bot_token_ids
            )

            if is_mask.sum() == 0:
                continue  # Skip batches where nothing was masked

            logits = model(x_t, src_key_padding_mask=pad_mask)

            loss_per_token = F.cross_entropy(
                logits.view(-1, vocab_size),
                x_0.view(-1),
                reduction='none',
                ignore_index=pad_id
            ).view_as(x_0)

            # FIX: Uniform loss weighting — no more problematic 1/t scaling
            # Old: scaled_loss = masked_loss * min(1.0 / (t + 1e-5), 5.0)
            # New: just the masked cross-entropy, no scaling
            masked_loss = (loss_per_token * is_mask.float()).sum() / (is_mask.sum() + 1e-8)

            masked_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss    += masked_loss.item()
            total_masked  += is_mask.sum().item()
            valid_batches += 1
            global_step   += 1

        if epoch >= warmup_epochs:
            scheduler.step()

        avg_train_loss = total_loss / max(valid_batches, 1)

        # ── Validation every 5 epochs ─────────────────────
        if (epoch + 1) % 5 == 0 or epoch == 0:
            val_loss = validate(
                model, val_loader, vocab_size,
                mask_id, pad_id, special_ids, bot_token_ids, device
            )
            lr_now = optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch+1:>4}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Masked: {int(total_masked):>8} | "
                f"LR: {lr_now:.6f}"
            )

            # Save best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "diffusion_model_best.pth")
                patience = 0
                print(f"  → New best val loss! Saved checkpoint.")
            else:
                patience += 1

            # Early stopping
            if patience >= max_patience:
                print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {max_patience} epochs)")
                break

        # Also track best training loss
        if 0 < avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss

    torch.save(model.state_dict(), "diffusion_model.pth")
    print(f"\n{'='*70}")
    print(f"Done! Best train loss: {best_train_loss:.4f} | Best val loss: {best_val_loss:.4f}")
    print("Use diffusion_model_best.pth for inference.")


if __name__ == "__main__":
    train_model()