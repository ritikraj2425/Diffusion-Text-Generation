import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from train import MaskedDiffusionModel
from tokenizer import MAX_SEQ_LENGTH
import os


def load_tokenizer_full(vocab_file="subword_tokenizer.json"):
    tokenizer = Tokenizer.from_file(vocab_file)
    vocab     = tokenizer.get_vocab()
    id2word   = {int(v): k for k, v in vocab.items()}
    return tokenizer, vocab, id2word


def decode_response(token_ids, tokenizer):
    """
    FIX: Use the tokenizer's built-in decode() instead of manual subword joining.
    The old code tried to clean up '##' markers (WordPiece), but we use BPE.
    The tokenizer.decode() handles BPE subwords correctly.
    """
    # Filter out special tokens
    special_ids = {
        tokenizer.token_to_id("[PAD]"),
        tokenizer.token_to_id("[BOS]"),
        tokenizer.token_to_id("[EOS]"),
        tokenizer.token_to_id("[MASK]"),
        tokenizer.token_to_id("[UNK]"),
    }

    filtered_ids = [tid for tid in token_ids if tid not in special_ids]

    if not filtered_ids:
        return ""

    # Let the tokenizer handle subword reassembly
    text = tokenizer.decode(filtered_ids)

    # Light cleanup
    text = text.strip()
    # Fix spacing around punctuation
    for p in [".", ",", "?", "!", "'", ":"]:
        text = text.replace(f" {p}", p)

    return text


def generate_response(
    model, tokenizer, id2word,
    prompt,
    max_response_length=24,   # Was 12 → 24 (longer sequences now)
    sampling_steps=40,        # Was 25 → 40 (more refinement steps)
    temperature=0.5,          # Slightly higher than 0.4 for more variety
    top_k=15                  # Was 10 → 15 (slightly wider sampling)
):
    model.eval()
    device = next(model.parameters()).device

    bos_id  = tokenizer.token_to_id("[BOS]")
    eos_id  = tokenizer.token_to_id("[EOS]")
    mask_id = tokenizer.token_to_id("[MASK]")
    pad_id  = tokenizer.token_to_id("[PAD]")

    formatted = f"user: {prompt.lower().strip()} bot:"
    input_ids = tokenizer.encode(formatted).ids

    # Clamp response length to available space
    max_resp = min(max_response_length, MAX_SEQ_LENGTH - len(input_ids) - 2)
    if max_resp <= 0:
        print("Prompt too long.")
        return ""

    sequence = [bos_id] + input_ids + [mask_id] * max_resp + [eos_id]
    sequence += [pad_id] * (MAX_SEQ_LENGTH - len(sequence))
    seq_tensor = torch.tensor([sequence], dtype=torch.long, device=device)

    response_start = 1 + len(input_ids)
    response_end   = response_start + max_resp
    mask_indices   = list(range(response_start, response_end))
    num_masks      = len(mask_indices)

    for step in range(1, sampling_steps + 1):
        with torch.no_grad():
            logits = model(seq_tensor)

        # Top-k filtering — zero out all but top-k logits
        response_logits = logits[0, mask_indices]  # [num_masks, vocab_size]
        if top_k > 0:
            top_k_vals, _ = torch.topk(response_logits, top_k, dim=-1)
            min_top_k     = top_k_vals[:, -1].unsqueeze(-1)
            response_logits = response_logits.masked_fill(response_logits < min_top_k, float('-inf'))

        # Temperature scaling and sampling
        scaled = response_logits / max(temperature, 1e-6)
        probs  = F.softmax(scaled, dim=-1)

        # At final step use greedy (argmax) for cleaner output
        if step == sampling_steps:
            predicted = torch.argmax(probs, dim=-1)
        else:
            predicted = torch.multinomial(probs, 1).squeeze(-1)

        # Confidence for remasking schedule
        true_probs  = F.softmax(response_logits, dim=-1)
        confidences = true_probs[torch.arange(num_masks), predicted]

        current = seq_tensor.squeeze(0).clone()
        for i, idx in enumerate(mask_indices):
            current[idx] = predicted[i]

        # Progressive remasking: reveal high-confidence tokens first
        if step < sampling_steps:
            target_revealed = int(num_masks * step / sampling_steps)
            num_remask      = num_masks - target_revealed
            if num_remask > 0:
                _, low_idx = torch.topk(confidences, k=num_remask, largest=False)
                for li in low_idx:
                    current[mask_indices[li]] = mask_id

        seq_tensor = current.unsqueeze(0)

    # FIX: Use tokenizer.decode() for proper BPE subword handling
    response_ids = seq_tensor[0][response_start:response_end].tolist()
    return decode_response(response_ids, tokenizer)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference on: {device}\n")

    tokenizer, vocab, id2word = load_tokenizer_full()

    # Must match train.py architecture
    model = MaskedDiffusionModel(
        vocab_size=len(vocab),
        d_model=256, nhead=8, num_layers=6,
        max_seq_len=MAX_SEQ_LENGTH
    ).to(device)

    ckpt = "diffusion_model_best.pth" if os.path.exists("diffusion_model_best.pth") else "diffusion_model.pth"
    try:
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"Loaded: {ckpt}\n")
    except Exception as e:
        print(f"Error: {e}")
        exit()

    test_prompts = [
        "hi",
        "how are you",
        "what is your name",
        "tell me a joke",
        "what do you do for fun",
        "i had a bad day",
    ]

    for prompt in test_prompts:
        response = generate_response(
            model, tokenizer, id2word,
            prompt=prompt,
            max_response_length=24,
            sampling_steps=40,
            temperature=0.5,
            top_k=15
        )
        print(f"User : {prompt}")
        print(f"Bot  : {response}")
        print("-" * 40)