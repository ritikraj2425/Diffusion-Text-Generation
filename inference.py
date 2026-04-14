import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from train import MaskedDiffusionModel



def load_tokenizer_full(vocab_file="subword_tokenizer.json"):
    tokenizer = Tokenizer.from_file(vocab_file)
    vocab = tokenizer.get_vocab()
    id2word = {int(v): k for k, v in vocab.items()}
    return tokenizer, vocab, id2word



def decode_sequence(seq_tensor, id2word):
    special_tokens = ["[PAD]", "[BOS]", "[EOS]"]
    words = []

    for token_id in seq_tensor.tolist():
        word = id2word.get(token_id, "[UNK]")
        if word not in special_tokens:
            words.append(word)

    return " ".join(words).replace(" ##", "").replace("##", "")


def generate_single_response(
    model,
    tokenizer,
    id2word,
    prompt,
    max_response_length=20,
    sampling_steps=15,
    temperature=0.7
):
    model.eval()
    device = next(model.parameters()).device

    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    mask_id = tokenizer.token_to_id("[MASK]")

    # Format input
    formatted_prompt = f"user: {prompt} bot: "
    input_ids = tokenizer.encode(formatted_prompt).ids

    # Create sequence
    sequence = [bos_id] + input_ids + [mask_id] * max_response_length + [eos_id]
    seq_tensor = torch.tensor([sequence], dtype=torch.long, device=device)

    # Identify mask positions
    is_masked = (seq_tensor == mask_id).squeeze(0)
    mask_indices = is_masked.nonzero(as_tuple=True)[0]
    num_masks = len(mask_indices)

    print(f"\nFormatted Input: '{formatted_prompt}'")
    print("=" * 60)

    for step in range(1, sampling_steps + 1):
        with torch.no_grad():
            logits = model(seq_tensor)

        # Temperature scaling
        scaled_logits = logits / max(temperature, 1e-6)
        probs = F.softmax(scaled_logits, dim=-1)

        # Sample tokens
        predicted_ids = torch.multinomial(
            probs.view(-1, probs.size(-1)), 1
        ).view(probs.shape[:-1]).squeeze(0)

        # Confidence calculation
        true_probs = F.softmax(logits, dim=-1).squeeze(0)
        confidences = torch.gather(
            true_probs, 1, predicted_ids.unsqueeze(1)
        ).squeeze(1)

        # Progressive unmasking schedule
        target_unmasked_ratio = step / sampling_steps
        target_unmasked_count = int(num_masks * target_unmasked_ratio)

        current_sequence = seq_tensor.squeeze(0).clone()

        # Fill masked positions
        for idx in mask_indices:
            current_sequence[idx] = predicted_ids[idx]

        # Remasking low-confidence tokens
        if step < sampling_steps:
            gen_confidences = confidences[mask_indices]
            num_to_remask = num_masks - target_unmasked_count

            if num_to_remask > 0:
                _, lowest_conf_rel_indices = torch.topk(
                    gen_confidences, k=num_to_remask, largest=False
                )

                lowest_conf_abs_indices = mask_indices[lowest_conf_rel_indices]

                for idx in lowest_conf_abs_indices:
                    current_sequence[idx] = mask_id

        seq_tensor = current_sequence.unsqueeze(0)

        full_decoded = decode_sequence(seq_tensor.squeeze(0), id2word)

        response_start_idx = 1 + len(input_ids)
        response_end_idx = response_start_idx + max_response_length
        response_tensor = seq_tensor[0][response_start_idx:response_end_idx]

        response_decoded = decode_sequence(response_tensor, id2word)

        print(response_decoded)



    print("\nFinal Generated Output:")
    print(response_decoded)
    print("=" * 60)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on: {device}")

    tokenizer, vocab, id2word = load_tokenizer_full()

    model = MaskedDiffusionModel(
        vocab_size=len(vocab),
        d_model=768,
        nhead=12,
        num_layers=12,
        max_seq_len=128
    ).to(device)

    try:
        model.load_state_dict(torch.load("diffusion_model.pth", map_location=device))
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        exit()

    # Test prompt
    my_prompt = "hi"

    generate_single_response(
        model,
        tokenizer,
        id2word,
        prompt=my_prompt,
        max_response_length=20,
        sampling_steps=15,
        temperature=0.7
    )