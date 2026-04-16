import json
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# ─────────────────────────────────────────────────────────────
#  Central config — train.py and inference.py import these
# ─────────────────────────────────────────────────────────────
MAX_SEQ_LENGTH = 64    # Was 32 → 64 for longer prompts + responses
VOCAB_SIZE     = 10000  # Was 5000 → 10000 for bigger dataset


def train_and_tokenize(input_file, vocab_file, tokenized_file):

    with open(input_file, 'r', encoding='utf-8') as f:
        sentences = json.load(f)

    print(f"Loaded {len(sentences)} sentences.")
    if not sentences:
        raise ValueError("processed_data.json is empty — run preprocess.py first!")

    print(f"Training Subword BPE Tokenizer (vocab_size={VOCAB_SIZE})...")
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MASK]"]
    trainer = BpeTrainer(special_tokens=special_tokens, vocab_size=VOCAB_SIZE)
    tokenizer.train_from_iterator(sentences, trainer=trainer)
    tokenizer.save(vocab_file)
    print(f"Tokenizer saved to {vocab_file} (vocab: {tokenizer.get_vocab_size()} tokens)")

    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    pad_id = tokenizer.token_to_id("[PAD]")

    print(f"Encoding dataset (MAX_SEQ_LENGTH={MAX_SEQ_LENGTH})...")
    tokenized_dataset = []
    skipped = 0

    for sent in sentences:
        encoded = tokenizer.encode(sent).ids
        sequence = [bos_id] + encoded + [eos_id]

        if len(sequence) > MAX_SEQ_LENGTH:
            skipped += 1
            sequence = sequence[:MAX_SEQ_LENGTH]
            sequence[-1] = eos_id

        padding_length = MAX_SEQ_LENGTH - len(sequence)
        sequence.extend([pad_id] * padding_length)
        tokenized_dataset.append(sequence)

    print(f"Skipped (too long): {skipped} / {len(sentences)} ({100*skipped/len(sentences):.1f}%)")

    with open(tokenized_file, 'w', encoding='utf-8') as f:
        json.dump(tokenized_dataset, f)

    print(f"Tokenized dataset saved to {tokenized_file}")

    # Sanity check
    print("\n--- Sanity Check ---")
    sample = sentences[0]
    enc = tokenizer.encode(sample)
    print(f"Text   : {sample}")
    print(f"Tokens : {enc.tokens}")
    print(f"IDs    : {enc.ids}")
    print(f"Sequence length: {MAX_SEQ_LENGTH}")

    # Content length distribution
    content_lens = [MAX_SEQ_LENGTH - seq.count(pad_id) for seq in tokenized_dataset]
    avg_content = sum(content_lens) / len(content_lens)
    print(f"Avg content tokens: {avg_content:.1f} / {MAX_SEQ_LENGTH}")


if __name__ == "__main__":
    train_and_tokenize("processed_data.json", "subword_tokenizer.json", "tokenized_data.json")