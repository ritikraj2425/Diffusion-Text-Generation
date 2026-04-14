import json
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def train_and_tokenize(input_file, vocab_file, tokenized_file, max_seq_length=24):

    with open(input_file, 'r', encoding='utf-8') as f:
        sentences = json.load(f)

    print("Training Subword BPE Tokenizer...")
    

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MASK]"]
    
    # We set a small vocab_size since our dataset is small, keeping the model lightweight
    trainer = BpeTrainer(special_tokens=special_tokens, vocab_size=5000)

    tokenizer.train_from_iterator(sentences, trainer=trainer)
    
    tokenizer.save(vocab_file)
    print(f"Tokenizer trained and saved to {vocab_file}")


    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    pad_id = tokenizer.token_to_id("[PAD]")

    print("Encoding dataset...")
    tokenized_dataset = []

    for sent in sentences:
        encoded = tokenizer.encode(sent).ids
        

        sequence = [bos_id] + encoded + [eos_id]
        
        if len(sequence) > max_seq_length:
            sequence = sequence[:max_seq_length]
            sequence[-1] = eos_id # Ensure it still ends with [EOS]
            
        # Pad if too short
        padding_length = max_seq_length - len(sequence)
        sequence.extend([pad_id] * padding_length)
        
        tokenized_dataset.append(sequence)

    with open(tokenized_file, 'w', encoding='utf-8') as f:
        json.dump(tokenized_dataset, f)
        
    print(f"Success! Tokenized dataset saved to {tokenized_file}")


    print("\n--- Subword Tokenization Sanity Check ---")
    sample_text = sentences[0] if sentences else "hello world"
    encoded_sample = tokenizer.encode(sample_text).ids
    
    # Notice how it splits words into subwords!
    print(f"Original Text: '{sample_text}'")
    print(f"Subword Tokens: {tokenizer.encode(sample_text).tokens}") 
    print(f"Encoded IDs:   {encoded_sample}")

if __name__ == "__main__":
    INPUT_DATA_FILE = "processed_data.json"
    OUTPUT_VOCAB_FILE = "subword_tokenizer.json"
    OUTPUT_TOKENIZED_FILE = "tokenized_data.json"
    
    train_and_tokenize(INPUT_DATA_FILE, OUTPUT_VOCAB_FILE, OUTPUT_TOKENIZED_FILE)