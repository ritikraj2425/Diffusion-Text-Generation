import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class ChatDataset(Dataset):
    def __init__(self, data):
        """
        Accepts a list of tokenized sequences (list of lists of ints).
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns a single sequence as a LongTensor for the Embedding layer.
        """
        return torch.tensor(self.data[idx], dtype=torch.long)


def create_dataloader(tokenized_file, batch_size=32, shuffle=True, val_split=0.1):
    """
    Creates train and (optionally) validation DataLoaders.

    Args:
        tokenized_file: Path to JSON file with tokenized sequences
        batch_size:     Batch size for DataLoader
        shuffle:        Whether to shuffle training data
        val_split:      Fraction of data to use for validation (0 to disable)

    Returns:
        If val_split > 0: (train_loader, val_loader, total_samples)
        If val_split == 0: (train_loader, total_samples)    [backward compatible]
    """
    print(f"Loading data from {tokenized_file}...")
    with open(tokenized_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    total = len(raw_data)
    full_dataset = ChatDataset(raw_data)

    if val_split > 0:
        val_size = int(total * val_split)
        train_size = total - val_size

        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

        print(f"Train: {train_size} samples ({len(train_loader)} batches)")
        print(f"Val:   {val_size} samples ({len(val_loader)} batches)")

        return train_loader, val_loader, total
    else:
        train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=shuffle)
        return train_loader, total


if __name__ == "__main__":
    INPUT_TOKENIZED_FILE = "tokenized_data.json"
    BATCH_SIZE = 32

    print("Initializing PyTorch Dataset and DataLoader...")
    train_loader, val_loader, total_samples = create_dataloader(
        INPUT_TOKENIZED_FILE, batch_size=BATCH_SIZE, val_split=0.1
    )

    print(f"\nTotal samples: {total_samples}")

    # Sanity check
    for batch in train_loader:
        print(f"\n--- DataLoader Sanity Check ---")
        print(f"Batch Shape: {batch.shape}")
        print(f"Data Type:   {batch.dtype}")
        print(f"First seq:   {batch[0][:20]}...")
        break