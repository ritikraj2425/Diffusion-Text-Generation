import json
import torch
from torch.utils.data import Dataset, DataLoader

class ChatDataset(Dataset):
    def __init__(self, tokenized_file):
        """
        We load our massive JSON list here.
        """
        print(f"Loading data from {tokenized_file}...")
        with open(tokenized_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        """
        Fetches a single sentence by its index and converts it to a PyTorch Tensor.
        We use 'torch.long' because these are integer IDs meant for an Embedding layer.
        """
        return torch.tensor(self.data[idx], dtype=torch.long)

def create_dataloader(tokenized_file, batch_size=32, shuffle=True):
    """
    Wraps the Dataset in a DataLoader to handle batching and shuffling automatically.
    """
    dataset = ChatDataset(tokenized_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader, len(dataset)

if __name__ == "__main__":
    INPUT_TOKENIZED_FILE = "tokenized_data.json"
    BATCH_SIZE = 16
    
    print("Initializing PyTorch Dataset and DataLoader...")
    dataloader, total_samples = create_dataloader(INPUT_TOKENIZED_FILE, batch_size=BATCH_SIZE)
    
    print(f"Total individual sentences: {total_samples}")
    print(f"Total batches per epoch: {len(dataloader)} (Total sentences / Batch Size)")
    
    # Let's grab exactly one batch to see what it looks like before sending it to a model
    for batch in dataloader:
        print("\n--- DataLoader Sanity Check ---")
        # The shape should be [16, 24] meaning [Batch_Size, Max_Sequence_Length]
        print(f"Batch Shape: {batch.shape}")
        print(f"Data Type: {batch.dtype}")
        print(f"First sequence in this batch:\n{batch[0]}")
        break