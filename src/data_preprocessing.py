import torch
from torch.utils.data import Dataset
from collections import Counter

class TextDataset(Dataset):
    def __init__(self, text, seq_length=50):
        self.chars = sorted(list(set(text)))
        self.char2idx = {ch: idx for idx, ch in enumerate(self.chars)}
        self.idx2char = {idx: ch for ch, idx in self.char2idx.items()}
        self.vocab_size = len(self.chars)
        self.seq_length = seq_length
        self.data = [self.char2idx[ch] for ch in text]

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx+self.seq_length], dtype=torch.long)
        y = torch.tensor(self.data[idx+1:idx+self.seq_length+1], dtype=torch.long)
        return x, y

def load_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().lower()

def get_data_loaders(path, seq_length=50, batch_size=64, split_ratio=0.9):
    text = load_text(path)
    split_idx = int(len(text) * split_ratio)
    train_text, val_text = text[:split_idx], text[split_idx:]

    train_dataset = TextDataset(train_text, seq_length)
    val_dataset = TextDataset(val_text, seq_length)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, train_dataset.vocab_size, train_dataset.char2idx, train_dataset.idx2char
