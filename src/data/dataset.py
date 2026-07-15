import json
import torch
from torch.utils.data import Dataset


class SyntheticDataset(Dataset):
    """
    PyTorch Dataset for synthetic algorithmic tasks.
    """

    def __init__(self, file_path):
        self.samples = []

        with open(file_path, "r") as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        x = torch.tensor(sample["input"], dtype=torch.long)
        y = torch.tensor(sample["target"], dtype=torch.long)

        return x, y