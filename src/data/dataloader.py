from torch.utils.data import DataLoader

from .dataset import SyntheticDataset


def get_dataloader(file_path, batch_size, shuffle=True):
    """
    Create a PyTorch DataLoader for a synthetic JSONL dataset.
    """

    dataset = SyntheticDataset(file_path)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )