from torch.utils.data import DataLoader

from .dataset import SyntheticDataset


def get_dataloader(
    f"data/synthetic/{config['task']}/train.jsonl",
    batch_size=config["batch_size"],
    shuffle=True
):
    """
    Create a PyTorch DataLoader for a synthetic dataset.
    """

    dataset = SyntheticDataset(file_path)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return loader