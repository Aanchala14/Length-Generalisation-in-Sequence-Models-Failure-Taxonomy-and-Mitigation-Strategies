import os
import numpy as np
import torch
from torch.utils.data import Dataset


def prepare_uea_dataset(dataset_name, data_root):
    try:
        from aeon.datasets import load_classification
    except ImportError as exc:
        raise ImportError(
            "aeon is required for real-world time-series datasets. "
            "Install it with: pip install aeon"
        ) from exc

    dataset_dir = os.path.join(data_root, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    processed_path = os.path.join(dataset_dir, "processed.npz")

    if os.path.exists(processed_path):
        return processed_path

    X_train, y_train = load_classification(
        dataset_name,
        split="train",
        extract_path=dataset_dir
    )

    X_test, y_test = load_classification(
        dataset_name,
        split="test",
        extract_path=dataset_dir
    )


    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)


    X_train = np.transpose(X_train, (0, 2, 1))
    X_test = np.transpose(X_test, (0, 2, 1))

    classes = sorted(set(y_train.tolist()) | set(y_test.tolist()))
    label_to_id = {label: index for index, label in enumerate(classes)}

    y_train = np.array([label_to_id[label] for label in y_train], dtype=np.int64)
    y_test = np.array([label_to_id[label] for label in y_test], dtype=np.int64)

    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    np.savez(
        processed_path,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        classes=np.array(classes),
    )

    return processed_path


class TimeSeriesPrefixDataset(Dataset):
    def __init__(self, dataset_name, data_root, split, length):
        processed_path = prepare_uea_dataset(dataset_name, data_root)
        data = np.load(processed_path, allow_pickle=True)

        if split == "train":
            self.X = data["X_train"]
            self.y = data["y_train"]
        elif split == "test":
            self.X = data["X_test"]
            self.y = data["y_test"]
        else:
            raise ValueError(f"Unknown split: {split}")

        self.length = length
        self.input_dim = self.X.shape[-1]
        self.num_classes = len(data["classes"])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x = self.X[index]
        y = self.y[index]

        if self.length <= x.shape[0]:
            x = x[:self.length]
        else:
            pad_length = self.length - x.shape[0]
            padding = np.zeros((pad_length, x.shape[1]), dtype=np.float32)
            x = np.concatenate([x, padding], axis=0)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)