import json

import torch

from src.data.dataset import SyntheticDataset


def test_synthetic_dataset_loads_jsonl(tmp_path):
    samples = [
        {
            "input": [1, 2, 3],
            "target": [3, 2, 1],
            "task": "reverse",
            "length": 3,
        },
        {
            "input": [4, 5, 6],
            "target": [6, 5, 4],
            "task": "reverse",
            "length": 3,
        },
    ]

    dataset_path = tmp_path / "samples.jsonl"

    with dataset_path.open("w", encoding="utf-8") as file:
        for sample in samples:
            json.dump(sample, file)
            file.write("\n")

    dataset = SyntheticDataset(dataset_path)

    assert len(dataset) == 2

    input_tensor, target_tensor = dataset[0]

    assert isinstance(input_tensor, torch.Tensor)
    assert isinstance(target_tensor, torch.Tensor)
    assert input_tensor.dtype == torch.long
    assert target_tensor.dtype == torch.long
    assert torch.equal(input_tensor, torch.tensor([1, 2, 3]))
    assert torch.equal(target_tensor, torch.tensor([3, 2, 1]))