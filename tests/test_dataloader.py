import json

import torch

from src.data.dataloader import get_dataloader


def test_dataloader_returns_expected_batches(tmp_path):
    samples = [
        {
            "input": [index, index + 1, index + 2],
            "target": [index + 2, index + 1, index],
            "task": "reverse",
            "length": 3,
        }
        for index in range(5)
    ]

    dataset_path = tmp_path / "samples.jsonl"

    with dataset_path.open("w", encoding="utf-8") as file:
        for sample in samples:
            json.dump(sample, file)
            file.write("\n")

    loader = get_dataloader(
        file_path=dataset_path,
        batch_size=2,
        shuffle=False,
    )

    assert len(loader) == 3

    input_batch, target_batch = next(iter(loader))

    assert isinstance(input_batch, torch.Tensor)
    assert isinstance(target_batch, torch.Tensor)
    assert input_batch.shape == (2, 3)
    assert target_batch.shape == (2, 3)

    assert torch.equal(
        input_batch[0],
        torch.tensor([0, 1, 2]),
    )
    assert torch.equal(
        target_batch[0],
        torch.tensor([2, 1, 0]),
    )