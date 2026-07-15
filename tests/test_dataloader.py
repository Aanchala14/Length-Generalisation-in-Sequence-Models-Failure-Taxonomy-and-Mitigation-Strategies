from src.data.dataloader import get_dataloader

loader = get_dataloader(
    "data/synthetic/copy/train.jsonl",
    batch_size=4
)

print("Number of batches:", len(loader))

for x, y in loader:

    print("Input batch shape :", x.shape)
    print("Target batch shape:", y.shape)

    print(x)
    print(y)

    break