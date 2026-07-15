from src.data.dataset import SyntheticDataset

dataset = SyntheticDataset("data/synthetic/copy/train.jsonl")

print("Dataset size:", len(dataset))

x, y = dataset[0]

print(type(x))
print(type(y))

print(x.shape)
print(y.shape)

print(x)
print(y)