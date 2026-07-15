'''Load trained model
        ↓
Load test dataset
        ↓
Run inference
        ↓
Compare predictions with targets
        ↓
Compute accuracy
        ↓
Repeat for every sequence length
        ↓
Save results'''

import torch

from src.models.transformer import TransformerModel
from src.data.dataloader import get_dataloader

PAD_TOKEN = None

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using device: {device}")


# -----------------------------
# Load model
# -----------------------------

model = TransformerModel()

model.load_state_dict(
    torch.load(
        "outputs/checkpoints/copy_baseline.pt",
        map_location=device
    )
)

model.to(device)

model.eval()

print("Model loaded successfully.")

# -----------------------------
# Evaluate All Sequence Lengths
# -----------------------------

TEST_LENGTHS = [20, 40, 60, 80, 100, 160]

results = []

for length in TEST_LENGTHS:

    print(f"\nEvaluating length {length}")

    test_loader = get_dataloader(
        f"data/synthetic/copy/test_{length}.jsonl",
        batch_size=32,
        shuffle=False
    )

    correct = 0
    total = 0

    with torch.no_grad():

        for x, y in test_loader:

            x = x.to(device)
            y = y.to(device)

            outputs = model(x)

            predictions = outputs.argmax(dim=-1)

            correct += (predictions == y).sum().item()

            total += y.numel()

    accuracy = 100 * correct / total

    results.append((length, accuracy))

    print(f"Accuracy: {accuracy:.2f}%")

print("\n==============================")
print("Length Generalisation Results")
print("==============================")

for length, accuracy in results:

    print(
        f"Length {length:>3} : "
        f"{accuracy:.2f}%"
    )

    import csv
import os

os.makedirs("outputs/results", exist_ok=True)

with open("outputs/results/copy_results.csv", "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "Task",
        "Train Length",
        "Test Length",
        "Token Accuracy"
    ])

    for length, accuracy in results:

        writer.writerow([
            "Copy",
            20,
            length,
            accuracy
        ])

print("\nResults saved!")