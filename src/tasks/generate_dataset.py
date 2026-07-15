from pathlib import Path

from .copy import CopyTask
from .addition import AdditionTask


# --------------------------------------------------
# Task Configuration
# --------------------------------------------------

TASK = "addition"      # Change to "copy" or "addition"

TRAIN_SAMPLES = 10000
VALIDATION_SAMPLES = 1000
TEST_SAMPLES = 1000

VOCAB_SIZE = 100

if TASK == "copy":
    TRAIN_LENGTH = 20
    TEST_LENGTHS = [20, 40, 60, 80, 100, 160]

elif TASK == "addition":
    TRAIN_LENGTH = 2          # Number of digits
    TEST_LENGTHS = [2, 3, 4, 5, 6]

else:
    raise ValueError(f"Unknown task: {TASK}")


# --------------------------------------------------
# Task Factory
# --------------------------------------------------

def create_task(length):
    """
    Create the appropriate task object.
    """

    if TASK == "copy":
        return CopyTask(
            vocab_size=VOCAB_SIZE,
            sequence_length=length
        )

    elif TASK == "addition":
        return AdditionTask(
            sequence_length=length
        )


# --------------------------------------------------
# Dataset Generation
# --------------------------------------------------

def generate_split(task, samples, output_file):
    dataset = task.generate_dataset(samples)
    task.save_jsonl(dataset, output_file)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    output_dir = Path(f"data/synthetic/{TASK}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------
    # Training
    # -----------------------

    train_task = create_task(TRAIN_LENGTH)

    generate_split(
        train_task,
        TRAIN_SAMPLES,
        output_dir / "train.jsonl"
    )

    # -----------------------
    # Validation
    # -----------------------

    validation_task = create_task(TRAIN_LENGTH)

    generate_split(
        validation_task,
        VALIDATION_SAMPLES,
        output_dir / "validation.jsonl"
    )

    # -----------------------
    # Testing
    # -----------------------

    for length in TEST_LENGTHS:

        test_task = create_task(length)

        generate_split(
            test_task,
            TEST_SAMPLES,
            output_dir / f"test_{length}.jsonl"
        )

    print(f"\nFinished generating {TASK} datasets!")


if __name__ == "__main__":
    main()