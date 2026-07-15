from abc import ABC, abstractmethod
import json
from pathlib import Path


class BaseTask(ABC):
    """
    Base class for all synthetic algorithmic tasks.
    """

    def __init__(self,
                 vocab_size,
                 sequence_length):

        self.vocab_size = vocab_size
        self.sequence_length = sequence_length


    def create_sample(
        self, input_sequence, target_sequence):
        """
        Create a standardized dataset sample.
        """

        return {
            "input": input_sequence,
            "target": target_sequence,
            "task": self.__class__.__name__.replace("Task", "").lower(),
            "length": self.sequence_length
            }

    def generate_dataset(self, n_samples):

        dataset = []

        for _ in range(n_samples):
            dataset.append(
                self.generate_example()
            )

        return dataset

    def save_jsonl(self,
                   dataset,
                   output_path):

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(output_path, "w") as f:

            for sample in dataset:

                json.dump(sample, f)

                f.write("\n")