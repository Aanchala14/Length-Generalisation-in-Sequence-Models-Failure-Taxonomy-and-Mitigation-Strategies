import random

from .base_task import BaseTask


class ReverseTask(BaseTask):
    """
    Reverse task.

    Input:
        [x1, x2, ..., xL]

    Target:
        [xL, ..., x2, x1]
    """

    def __init__(
        self,
        vocab_size,
        sequence_length
    ):
        super().__init__(
            vocab_size=vocab_size,
            sequence_length=sequence_length
        )

    def generate_example(self):
        sequence = [
            random.randint(
                0,
                self.vocab_size - 1
            )
            for _ in range(self.sequence_length)
        ]

        target_sequence = list(reversed(sequence))

        return self.create_sample(
            input_sequence=sequence,
            target_sequence=target_sequence
        )