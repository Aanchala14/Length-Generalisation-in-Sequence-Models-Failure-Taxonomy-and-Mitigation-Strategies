import random

from .base_task import BaseTask


class CopyTask(BaseTask):
    """
    Delayed copy task.

    The model sees a sequence followed by a separator and blank tokens.
    It must reproduce the original sequence after the separator.
    """

    def __init__(
        self,
        vocab_size,
        sequence_length,
        separator_token=None,
        pad_token=None
    ):
        self.separator_token = (
            vocab_size if separator_token is None
            else separator_token
        )

        self.pad_token = (
            vocab_size + 1 if pad_token is None
            else pad_token
        )

        super().__init__(
            vocab_size=vocab_size + 2,
            sequence_length=sequence_length
        )

    def generate_example(self):
        sequence = [
            random.randint(
                0,
                self.vocab_size - 3
            )
            for _ in range(self.sequence_length)
        ]

        input_sequence = (
            sequence
            + [self.separator_token]
            + [self.pad_token] * self.sequence_length
        )

        target_sequence = (
            [self.pad_token] * self.sequence_length
            + [self.pad_token]
            + sequence
        )

        return self.create_sample(
            input_sequence=input_sequence,
            target_sequence=target_sequence
        )