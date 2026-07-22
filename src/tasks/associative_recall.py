import random

from .base_task import BaseTask


class AssociativeRecallTask(BaseTask):
    """
    Associative recall task.

    Input:
        key/value pairs + QUERY token + query key

    Target:
        PAD tokens everywhere except the final position,
        where the model must output the value associated with the query key.
    """

    def __init__(
        self,
        vocab_size,
        sequence_length,
        query_token=None,
        pad_token=None
    ):
        self.query_token = (
            vocab_size - 2 if query_token is None
            else query_token
        )

        self.pad_token = (
            vocab_size - 1 if pad_token is None
            else pad_token
        )

        super().__init__(
            vocab_size=vocab_size,
            sequence_length=sequence_length
        )

    def generate_example(self):
        available_keys = list(range(self.vocab_size - 2))

        keys = random.sample(
            available_keys,
            self.sequence_length
        )

        values = [
            random.randint(0, self.vocab_size - 3)
            for _ in range(self.sequence_length)
        ]

        query_index = random.randint(
            0,
            self.sequence_length - 1
        )

        query_key = keys[query_index]
        correct_value = values[query_index]

        input_sequence = []

        for key, value in zip(keys, values):
            input_sequence.extend([key, value])

        input_sequence.extend([
            self.query_token,
            query_key
        ])

        target_sequence = (
            [self.pad_token] * (len(input_sequence) - 1)
            + [correct_value]
        )

        return self.create_sample(
            input_sequence=input_sequence,
            target_sequence=target_sequence
        )