from .base_task import BaseTask
import random


class CopyTask(BaseTask):

    def generate_example(self):

        sequence = [
            random.randint(
                0,
                self.vocab_size - 1
            )
            for _ in range(
                self.sequence_length
            )
        ]

        return self.create_sample(
            input_sequence=sequence,
            target_sequence=sequence
            )