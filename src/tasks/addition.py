import random

from .base_task import BaseTask


class AdditionTask(BaseTask):

    PLUS_TOKEN = 10

    PAD_TOKEN = 11

    def __init__(
        self,
        sequence_length
    ):

        super().__init__(
            vocab_size=12,
            sequence_length=sequence_length
        )

    def number_to_digits(
        self,
        number
    ):

        return [
            int(digit)
            for digit in str(number)
        ]
    def generate_numbers(self):

        minimum = 10 ** (self.sequence_length - 1)

        maximum = (10 ** self.sequence_length) - 1

        a = random.randint(
            minimum,
            maximum
        )

        b = random.randint(
            minimum,
            maximum
        )

        return a, b

    def encode_input(
        self,
        a,
        b
    ):

        return (

            self.number_to_digits(a)

            +

            [self.PLUS_TOKEN]

            +

            self.number_to_digits(b)

        )

    def encode_target(
            self,
            answer):
            """
            Encode the answer and pad it so that the
            target length matches the input length.
            """

            digits = self.number_to_digits(answer)

            input_length = (
                 2 * self.sequence_length
            ) + 1

            padding = [self.PAD_TOKEN] * (
                 input_length - len(digits)
            )

            return padding + digits
            

    def generate_example(self):
        """
        Generate one addition example.
        """

        # Generate two random numbers
        a, b = self.generate_numbers()

        # Compute the correct answer
        answer = a + b

        # Encode input and target
        input_sequence = self.encode_input(a, b)
        target_sequence = self.encode_target(answer)

        return self.create_sample(
            input_sequence=input_sequence,
            target_sequence=target_sequence
        )