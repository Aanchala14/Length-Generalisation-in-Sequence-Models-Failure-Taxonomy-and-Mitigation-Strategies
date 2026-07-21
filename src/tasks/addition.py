import random

from .base_task import BaseTask


class AdditionTask(BaseTask):
    """
    Long digit-wise addition task.

    Input:
        digits_a + PLUS + digits_b

    Target:
        PAD tokens followed by the sum digits, aligned to input length.
    """

    PLUS_TOKEN = 10
    PAD_TOKEN = 11

    def __init__(
        self,
        sequence_length,
        plus_token=PLUS_TOKEN,
        pad_token=PAD_TOKEN
    ):
        self.plus_token = plus_token
        self.pad_token = pad_token

        super().__init__(
            vocab_size=12,
            sequence_length=sequence_length
        )

    def generate_digits(self):
        digits = [
            random.randint(0, 9)
            for _ in range(self.sequence_length)
        ]

        # Avoid leading zero so the logical length is always fixed.
        digits[0] = random.randint(1, 9)

        return digits

    def add_digits(self, a_digits, b_digits):
        carry = 0
        result = []

        for a, b in zip(
            reversed(a_digits),
            reversed(b_digits)
        ):
            total = a + b + carry
            result.append(total % 10)
            carry = total // 10

        if carry:
            result.append(carry)

        return list(reversed(result))

    def encode_input(self, a_digits, b_digits):
        return (
            a_digits
            + [self.plus_token]
            + b_digits
        )

    def encode_target(self, sum_digits):
        input_length = (2 * self.sequence_length) + 1

        padding_length = input_length - len(sum_digits)

        return (
            [self.pad_token] * padding_length
            + sum_digits
        )

    def generate_example(self):
        a_digits = self.generate_digits()
        b_digits = self.generate_digits()

        sum_digits = self.add_digits(
            a_digits,
            b_digits
        )

        input_sequence = self.encode_input(
            a_digits,
            b_digits
        )

        target_sequence = self.encode_target(sum_digits)

        return self.create_sample(
            input_sequence=input_sequence,
            target_sequence=target_sequence
        )