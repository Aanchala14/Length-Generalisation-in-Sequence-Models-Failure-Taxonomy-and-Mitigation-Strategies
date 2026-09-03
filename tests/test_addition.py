import random

from src.tasks.addition import AdditionTask


def test_add_digits_with_carry():
    task = AdditionTask(sequence_length=2)

    result = task.add_digits(
        a_digits=[9, 9],
        b_digits=[0, 1],
    )

    assert result == [1, 0, 0]


def test_generated_addition_example():
    random.seed(42)
    task = AdditionTask(sequence_length=2)

    sample = task.generate_example()

    assert sample["task"] == "addition"
    assert sample["length"] == 2
    assert len(sample["input"]) == 5
    assert len(sample["target"]) == 5
    assert sample["input"][2] == task.plus_token

    first_operand = sample["input"][:2]
    second_operand = sample["input"][3:]

    target_digits = [
        token
        for token in sample["target"]
        if token != task.pad_token
    ]

    first_number = int("".join(map(str, first_operand)))
    second_number = int("".join(map(str, second_operand)))
    target_number = int("".join(map(str, target_digits)))

    assert first_operand[0] != 0
    assert second_operand[0] != 0
    assert target_number == first_number + second_number