import torch

from src.models.positional_encoding import (
    LearnedPositionalEncoding,
    NoPositionalEncoding,
    SinusoidalPositionalEncoding,
)


def test_learned_positional_encoding_shape():
    encoding = LearnedPositionalEncoding(
        max_length=20,
        embedding_dim=128,
    )

    inputs = torch.zeros(4, 20, 128)
    outputs = encoding(inputs)

    assert outputs.shape == inputs.shape
    assert not torch.equal(outputs, inputs)
    assert not torch.equal(outputs[:, 0, :], outputs[:, 1, :])


def test_sinusoidal_positional_encoding_shape_and_values():
    encoding = SinusoidalPositionalEncoding(
        max_length=20,
        embedding_dim=128,
    )

    inputs = torch.zeros(4, 20, 128)
    outputs = encoding(inputs)

    assert outputs.shape == inputs.shape

    expected_even_dimensions = torch.zeros(64)
    expected_odd_dimensions = torch.ones(64)

    assert torch.allclose(
        outputs[0, 0, 0::2],
        expected_even_dimensions,
    )
    assert torch.allclose(
        outputs[0, 0, 1::2],
        expected_odd_dimensions,
    )


def test_no_positional_encoding_preserves_input():
    encoding = NoPositionalEncoding()

    inputs = torch.randn(4, 20, 128)
    outputs = encoding(inputs)

    assert torch.equal(outputs, inputs)