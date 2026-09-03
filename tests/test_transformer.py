import pytest
import torch

from src.models.transformer import TransformerModel


@pytest.mark.parametrize(
    "positional_encoding",
    [
        "learned",
        "sinusoidal",
        "none",
        "alibi",
        "rope",
    ],
)
def test_transformer_output_shape(positional_encoding):
    model = TransformerModel(
        vocab_size=12,
        embedding_dim=16,
        num_heads=4,
        num_layers=2,
        feedforward_dim=32,
        dropout=0.0,
        max_length=16,
        positional_encoding=positional_encoding,
    )
    model.eval()

    inputs = torch.randint(
        low=0,
        high=12,
        size=(2, 8),
    )

    with torch.no_grad():
        outputs = model(inputs)

    assert outputs.shape == (2, 8, 12)
    assert torch.isfinite(outputs).all()


def test_transformer_returns_attention_maps():
    model = TransformerModel(
        vocab_size=12,
        embedding_dim=16,
        num_heads=4,
        num_layers=2,
        feedforward_dim=32,
        dropout=0.0,
        max_length=16,
        positional_encoding="learned",
    )
    model.eval()

    inputs = torch.randint(
        low=0,
        high=12,
        size=(2, 8),
    )

    with torch.no_grad():
        outputs, attention_maps = model(
            inputs,
            return_attention=True,
        )

    assert outputs.shape == (2, 8, 12)
    assert len(attention_maps) == 2

    for attention_map in attention_maps:
        assert attention_map.shape == (2, 4, 8, 8)
        assert torch.isfinite(attention_map).all()


def test_unknown_positional_encoding_raises_error():
    with pytest.raises(ValueError, match="Unknown positional encoding"):
        TransformerModel(
            positional_encoding="unsupported",
        )