import pytest
import torch


MPS_AVAILABLE = (
    hasattr(torch.backends, "mps")
    and torch.backends.mps.is_available()
)


@pytest.mark.skipif(
    not MPS_AVAILABLE,
    reason="Apple MPS is not available on this machine.",
)
def test_mps_tensor_creation():
    device = torch.device("mps")
    tensor = torch.rand(3, 3, device=device)

    assert tensor.shape == (3, 3)
    assert tensor.device.type == "mps"