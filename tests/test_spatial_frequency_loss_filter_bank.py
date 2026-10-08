import pytest
import torch
from spatial_frequency_loss import SpatialFrequencyLoss

# Command
# -------
# python3 -m pytest -q tests/test_spatial_frequency_loss_filter_bank.py
@pytest.mark.parametrize("filter_backend", ["conv2d", "separable"])
def test_filter_bank_matches_forward_subbands(filter_backend: str) -> None:
    torch.manual_seed(10)
    loss_module = SpatialFrequencyLoss(
        num_image_channels=3,
        base_scale=0.6,
        num_subbands=4,
        scale_radix=2.0,
        filter_size_factor=4.0,
        filter_backend=filter_backend,
        ier_update_mode="fixed",
        ier_reduction="sample",
        eps=1.0e-12,
        subband_weights=[1.0, 2.0, 3.0, 4.0],
        normalize_gaussian=False,
    )
    generated_image = torch.rand(2, 3, 32, 32, requires_grad=True)
    target_image = torch.rand(2, 3, 32, 32)
    filter_bank_subbands = loss_module.filter_bank(generated_image)
    (
        _,
        _,
        forward_generated_subbands,
        _,
        _,
    ) = loss_module(generated_image, target_image)
    assert len(filter_bank_subbands) == 4, (
        "filter_bank() returned an unexpected number of subbands: "
        f"actual={len(filter_bank_subbands)}, expected=4, "
        f"filter_backend={filter_backend!r}."
    )
    assert len(forward_generated_subbands) == 4, (
        "forward() returned an unexpected number of generated subbands: "
        f"actual={len(forward_generated_subbands)}, expected=4, "
        f"filter_backend={filter_backend!r}."
    )
    for subband_index, (
        filter_bank_subband,
        forward_generated_subband,
    ) in enumerate(zip(filter_bank_subbands, forward_generated_subbands)):
        assert filter_bank_subband.shape == generated_image.shape, (
            "filter_bank() returned a subband with an unexpected shape: "
            f"subband_index={subband_index}, "
            f"actual_shape={tuple(filter_bank_subband.shape)}, "
            f"expected_shape={tuple(generated_image.shape)}, "
            f"filter_backend={filter_backend!r}."
        )
        torch.testing.assert_close(
            filter_bank_subband,
            forward_generated_subband,
            rtol=0.0,
            atol=0.0,
        )
    public_filter_bank_loss = sum(
        subband.square().mean() for subband in filter_bank_subbands
    )
    public_filter_bank_loss.backward()
    assert generated_image.grad is not None, (
        "No generated-image gradient was produced through the public "
        f"filter_bank() path for filter_backend={filter_backend!r}."
    )
    assert torch.isfinite(generated_image.grad).all(), (
        "A non-finite generated-image gradient was produced through the "
        f"public filter_bank() path for filter_backend={filter_backend!r}."
    )
