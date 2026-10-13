import torch

from spatial_frequency_loss import SpatialFrequencyLoss


# Command
# -------
# python3 -m pytest -q tests/test_spatial_frequency_loss_default_backend_20261013.py
#
# Summary
# -------
# test_default_filter_backend_is_separable verifies the public default backend.
#
# Parameters
# ----------
# None
#
# Returns
# -------
# None
def test_default_filter_backend_is_separable() -> None:
    loss_module = SpatialFrequencyLoss(
        num_image_channels=3,
        base_scale=0.6,
        num_subbands=4,
        ier_update_mode="fixed",
        subband_weights=[1.0, 1.0, 1.0, 1.0],
    )
    assert loss_module.filter_backend == "separable", (
        "The public default filter backend must be separable, but received "
        f"filter_backend={loss_module.filter_backend!r}."
    )
    generated_image = torch.rand(1, 3, 32, 32, requires_grad=True)
    target_image = torch.rand(1, 3, 32, 32)
    _, weighted_loss, _, _, _ = loss_module(
        generated_image,
        target_image,
    )
    weighted_loss.backward()
    assert generated_image.grad is not None, (
        "The default separable backend did not propagate image gradients."
    )
    assert torch.isfinite(generated_image.grad).all(), (
        "The default separable backend produced non-finite image gradients."
    )
