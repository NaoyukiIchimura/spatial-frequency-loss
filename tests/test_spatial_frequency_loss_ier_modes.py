import pytest
import torch

from spatial_frequency_loss import SpatialFrequencyLoss


# Command
# -------
# python3 -m pytest -q tests/test_spatial_frequency_loss_ier_modes.py::test_fixed_mode_uses_supplied_weights
#
# Summary
# -------
# test_fixed_mode_uses_supplied_weights verifies fixed-weight forward behavior.
#
# Parameters
# ----------
# None
#
# Returns
# -------
# None
def test_fixed_mode_uses_supplied_weights() -> None:
    expected_weights = torch.tensor([1.0, 2.0, 3.0])
    loss_module = SpatialFrequencyLoss(
        num_image_channels=3,
        base_scale=0.8,
        num_subbands=3,
        ier_update_mode="fixed",
        subband_weights=expected_weights.tolist(),
    )
    generated_image = torch.rand(2, 3, 16, 16, requires_grad=True)
    target_image = torch.rand(2, 3, 16, 16)

    (
        subband_losses,
        weighted_subband_loss,
        _,
        _,
        active_weights,
    ) = loss_module(generated_image, target_image)

    torch.testing.assert_close(
        active_weights,
        expected_weights,
        rtol=0.0,
        atol=0.0,
    )
    expected_weighted_loss = torch.sum(expected_weights * subband_losses)
    torch.testing.assert_close(
        weighted_subband_loss,
        expected_weighted_loss,
        rtol=0.0,
        atol=0.0,
    )
    weighted_subband_loss.backward()
    assert generated_image.grad is not None, (
        "Fixed mode must propagate gradients to the generated image."
    )
    assert torch.isfinite(generated_image.grad).all(), (
        "Fixed mode produced a non-finite generated-image gradient."
    )


# Command
# -------
# python3 -m pytest -q tests/test_spatial_frequency_loss_ier_modes.py::test_calibration_mode_prepares_and_reuses_weights
#
# Summary
# -------
# test_calibration_mode_prepares_and_reuses_weights verifies calibration behavior.
#
# Parameters
# ----------
# None
#
# Returns
# -------
# None
def test_calibration_mode_prepares_and_reuses_weights() -> None:
    loss_module = SpatialFrequencyLoss(
        num_image_channels=3,
        base_scale=0.8,
        num_subbands=3,
        ier_update_mode="calibration",
    )
    generated_image = torch.rand(2, 3, 16, 16)
    target_image = torch.rand(2, 3, 16, 16)

    loss_module.accumulate_calibration_ier(
        generated_image,
        target_image,
    )
    loss_module.finalize_ier_calibration()
    calibrated_weights = loss_module.subband_weights.detach().clone()

    assert bool(loss_module._ier_weights_ready.item()), (
        "Calibration must mark IER weights as ready for optimization."
    )
    assert loss_module._num_ier_observations.item() == 0, (
        "Calibration finalization must clear the observation count, but "
        f"received count={loss_module._num_ier_observations.item()}."
    )
    assert torch.count_nonzero(loss_module._ier_sum).item() == 0, (
        "Calibration finalization must clear the accumulated IER sum."
    )

    loss_module.train()
    loss_module(generated_image, target_image)
    torch.testing.assert_close(
        loss_module.subband_weights,
        calibrated_weights,
        rtol=0.0,
        atol=0.0,
    )
    assert loss_module._num_ier_observations.item() == 0, (
        "Training forward passes must not accumulate new IER observations "
        "after calibration."
    )


# Command
# -------
# python3 -m pytest -q tests/test_spatial_frequency_loss_ier_modes.py::test_fixed_mode_rejects_calibration_calls
#
# Summary
# -------
# test_fixed_mode_rejects_calibration_calls verifies calibration API protection.
#
# Parameters
# ----------
# None
#
# Returns
# -------
# None
def test_fixed_mode_rejects_calibration_calls() -> None:
    loss_module = SpatialFrequencyLoss(
        num_image_channels=3,
        base_scale=0.8,
        num_subbands=3,
        ier_update_mode="fixed",
        subband_weights=[1.0, 2.0, 3.0],
    )
    generated_image = torch.rand(2, 3, 16, 16)
    target_image = torch.rand(2, 3, 16, 16)

    with pytest.raises(
        AssertionError,
        match="requires ier_update_mode='calibration'",
    ):
        loss_module.accumulate_calibration_ier(
            generated_image,
            target_image,
        )

    with pytest.raises(
        AssertionError,
        match="requires ier_update_mode='calibration'",
    ):
        loss_module.finalize_ier_calibration()


# Command
# -------
# python3 -m pytest -q tests/test_spatial_frequency_loss_ier_modes.py::test_invalid_ier_mode_is_rejected
#
# Summary
# -------
# test_invalid_ier_mode_is_rejected verifies IER mode validation.
#
# Parameters
# ----------
# None
#
# Returns
# -------
# None
def test_invalid_ier_mode_is_rejected() -> None:
    with pytest.raises(
        AssertionError,
        match="must be 'calibration' or 'fixed'",
    ):
        SpatialFrequencyLoss(
            num_image_channels=3,
            base_scale=0.8,
            num_subbands=3,
            ier_update_mode="unsupported",
        )

# Command
# -------
# python3 -m pytest -q tests/test_spatial_frequency_loss_ier_modes.py::test_default_ier_reduction_is_sample
#
# Summary
# -------
# test_default_ier_reduction_is_sample verifies the public default reduction.
#
# Parameters
# ----------
# None
#
# Returns
# -------
# None
def test_default_ier_reduction_is_sample() -> None:
    loss_module = SpatialFrequencyLoss(
        num_image_channels=3,
        base_scale=0.8,
        num_subbands=3,
        ier_update_mode="calibration",
    )

    assert loss_module.ier_reduction == "sample", (
        "The default IER reduction must be 'sample' to match the "
        "sample-wise IER definition, but received "
        f"ier_reduction={loss_module.ier_reduction!r}."
    )


# Command
# -------
# python3 -m pytest -q tests/test_spatial_frequency_loss_ier_modes.py::test_sample_ier_reduction_averages_sample_wise_ratios
#
# Summary
# -------
# test_sample_ier_reduction_averages_sample_wise_ratios verifies sample reduction.
#
# Parameters
# ----------
# None
#
# Returns
# -------
# None
def test_sample_ier_reduction_averages_sample_wise_ratios() -> None:
    loss_module = SpatialFrequencyLoss(
        num_image_channels=3,
        base_scale=0.8,
        num_subbands=2,
        ier_update_mode="calibration",
        ier_reduction="sample",
    )
    pixel_loss = torch.tensor(5.0)
    subband_losses = torch.tensor([2.5, 3.0])
    sample_pixel_losses = torch.tensor([2.0, 8.0])
    sample_subband_losses = torch.tensor(
        [
            [1.0, 4.0],
            [4.0, 2.0],
        ]
    )

    ier_sum_contribution, num_observations = (
        loss_module._compute_ier_contribution(
            pixel_loss,
            subband_losses,
            sample_pixel_losses,
            sample_subband_losses,
        )
    )
    actual_ier_weights = ier_sum_contribution / num_observations
    expected_ier_weights = torch.tensor(
        [2.0, 2.25],
        dtype=torch.float64,
    )

    assert num_observations == 2, (
        "Sample reduction must count one IER observation per sample, but "
        f"received num_observations={num_observations}."
    )
    torch.testing.assert_close(
        actual_ier_weights,
        expected_ier_weights,
        rtol=0.0,
        atol=0.0,
    )


# Command
# -------
# python3 -m pytest -q tests/test_spatial_frequency_loss_ier_modes.py::test_batch_ier_reduction_uses_ratio_of_batch_mean_losses
#
# Summary
# -------
# test_batch_ier_reduction_uses_ratio_of_batch_mean_losses verifies batch reduction.
#
# Parameters
# ----------
# None
#
# Returns
# -------
# None
def test_batch_ier_reduction_uses_ratio_of_batch_mean_losses() -> None:
    loss_module = SpatialFrequencyLoss(
        num_image_channels=3,
        base_scale=0.8,
        num_subbands=2,
        ier_update_mode="calibration",
        ier_reduction="batch",
    )
    pixel_loss = torch.tensor(5.0)
    subband_losses = torch.tensor([2.5, 3.0])
    sample_pixel_losses = torch.tensor([2.0, 8.0])
    sample_subband_losses = torch.tensor(
        [
            [1.0, 4.0],
            [4.0, 2.0],
        ]
    )

    ier_sum_contribution, num_observations = (
        loss_module._compute_ier_contribution(
            pixel_loss,
            subband_losses,
            sample_pixel_losses,
            sample_subband_losses,
        )
    )
    expected_ier_weights = torch.tensor(
        [2.0, 5.0 / 3.0],
        dtype=torch.float32,
    ).to(torch.float64)

    assert num_observations == 1, (
        "Batch reduction must count one IER observation per batch, but "
        f"received num_observations={num_observations}."
    )
    torch.testing.assert_close(
        ier_sum_contribution,
        expected_ier_weights,
        rtol=0.0,
        atol=0.0,
    )
