#
# test_spatial_frequency_loss_filter_backends_20261009.py
#
# Numerical-equivalence tests for the Conv2d and separable DoG backends.
#
import pytest
import torch

from spatial_frequency_loss import SpatialFrequencyLoss


NUM_IMAGE_CHANNELS = 3
BASE_SCALE = 0.6
NUM_SUBBANDS = 7
SCALE_RADIX = 2.0
FILTER_SIZE_FACTOR = 8.0
FIXED_WEIGHTS = [464.5, 277.2, 140.2, 69.70, 40.53, 26.70, 1.692]
ABSOLUTE_TOLERANCE = 2.0e-6
RELATIVE_TOLERANCE = 2.0e-5
SUBBAND_ABSOLUTE_TOLERANCE = 2.0e-5
IER_ABSOLUTE_TOLERANCE = 2.0e-4
IER_RELATIVE_TOLERANCE = 5.0e-4


# Summary
# -------
# _make_images creates deterministic generated and target image tensors.
#
# Parameters
# ----------
# requires_grad : bool
#     Whether the generated image requires gradients.
#
# Returns
# -------
# tuple[torch.Tensor, torch.Tensor]
#     Generated and target image tensors in BCHW order.
def _make_images(
    requires_grad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20261009)
    generated_image = torch.rand(
        2,
        NUM_IMAGE_CHANNELS,
        48,
        40,
        generator=generator,
        dtype=torch.float32,
        requires_grad=requires_grad,
    )
    target_image = torch.rand(
        2,
        NUM_IMAGE_CHANNELS,
        48,
        40,
        generator=generator,
        dtype=torch.float32,
    )
    return generated_image, target_image


# Summary
# -------
# _make_fixed_sfl creates a fixed-mode SFL module for one filter backend.
#
# Parameters
# ----------
# filter_backend : str
#     Filter backend name.
# normalize_gaussian : bool
#     Whether finite Gaussian kernels are normalized to sum to one.
#
# Returns
# -------
# SpatialFrequencyLoss
#     Configured fixed-mode SFL module.
def _make_fixed_sfl(
    filter_backend: str,
    normalize_gaussian: bool,
) -> SpatialFrequencyLoss:
    return SpatialFrequencyLoss(
        num_image_channels=NUM_IMAGE_CHANNELS,
        base_scale=BASE_SCALE,
        num_subbands=NUM_SUBBANDS,
        scale_radix=SCALE_RADIX,
        filter_size_factor=FILTER_SIZE_FACTOR,
        filter_backend=filter_backend,
        ier_update_mode="fixed",
        subband_weights=FIXED_WEIGHTS,
        normalize_gaussian=normalize_gaussian,
    )


# Summary
# -------
# _make_calibration_sfl creates a calibration-mode SFL module for one backend.
#
# Parameters
# ----------
# filter_backend : str
#     Filter backend name.
# normalize_gaussian : bool
#     Whether finite Gaussian kernels are normalized to sum to one.
#
# Returns
# -------
# SpatialFrequencyLoss
#     Configured calibration-mode SFL module.
def _make_calibration_sfl(
    filter_backend: str,
    normalize_gaussian: bool,
) -> SpatialFrequencyLoss:
    return SpatialFrequencyLoss(
        num_image_channels=NUM_IMAGE_CHANNELS,
        base_scale=BASE_SCALE,
        num_subbands=NUM_SUBBANDS,
        scale_radix=SCALE_RADIX,
        filter_size_factor=FILTER_SIZE_FACTOR,
        filter_backend=filter_backend,
        ier_update_mode="calibration",
        ier_reduction="sample",
        normalize_gaussian=normalize_gaussian,
    )


# Summary
# -------
# _assert_close checks numerical agreement and reports the maximum difference.
#
# Parameters
# ----------
# actual : torch.Tensor
#     Tensor produced by the separable backend.
# expected : torch.Tensor
#     Tensor produced by the Conv2d backend.
# description : str
#     Human-readable quantity name for an assertion failure.
# atol : float
#     Absolute tolerance.
# rtol : float
#     Relative tolerance.
#
# Returns
# -------
# None
def _assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    description: str,
    atol: float = ABSOLUTE_TOLERANCE,
    rtol: float = RELATIVE_TOLERANCE,
) -> None:
    maximum_absolute_difference = torch.max(
        torch.abs(actual - expected)
    ).item()
    assert torch.allclose(
        actual,
        expected,
        atol=atol,
        rtol=rtol,
    ), (
        f"{description} differs between filter backends: "
        f"maximum_absolute_difference={maximum_absolute_difference}, "
        f"atol={atol}, rtol={rtol}."
    )


# Summary
# -------
# test_filter_bank_outputs_match verifies all generated subband images.
#
# Command
# -------
# python3 -m pytest -q tests/test_spatial_frequency_loss_filter_backends_20261009.py::test_filter_bank_outputs_match
#
# Parameters
# ----------
# normalize_gaussian : bool
#     Parameter supplied by pytest.
#
# Returns
# -------
# None
@pytest.mark.parametrize("normalize_gaussian", [False, True])
def test_filter_bank_outputs_match(
    normalize_gaussian: bool,
) -> None:
    conv2d_sfl = _make_fixed_sfl("conv2d", normalize_gaussian)
    separable_sfl = _make_fixed_sfl("separable", normalize_gaussian)
    generated_image, _ = _make_images()

    conv2d_subbands = conv2d_sfl.filter_bank(generated_image)
    separable_subbands = separable_sfl.filter_bank(generated_image)

    assert len(conv2d_subbands) == NUM_SUBBANDS, (
        "The Conv2d backend returned an unexpected number of subbands: "
        f"actual={len(conv2d_subbands)}, expected={NUM_SUBBANDS}."
    )
    assert len(separable_subbands) == NUM_SUBBANDS, (
        "The separable backend returned an unexpected number of subbands: "
        f"actual={len(separable_subbands)}, expected={NUM_SUBBANDS}."
    )
    for index, (conv2d_subband, separable_subband) in enumerate(
        zip(conv2d_subbands, separable_subbands)
    ):
        _assert_close(
            separable_subband,
            conv2d_subband,
            f"Subband image {index + 1}",
            atol=SUBBAND_ABSOLUTE_TOLERANCE,
            rtol=RELATIVE_TOLERANCE,
        )


# Summary
# -------
# test_losses_match verifies subband losses, weighted SFL, and active weights.
#
# Command
# -------
# python3 -m pytest -q tests/test_spatial_frequency_loss_filter_backends_20261009.py::test_losses_match
#
# Parameters
# ----------
# normalize_gaussian : bool
#     Parameter supplied by pytest.
#
# Returns
# -------
# None
@pytest.mark.parametrize("normalize_gaussian", [False, True])
def test_losses_match(
    normalize_gaussian: bool,
) -> None:
    conv2d_sfl = _make_fixed_sfl("conv2d", normalize_gaussian)
    separable_sfl = _make_fixed_sfl("separable", normalize_gaussian)
    generated_image, target_image = _make_images()

    conv2d_outputs = conv2d_sfl(generated_image, target_image)
    separable_outputs = separable_sfl(generated_image, target_image)

    _assert_close(
        separable_outputs[0],
        conv2d_outputs[0],
        "Subband losses",
    )
    _assert_close(
        separable_outputs[1],
        conv2d_outputs[1],
        "Weighted Spatial Frequency Loss",
    )
    assert torch.equal(separable_outputs[4], conv2d_outputs[4]), (
        "Active weights differ between fixed-mode filter backends."
    )


# Summary
# -------
# test_input_gradients_match verifies generated-image gradients from weighted SFL.
#
# Command
# -------
# python3 -m pytest -q tests/test_spatial_frequency_loss_filter_backends_20261009.py::test_input_gradients_match
#
# Parameters
# ----------
# normalize_gaussian : bool
#     Parameter supplied by pytest.
#
# Returns
# -------
# None
@pytest.mark.parametrize("normalize_gaussian", [False, True])
def test_input_gradients_match(
    normalize_gaussian: bool,
) -> None:
    conv2d_sfl = _make_fixed_sfl("conv2d", normalize_gaussian)
    separable_sfl = _make_fixed_sfl("separable", normalize_gaussian)
    base_generated_image, target_image = _make_images()
    conv2d_generated_image = base_generated_image.detach().clone().requires_grad_(True)
    separable_generated_image = base_generated_image.detach().clone().requires_grad_(True)

    conv2d_weighted_loss = conv2d_sfl(
        conv2d_generated_image,
        target_image,
    )[1]
    separable_weighted_loss = separable_sfl(
        separable_generated_image,
        target_image,
    )[1]
    conv2d_weighted_loss.backward()
    separable_weighted_loss.backward()

    assert conv2d_generated_image.grad is not None, (
        "The Conv2d backend produced no generated-image gradient."
    )
    assert separable_generated_image.grad is not None, (
        "The separable backend produced no generated-image gradient."
    )
    _assert_close(
        separable_generated_image.grad,
        conv2d_generated_image.grad,
        "Generated-image gradients",
        atol=5.0e-6,
        rtol=5.0e-5,
    )


# Summary
# -------
# test_calibrated_ier_weights_match verifies sample-reduced calibration results.
#
# Command
# -------
# python3 -m pytest -q tests/test_spatial_frequency_loss_filter_backends_20261009.py::test_calibrated_ier_weights_match
#
# Parameters
# ----------
# normalize_gaussian : bool
#     Parameter supplied by pytest.
#
# Returns
# -------
# None
@pytest.mark.parametrize("normalize_gaussian", [False, True])
def test_calibrated_ier_weights_match(
    normalize_gaussian: bool,
) -> None:
    conv2d_sfl = _make_calibration_sfl("conv2d", normalize_gaussian)
    separable_sfl = _make_calibration_sfl("separable", normalize_gaussian)
    generated_image, target_image = _make_images()

    conv2d_sfl.accumulate_calibration_ier(
        generated_image,
        target_image,
    )
    separable_sfl.accumulate_calibration_ier(
        generated_image,
        target_image,
    )
    conv2d_sfl.finalize_ier_calibration()
    separable_sfl.finalize_ier_calibration()

    _assert_close(
        separable_sfl.subband_weights,
        conv2d_sfl.subband_weights,
        "Calibrated IER weights",
        atol=IER_ABSOLUTE_TOLERANCE,
        rtol=IER_RELATIVE_TOLERANCE,
    )
    assert torch.count_nonzero(conv2d_sfl._ier_sum).item() == 0, (
        "The Conv2d backend retained IER statistics after finalization."
    )
    assert torch.count_nonzero(separable_sfl._ier_sum).item() == 0, (
        "The separable backend retained IER statistics after finalization."
    )
    assert conv2d_sfl._num_ier_observations.item() == 0, (
        "The Conv2d backend retained an observation count after finalization."
    )
    assert separable_sfl._num_ier_observations.item() == 0, (
        "The separable backend retained an observation count after finalization."
    )
