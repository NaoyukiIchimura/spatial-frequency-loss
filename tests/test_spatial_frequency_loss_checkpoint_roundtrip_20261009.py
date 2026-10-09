#
# test_spatial_frequency_loss_checkpoint_roundtrip_20261009.py
#
# Checkpoint round-trip tests for SpatialFrequencyLoss.
#
import io

import pytest
import torch

from spatial_frequency_loss import SpatialFrequencyLoss


NUM_IMAGE_CHANNELS = 3
BASE_SCALE = 0.6
NUM_SUBBANDS = 4
SCALE_RADIX = 2.0
FILTER_SIZE_FACTOR = 8.0
FIXED_WEIGHTS = [4.0, 3.0, 2.0, 1.0]


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
    torch.manual_seed(20261009)
    generated_image = torch.rand(
        2,
        NUM_IMAGE_CHANNELS,
        32,
        32,
        dtype=torch.float32,
        requires_grad=requires_grad,
    )
    target_image = torch.rand(
        2,
        NUM_IMAGE_CHANNELS,
        32,
        32,
        dtype=torch.float32,
    )
    return generated_image, target_image


# Summary
# -------
# _round_trip_state_dict serializes and reloads a module state dictionary.
#
# Parameters
# ----------
# source : SpatialFrequencyLoss
#     Module whose state is serialized.
# destination : SpatialFrequencyLoss
#     Module into which the serialized state is loaded.
#
# Returns
# -------
# None
def _round_trip_state_dict(
    source: SpatialFrequencyLoss,
    destination: SpatialFrequencyLoss,
) -> None:
    checkpoint_buffer = io.BytesIO()
    torch.save(source.state_dict(), checkpoint_buffer)
    checkpoint_buffer.seek(0)
    restored_state_dict = torch.load(
        checkpoint_buffer,
        map_location="cpu",
        weights_only=True,
    )
    destination.load_state_dict(restored_state_dict)


# Summary
# -------
# test_fixed_checkpoint_round_trip verifies fixed-mode state, outputs, and
# gradients after checkpoint restoration.
#
# Command
# -------
# python3 -m pytest -q tests/test_spatial_frequency_loss_checkpoint_roundtrip_20261009.py::test_fixed_checkpoint_round_trip
#
# Parameters
# ----------
# None
#
# Returns
# -------
# None
def test_fixed_checkpoint_round_trip() -> None:
    source = SpatialFrequencyLoss(
        num_image_channels=NUM_IMAGE_CHANNELS,
        base_scale=BASE_SCALE,
        num_subbands=NUM_SUBBANDS,
        scale_radix=SCALE_RADIX,
        filter_size_factor=FILTER_SIZE_FACTOR,
        filter_backend="conv2d",
        ier_update_mode="fixed",
        subband_weights=FIXED_WEIGHTS,
        normalize_gaussian=False,
    )
    destination = SpatialFrequencyLoss(
        num_image_channels=NUM_IMAGE_CHANNELS,
        base_scale=BASE_SCALE,
        num_subbands=NUM_SUBBANDS,
        scale_radix=SCALE_RADIX,
        filter_size_factor=FILTER_SIZE_FACTOR,
        filter_backend="conv2d",
        ier_update_mode="fixed",
        subband_weights=[1.0, 1.0, 1.0, 1.0],
        normalize_gaussian=False,
    )

    _round_trip_state_dict(source, destination)

    assert torch.equal(destination.subband_weights, source.subband_weights), (
        "Fixed subband weights changed during checkpoint restoration: "
        f"source={source.subband_weights}, "
        f"destination={destination.subband_weights}."
    )

    generated_image, target_image = _make_images(requires_grad=True)
    source_outputs = source(generated_image, target_image)
    destination_outputs = destination(generated_image, target_image)

    assert torch.equal(source_outputs[0], destination_outputs[0]), (
        "Subband losses differ after fixed-mode checkpoint restoration."
    )
    assert torch.equal(source_outputs[1], destination_outputs[1]), (
        "Weighted SFL differs after fixed-mode checkpoint restoration."
    )

    destination_outputs[1].backward()
    assert generated_image.grad is not None, (
        "No gradient was produced after fixed-mode checkpoint restoration."
    )
    assert torch.isfinite(generated_image.grad).all(), (
        "A non-finite gradient was produced after fixed-mode checkpoint "
        "restoration."
    )


# Summary
# -------
# test_finalized_calibration_checkpoint_round_trip verifies that finalized IER
# weights and readiness state are restored without recalibration.
#
# Command
# -------
# python3 -m pytest -q tests/test_spatial_frequency_loss_checkpoint_roundtrip_20261009.py::test_finalized_calibration_checkpoint_round_trip
#
# Parameters
# ----------
# None
#
# Returns
# -------
# None
def test_finalized_calibration_checkpoint_round_trip() -> None:
    source = SpatialFrequencyLoss(
        num_image_channels=NUM_IMAGE_CHANNELS,
        base_scale=BASE_SCALE,
        num_subbands=NUM_SUBBANDS,
        filter_backend="conv2d",
        ier_update_mode="calibration",
        ier_reduction="sample",
        normalize_gaussian=False,
    )
    generated_image, target_image = _make_images()
    source.accumulate_calibration_ier(generated_image, target_image)
    source.finalize_ier_calibration()

    assert torch.count_nonzero(source._ier_sum).item() == 0, (
        "Finalized calibration retained nonzero accumulated IER values."
    )
    assert source._num_ier_observations.item() == 0, (
        "Finalized calibration retained a nonzero IER observation count."
    )
    assert bool(source._ier_weights_ready.item()), (
        "Finalized calibration did not mark IER weights as ready."
    )

    destination = SpatialFrequencyLoss(
        num_image_channels=NUM_IMAGE_CHANNELS,
        base_scale=BASE_SCALE,
        num_subbands=NUM_SUBBANDS,
        filter_backend="conv2d",
        ier_update_mode="calibration",
        ier_reduction="sample",
        normalize_gaussian=False,
    )
    _round_trip_state_dict(source, destination)

    assert torch.equal(destination.subband_weights, source.subband_weights), (
        "Calibrated weights changed during checkpoint restoration."
    )
    assert torch.count_nonzero(destination._ier_sum).item() == 0, (
        "Restored finalized calibration has nonzero accumulated IER values."
    )
    assert destination._num_ier_observations.item() == 0, (
        "Restored finalized calibration has a nonzero observation count."
    )
    assert bool(destination._ier_weights_ready.item()), (
        "Restored finalized calibration is not marked ready."
    )

    source_outputs = source(generated_image, target_image)
    destination_outputs = destination(generated_image, target_image)
    assert torch.equal(source_outputs[0], destination_outputs[0]), (
        "Subband losses differ after finalized-calibration restoration."
    )
    assert torch.equal(source_outputs[1], destination_outputs[1]), (
        "Weighted SFL differs after finalized-calibration restoration."
    )


# Summary
# -------
# test_partial_calibration_checkpoint_round_trip verifies restoration and
# continuation of an unfinished calibration pass.
#
# Command
# -------
# python3 -m pytest -q tests/test_spatial_frequency_loss_checkpoint_roundtrip_20261009.py::test_partial_calibration_checkpoint_round_trip
#
# Parameters
# ----------
# None
#
# Returns
# -------
# None
def test_partial_calibration_checkpoint_round_trip() -> None:
    source = SpatialFrequencyLoss(
        num_image_channels=NUM_IMAGE_CHANNELS,
        base_scale=BASE_SCALE,
        num_subbands=NUM_SUBBANDS,
        filter_backend="conv2d",
        ier_update_mode="calibration",
        ier_reduction="sample",
        normalize_gaussian=False,
    )
    generated_image, target_image = _make_images()
    source.accumulate_calibration_ier(generated_image, target_image)

    destination = SpatialFrequencyLoss(
        num_image_channels=NUM_IMAGE_CHANNELS,
        base_scale=BASE_SCALE,
        num_subbands=NUM_SUBBANDS,
        filter_backend="conv2d",
        ier_update_mode="calibration",
        ier_reduction="sample",
        normalize_gaussian=False,
    )
    _round_trip_state_dict(source, destination)

    assert torch.equal(destination._ier_sum, source._ier_sum), (
        "Accumulated IER values changed during partial-calibration restoration."
    )
    assert torch.equal(
        destination._num_ier_observations,
        source._num_ier_observations,
    ), "IER observation count changed during partial-calibration restoration."
    assert not bool(destination._ier_weights_ready.item()), (
        "Partial calibration was incorrectly restored as ready."
    )

    with pytest.raises(AssertionError, match="IER weights are not ready"):
        destination(generated_image, target_image)

    destination.accumulate_calibration_ier(generated_image, target_image)
    destination.finalize_ier_calibration()

    assert bool(destination._ier_weights_ready.item()), (
        "Continued calibration did not mark restored IER weights as ready."
    )
    assert torch.count_nonzero(destination._ier_sum).item() == 0, (
        "Continued calibration did not clear accumulated IER values."
    )
    assert destination._num_ier_observations.item() == 0, (
        "Continued calibration did not clear the IER observation count."
    )


# Summary
# -------
# test_incompatible_checkpoint_configuration_is_rejected verifies that state
# dictionaries cannot be loaded into incompatible subband or backend layouts.
#
# Command
# -------
# python3 -m pytest -q tests/test_spatial_frequency_loss_checkpoint_roundtrip_20261009.py::test_incompatible_checkpoint_configuration_is_rejected
#
# Parameters
# ----------
# None
#
# Returns
# -------
# None
def test_incompatible_checkpoint_configuration_is_rejected() -> None:
    source = SpatialFrequencyLoss(
        num_image_channels=NUM_IMAGE_CHANNELS,
        base_scale=BASE_SCALE,
        num_subbands=NUM_SUBBANDS,
        filter_backend="conv2d",
        ier_update_mode="fixed",
        subband_weights=FIXED_WEIGHTS,
        normalize_gaussian=False,
    )

    incompatible_subbands = SpatialFrequencyLoss(
        num_image_channels=NUM_IMAGE_CHANNELS,
        base_scale=BASE_SCALE,
        num_subbands=NUM_SUBBANDS + 1,
        filter_backend="conv2d",
        ier_update_mode="fixed",
        subband_weights=[5.0, 4.0, 3.0, 2.0, 1.0],
        normalize_gaussian=False,
    )
    with pytest.raises(RuntimeError):
        incompatible_subbands.load_state_dict(source.state_dict())

    incompatible_backend = SpatialFrequencyLoss(
        num_image_channels=NUM_IMAGE_CHANNELS,
        base_scale=BASE_SCALE,
        num_subbands=NUM_SUBBANDS,
        filter_backend="separable",
        ier_update_mode="fixed",
        subband_weights=FIXED_WEIGHTS,
        normalize_gaussian=False,
    )
    with pytest.raises(RuntimeError):
        incompatible_backend.load_state_dict(source.state_dict())
