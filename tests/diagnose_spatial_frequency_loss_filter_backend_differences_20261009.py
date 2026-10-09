#
# diagnose_spatial_frequency_loss_filter_backend_differences_20261009.py
#
# Measure numerical differences between Conv2d and separable DoG backends.
#
import torch

from spatial_frequency_loss import SpatialFrequencyLoss


NUM_IMAGE_CHANNELS = 3
BASE_SCALE = 0.6
NUM_SUBBANDS = 7
SCALE_RADIX = 2.0
FILTER_SIZE_FACTOR = 8.0
FIXED_WEIGHTS = [464.5, 277.2, 140.2, 69.70, 40.53, 26.70, 1.692]


# Summary
# -------
# _make_images creates deterministic generated and target image tensors.
#
# Parameters
# ----------
# None
#
# Returns
# -------
# tuple[torch.Tensor, torch.Tensor]
#     Generated and target image tensors in BCHW order.
def _make_images() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20261009)
    generated_image = torch.rand(
        2,
        NUM_IMAGE_CHANNELS,
        48,
        40,
        generator=generator,
        dtype=torch.float32,
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
# _make_sfl creates a fixed-mode SFL module for one filter backend.
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
def _make_sfl(
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
# _print_tensor_difference reports absolute-difference statistics.
#
# Parameters
# ----------
# description : str
#     Human-readable quantity name.
# actual : torch.Tensor
#     Tensor produced by the separable backend.
# expected : torch.Tensor
#     Tensor produced by the Conv2d backend.
#
# Returns
# -------
# float
#     Maximum absolute difference.
def _print_tensor_difference(
    description: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> float:
    absolute_difference = torch.abs(actual - expected)
    maximum_absolute_difference = absolute_difference.max().item()
    mean_absolute_difference = absolute_difference.mean().item()
    root_mean_square_difference = torch.sqrt(
        torch.mean(torch.square(actual - expected))
    ).item()
    maximum_reference_magnitude = torch.abs(expected).max().item()
    print(
        f"{description}: "
        f"max_abs={maximum_absolute_difference:.12e}, "
        f"mean_abs={mean_absolute_difference:.12e}, "
        f"rms={root_mean_square_difference:.12e}, "
        f"max_reference_abs={maximum_reference_magnitude:.12e}"
    )
    return maximum_absolute_difference


# Summary
# -------
# main measures all seven subbands without stopping at the first difference.
#
# Parameters
# ----------
# None
#
# Returns
# -------
# None
#
# Command
# -------
# python3 tests/diagnose_spatial_frequency_loss_filter_backend_differences_20261009.py
def main() -> None:
    generated_image, target_image = _make_images()

    for normalize_gaussian in (False, True):
        print(
            "NORMALIZE_GAUSSIAN="
            f"{normalize_gaussian}"
        )
        conv2d_sfl = _make_sfl("conv2d", normalize_gaussian)
        separable_sfl = _make_sfl("separable", normalize_gaussian)

        conv2d_subbands = conv2d_sfl.filter_bank(generated_image)
        separable_subbands = separable_sfl.filter_bank(generated_image)

        maximum_subband_difference = 0.0
        maximum_subband_index = 0
        for index, (conv2d_subband, separable_subband) in enumerate(
            zip(conv2d_subbands, separable_subbands),
            start=1,
        ):
            current_maximum = _print_tensor_difference(
                f"subband_{index}",
                separable_subband,
                conv2d_subband,
            )
            if current_maximum > maximum_subband_difference:
                maximum_subband_difference = current_maximum
                maximum_subband_index = index

        conv2d_outputs = conv2d_sfl(generated_image, target_image)
        separable_outputs = separable_sfl(generated_image, target_image)
        _print_tensor_difference(
            "subband_losses",
            separable_outputs[0],
            conv2d_outputs[0],
        )
        _print_tensor_difference(
            "weighted_sfl",
            separable_outputs[1],
            conv2d_outputs[1],
        )
        print(
            "largest_subband_difference: "
            f"subband={maximum_subband_index}, "
            f"max_abs={maximum_subband_difference:.12e}"
        )
        print()


if __name__ == "__main__":
    main()
