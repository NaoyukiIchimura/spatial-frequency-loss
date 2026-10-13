#
# spatial_frequency_loss.py
#
# Spatial Frequency Loss (SFL) with integrated IER weight management.
#

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn

# Summary
# -------
# _Conv2dDoGFilterBank applies the fixed DoG filter bank using
# two-dimensional depthwise convolution.
#
# Parameters
# ----------
# num_image_channels : int
#     Number of image channels.
# base_scale : float
#     Scale of the highest-frequency Gaussian component.
# num_subbands : int
#     Number of SFL subbands, including the final low-pass subband.
# scale_radix : float
#     Ratio between adjacent Gaussian scales.
# filter_size_factor : float
#     Multiplicative factor used to determine each finite filter size.
# normalize_gaussian : bool
#     Whether each finite Gaussian kernel is normalized to sum to one.
#
# Returns
# -------
# None
class _Conv2dDoGFilterBank(nn.Module):

    # Summary
    # -------
    # _Conv2dDoGFilterBank.__init__ initializes the fixed DoG filter bank.
    def __init__(
        self,
        num_image_channels: int,
        base_scale: float,
        num_subbands: int,
        scale_radix: float,
        filter_size_factor: float,
        normalize_gaussian: bool,
    ) -> None:

        # Call the constructor of the parent class
        super().__init__()

        # Check the arguments
        assert num_image_channels > 0, (
            "The number of image channels must be positive, "
            f"but received num_image_channels={num_image_channels}."
        )
        assert base_scale > 0.0, (
            "The base scale must be positive, "
            f"but received base_scale={base_scale}."
        )
        assert num_subbands >= 2, (
            "At least two subbands are required, "
            f"but received num_subbands={num_subbands}."
        )
        assert scale_radix > 1.0, (
            "The scale radix must be greater than one, "
            f"but received scale_radix={scale_radix}."
        )
        assert filter_size_factor > 0.0, (
            "The filter-size factor must be positive, "
            f"but received filter_size_factor={filter_size_factor}."
        )

        # Set the class variables
        self.num_image_channels = num_image_channels
        self.num_subbands = num_subbands
        self.normalize_gaussian = normalize_gaussian

        # Set the scales and sizes of Gaussian kernels
        num_gaussian_scales = num_subbands - 1
        self.gaussian_scales = [
            base_scale * (scale_radix**index)
            for index in range(num_gaussian_scales)
        ]
        self.filter_sizes = [
            self._make_odd(
                math.ceil(filter_size_factor * scale)
            )
            for scale in self.gaussian_scales
        ]
        self.half_filter_sizes = [
            filter_size // 2
            for filter_size in self.filter_sizes
        ]

        # Set the Gaussian filters as convolutional layers
        self.gaussian_filters = nn.ModuleList()
        for scale, filter_size in zip(
            self.gaussian_scales,
            self.filter_sizes,
        ):
            # Apply each Gaussian kernel independently to every image channel
            convolution = nn.Conv2d(
                in_channels=num_image_channels,
                out_channels=num_image_channels,
                kernel_size=(filter_size, filter_size),
                stride=(1, 1),
                padding=(0, 0),  # Replicate padding is applied in forward().
                groups=num_image_channels,  # Apply depthwise convolution.
                bias=False,
            )
            kernel_array = self._create_gaussian_kernel_2d(
                filter_size=filter_size,
                scale=scale,
                normalize=normalize_gaussian,
            )
            kernel_tensor = torch.from_numpy(kernel_array).to(
                dtype=convolution.weight.dtype
            )
            # Reshape the two-dimensional kernel for depthwise Conv2d
            kernel_4d = kernel_tensor.view(
                1,
                1,
                filter_size,
                filter_size,
            )

            # Copy the same Gaussian kernel to every image channel
            depthwise_kernel = kernel_4d.repeat(
                num_image_channels,
                1,
                1,
                1,
            )
            with torch.no_grad():
                convolution.weight.copy_(depthwise_kernel)
            convolution.weight.requires_grad_(False)
            self.gaussian_filters.append(convolution)


    # Summary
    # -------
    # _create_gaussian_kernel_2d creates a finite-support two-dimensional
    # Gaussian kernel.
    #
    # Parameters
    # ----------
    # filter_size : int
    #     Odd kernel size.
    # scale : float
    #     Standard deviation of the Gaussian.
    # normalize : bool
    #     If True, normalize the finite kernel so that its sum is one.
    #
    # Returns
    # -------
    # np.ndarray
    #     Gaussian kernel with shape (filter_size, filter_size).
    @staticmethod
    def _create_gaussian_kernel_2d(
        filter_size: int,
        scale: float,
        normalize: bool = False,
    ) -> np.ndarray:

        # Check the arguments
        assert filter_size > 0 and filter_size % 2 == 1, (
            "The filter size of a Gaussian filter must be a positive odd "
            f"integer, but received filter_size={filter_size}."
        )
        assert scale > 0.0, (
            "The scale of a Gaussian filter must be positive, "
            f"but received scale={scale}."
        )

        # Compute half of the filter size
        half_filter_size = filter_size // 2

        # Create image-coordinate grids in the paper's (x, y) order
        # The x coordinate increases to the right, and the y coordinate
        # increases downward. Array elements remain indexed as [y, x].
        coordinates = np.arange(
            -half_filter_size,
            half_filter_size + 1,
            dtype=np.float64,
        )
        x_coordinates, y_coordinates = np.meshgrid(
            coordinates,
            coordinates,
            indexing="xy",
        )

        # Compute the finite-support Gaussian kernel
        variance = scale * scale
        squared_radius = x_coordinates**2 + y_coordinates**2
        normalized_constant = 1.0 / (2.0 * math.pi * variance)
        kernel = normalized_constant * np.exp(
            -squared_radius / (2.0 * variance)
        )

        # Normalize the kernel
        if normalize:
            kernel_sum = kernel.sum()
            assert kernel_sum > 0.0, (
                "The Gaussian kernel cannot be normalized because its sum is "
                f"not positive: kernel_sum={kernel_sum}."
            )
            kernel = kernel / kernel_sum

        return kernel

    # Summary
    # -------
    # _make_odd returns an odd integer greater than or equal to value.
    @staticmethod
    def _make_odd(value: int) -> int:
        return value if value % 2 == 1 else value + 1

    # Summary
    # -------
    # _Conv2dDoGFilterBank.forward decomposes an image into SFL subbands.
    def forward(self, image: Tensor) -> List[Tensor]:

        # Check the input image
        assert image.ndim == 4, (
            "The input image tensor must have BCHW order and four "
            f"dimensions, but received shape={tuple(image.shape)}."
        )
        assert image.shape[1] == self.num_image_channels, (
            "The number of input channels does not match the configured "
            f"filter bank: input_channels={image.shape[1]}, "
            f"configured_channels={self.num_image_channels}."
        )

        # Apply the Gaussian filters
        blurred_images = []
        for convolution, half_filter_size in zip(
            self.gaussian_filters,
            self.half_filter_sizes,
        ):
            # Pad the image by replicating its boundary values
            padded_image = torch.nn.functional.pad(
                image,
                pad=(
                    half_filter_size,
                    half_filter_size,
                    half_filter_size,
                    half_filter_size,
                ),
                mode="replicate",
            )
            blurred_images.append(convolution(padded_image))

        # Compute subband images
        subband_images = [image - blurred_images[0]]
        for index in range(1, len(blurred_images)):
            subband_images.append(
                blurred_images[index - 1] - blurred_images[index]
            )
        subband_images.append(blurred_images[-1])

        # Check the subband images
        assert len(subband_images) == self.num_subbands, (
            "The number of generated subbands is inconsistent with the "
            f"configuration: generated={len(subband_images)}, "
            f"expected={self.num_subbands}."
        )
        for index, subband_image in enumerate(subband_images):
            assert subband_image.shape == image.shape, (
                "A subband image has a shape inconsistent with the input: "
                f"subband_index={index}, subband_shape="
                f"{tuple(subband_image.shape)}, input_shape="
                f"{tuple(image.shape)}."
            )

        return subband_images

# Summary
# -------
# _SeparableDoGFilterBank applies the fixed DoG filter bank using separable
# horizontal and vertical depthwise convolutions.
#
# Parameters
# ----------
# num_image_channels : int
#     Number of image channels.
# base_scale : float
#     Scale of the highest-frequency Gaussian component.
# num_subbands : int
#     Number of SFL subbands, including the final low-pass subband.
# scale_radix : float
#     Ratio between adjacent Gaussian scales.
# filter_size_factor : float
#     Multiplicative factor used to determine each finite filter size.
# normalize_gaussian : bool
#     Whether each finite one-dimensional Gaussian kernel is normalized to
#     sum to one.
#
# Returns
# -------
# None
class _SeparableDoGFilterBank(nn.Module):

    # Summary
    # -------
    # _SeparableDoGFilterBank.__init__ initializes the separable DoG filter
    # bank.
    def __init__(
        self,
        num_image_channels: int,
        base_scale: float,
        num_subbands: int,
        scale_radix: float,
        filter_size_factor: float,
        normalize_gaussian: bool,
    ) -> None:

        # Call the constructor of the parent class
        super().__init__()

        # Check the arguments
        assert num_image_channels > 0, (
            "The number of image channels must be positive, "
            f"but received num_image_channels={num_image_channels}."
        )
        assert base_scale > 0.0, (
            "The base scale must be positive, "
            f"but received base_scale={base_scale}."
        )
        assert num_subbands >= 2, (
            "At least two subbands are required, "
            f"but received num_subbands={num_subbands}."
        )
        assert scale_radix > 1.0, (
            "The scale radix must be greater than one, "
            f"but received scale_radix={scale_radix}."
        )
        assert filter_size_factor > 0.0, (
            "The filter-size factor must be positive, "
            f"but received filter_size_factor={filter_size_factor}."
        )

        # Set the class variables
        self.num_image_channels = num_image_channels
        self.num_subbands = num_subbands
        self.normalize_gaussian = normalize_gaussian

        # Set the scales and sizes of Gaussian kernels
        num_gaussian_scales = num_subbands - 1
        self.gaussian_scales = [
            base_scale * (scale_radix**index)
            for index in range(num_gaussian_scales)
        ]
        self.filter_sizes = [
            self._make_odd(
                math.ceil(filter_size_factor * scale)
            )
            for scale in self.gaussian_scales
        ]
        self.half_filter_sizes = [
            filter_size // 2
            for filter_size in self.filter_sizes
        ]

        # Create horizontal and vertical depthwise Gaussian filters
        self.horizontal_gaussian_filters = nn.ModuleList()
        self.vertical_gaussian_filters = nn.ModuleList()

        for scale, filter_size in zip(
            self.gaussian_scales,
            self.filter_sizes,
        ):
            horizontal_convolution = nn.Conv2d(
                in_channels=num_image_channels,
                out_channels=num_image_channels,
                kernel_size=(1, filter_size),
                stride=(1, 1),
                padding=(0, 0),  # Replicate padding is applied in forward().
                groups=num_image_channels,
                bias=False,
            )
            vertical_convolution = nn.Conv2d(
                in_channels=num_image_channels,
                out_channels=num_image_channels,
                kernel_size=(filter_size, 1),
                stride=(1, 1),
                padding=(0, 0),  # Replicate padding is applied in forward().
                groups=num_image_channels,
                bias=False,
            )

            kernel_array = self._create_gaussian_kernel_1d(
                filter_size=filter_size,
                scale=scale,
                normalize=normalize_gaussian,
            )
            kernel_tensor = torch.from_numpy(kernel_array).to(
                dtype=horizontal_convolution.weight.dtype
            )

            # Shape: [1, 1, 1, filter_size]
            horizontal_kernel = kernel_tensor.view(
                1,
                1,
                1,
                filter_size,
            )
            horizontal_depthwise_kernel = horizontal_kernel.repeat(
                num_image_channels,
                1,
                1,
                1,
            )

            # Shape: [1, 1, filter_size, 1]
            vertical_kernel = kernel_tensor.view(
                1,
                1,
                filter_size,
                1,
            )
            vertical_depthwise_kernel = vertical_kernel.repeat(
                num_image_channels,
                1,
                1,
                1,
            )

            with torch.no_grad():
                horizontal_convolution.weight.copy_(
                    horizontal_depthwise_kernel
                )
                vertical_convolution.weight.copy_(
                    vertical_depthwise_kernel
                )

            horizontal_convolution.weight.requires_grad_(False)
            vertical_convolution.weight.requires_grad_(False)

            self.horizontal_gaussian_filters.append(
                horizontal_convolution
            )
            self.vertical_gaussian_filters.append(
                vertical_convolution
            )

    # Summary
    # -------
    # _create_gaussian_kernel_1d creates a finite-support one-dimensional
    # Gaussian kernel.
    #
    # Parameters
    # ----------
    # filter_size : int
    #     Odd kernel size.
    # scale : float
    #     Standard deviation of the Gaussian.
    # normalize : bool
    #     If True, normalize the finite kernel so that its sum is one.
    #
    # Returns
    # -------
    # np.ndarray
    #     Gaussian kernel with shape (filter_size,).
    @staticmethod
    def _create_gaussian_kernel_1d(
        filter_size: int,
        scale: float,
        normalize: bool = False,
    ) -> np.ndarray:

        # Check the arguments
        assert filter_size > 0 and filter_size % 2 == 1, (
            "The filter size of a Gaussian filter must be a positive odd "
            f"integer, but received filter_size={filter_size}."
        )
        assert scale > 0.0, (
            "The scale of a Gaussian filter must be positive, "
            f"but received scale={scale}."
        )

        # Compute half of the filter size
        half_filter_size = filter_size // 2

        # Create the one-dimensional image-coordinate axis
        coordinates = np.arange(
            -half_filter_size,
            half_filter_size + 1,
            dtype=np.float64,
        )

        # Compute the finite-support one-dimensional Gaussian kernel
        variance = scale * scale
        normalized_constant = 1.0 / math.sqrt(
            2.0 * math.pi * variance
        )
        kernel = normalized_constant * np.exp(
            -(coordinates**2) / (2.0 * variance)
        )

        # Normalize the finite one-dimensional kernel
        if normalize:
            kernel_sum = kernel.sum()
            assert kernel_sum > 0.0, (
                "The Gaussian kernel cannot be normalized because its sum is "
                f"not positive: kernel_sum={kernel_sum}."
            )
            kernel = kernel / kernel_sum

        return kernel

    # Summary
    # -------
    # _make_odd returns an odd integer greater than or equal to value.
    @staticmethod
    def _make_odd(value: int) -> int:
        return value if value % 2 == 1 else value + 1

    # Summary
    # -------
    # _SeparableDoGFilterBank.forward decomposes an image into SFL subbands.
    def forward(self, image: Tensor) -> List[Tensor]:

        # Check the input image
        assert image.ndim == 4, (
            "The input image tensor must have BCHW order and four "
            f"dimensions, but received shape={tuple(image.shape)}."
        )
        assert image.shape[1] == self.num_image_channels, (
            "The number of input channels does not match the configured "
            f"filter bank: input_channels={image.shape[1]}, "
            f"configured_channels={self.num_image_channels}."
        )

        # Apply each separable Gaussian filter
        blurred_images = []

        for (
            horizontal_convolution,
            vertical_convolution,
            half_filter_size,
        ) in zip(
            self.horizontal_gaussian_filters,
            self.vertical_gaussian_filters,
            self.half_filter_sizes,
        ):
            # Pad and filter along x, the horizontal image coordinate
            horizontally_padded_image = torch.nn.functional.pad(
                image,
                pad=(
                    half_filter_size,
                    half_filter_size,
                    0,
                    0,
                ),
                mode="replicate",
            )
            horizontally_blurred_image = horizontal_convolution(
                horizontally_padded_image
            )

            # Pad and filter along y, the downward vertical image coordinate
            vertically_padded_image = torch.nn.functional.pad(
                horizontally_blurred_image,
                pad=(
                    0,
                    0,
                    half_filter_size,
                    half_filter_size,
                ),
                mode="replicate",
            )
            blurred_image = vertical_convolution(
                vertically_padded_image
            )
            blurred_images.append(blurred_image)

        # Compute subband images
        subband_images = [image - blurred_images[0]]

        for index in range(1, len(blurred_images)):
            subband_images.append(
                blurred_images[index - 1] - blurred_images[index]
            )

        subband_images.append(blurred_images[-1])

        # Check the subband images
        assert len(subband_images) == self.num_subbands, (
            "The number of generated subbands is inconsistent with the "
            f"configuration: generated={len(subband_images)}, "
            f"expected={self.num_subbands}."
        )

        for index, subband_image in enumerate(subband_images):
            assert subband_image.shape == image.shape, (
                "A subband image has a shape inconsistent with the input: "
                f"subband_index={index}, subband_shape="
                f"{tuple(subband_image.shape)}, input_shape="
                f"{tuple(image.shape)}."
            )

        return subband_images


# Summary
# -------
# SpatialFrequencyLoss computes Spatial Frequency Loss and manages calibrated
# or explicitly fixed IER weights.
#
# Parameters
# ----------
# num_image_channels : int
#     Number of image channels.
# base_scale : float
#     Base Gaussian scale.
# num_subbands : int
#     Number of SFL subbands.
# scale_radix : float
#     Ratio between adjacent Gaussian scales.
# filter_size_factor : float
#     Factor used to determine finite filter sizes.
# filter_backend : str
#     "separable" uses consecutive horizontal and vertical depthwise
#     Gaussian convolutions and is the recommended default backend.
#     "conv2d" uses two-dimensional depthwise Gaussian convolution and is
#     retained for compatibility and numerical validation.
# ier_update_mode : str
#     "calibration" estimates IER weights in a forward-only calibration pass
#     before optimization and then keeps them fixed. "fixed" uses explicitly
#     supplied subband weights.
# ier_reduction : str
#     "batch" computes one IER vector from batch-mean losses and averages
#     those vectors over calibration batches. "sample" computes one IER vector
#     per sample and averages those vectors over all calibration samples.
# eps : float
#     Positive lower bound for an IER denominator.
# subband_weights : sequence of float, optional
#     Required when ier_update_mode is "fixed".
# normalize_gaussian : bool
#     Whether finite Gaussian kernels are normalized to sum to one.
#
# Returns
# -------
# None
class SpatialFrequencyLoss(nn.Module):

    # Summary
    # -------
    # SpatialFrequencyLoss.__init__ initializes SFL and its IER state.
    def __init__(
        self,
        num_image_channels: int,
        base_scale: float,
        num_subbands: int,
        scale_radix: float = 2.0,
        filter_size_factor: float = 8.0,
        filter_backend: str = "separable",
        ier_update_mode: str = "calibration",
        ier_reduction: str = "sample",
        eps: float = 1.0e-12,
        subband_weights: Optional[Sequence[float]] = None,
        normalize_gaussian: bool = False,
    ) -> None:

        # Call the constructor of the parent class
        super().__init__()

        # Check the arguments
        assert ier_update_mode in {"calibration", "fixed"}, (
            "The IER update mode must be 'calibration' or 'fixed', "
            f"but received ier_update_mode={ier_update_mode!r}."
        )
        assert ier_reduction in {"batch", "sample"}, (
            "The IER reduction must be 'batch' or 'sample', "
            f"but received ier_reduction={ier_reduction!r}."
        )
        assert eps > 0.0, (
            "The IER denominator lower bound must be positive, "
            f"but received eps={eps}."
        )

        # Set the class variables
        self.num_subbands = num_subbands
        self.filter_backend = filter_backend
        self.ier_update_mode = ier_update_mode
        self.ier_reduction = ier_reduction
        self.eps = eps

        # Create the selected filter-bank backend
        if filter_backend == "conv2d":
            self._filter_bank = _Conv2dDoGFilterBank(
                num_image_channels=num_image_channels,
                base_scale=base_scale,
                num_subbands=num_subbands,
                scale_radix=scale_radix,
                filter_size_factor=filter_size_factor,
                normalize_gaussian=normalize_gaussian,
            )
        elif filter_backend == "separable":
            self._filter_bank = _SeparableDoGFilterBank(
                num_image_channels=num_image_channels,
                base_scale=base_scale,
                num_subbands=num_subbands,
                scale_radix=scale_radix,
                filter_size_factor=filter_size_factor,
                normalize_gaussian=normalize_gaussian,
            )
        else:
            raise ValueError(
                "The filter backend must be either 'conv2d' or 'separable', "
                f"but received filter_backend={filter_backend!r}."
            )

        # Initialize placeholder subband weights with neutral unit values
        initial_weights = torch.ones(
            num_subbands,
            dtype=torch.float32,
        )
        if subband_weights is not None:
            assert len(subband_weights) == num_subbands, (
                "The number of supplied subband weights is inconsistent "
                f"with num_subbands: supplied={len(subband_weights)}, "
                f"expected={num_subbands}."
            )
            initial_weights = torch.as_tensor(
                subband_weights,
                dtype=torch.float32,
            ).clone()

        if ier_update_mode == "fixed":
            assert subband_weights is not None, (
                "subband_weights must be supplied when "
                "ier_update_mode='fixed'."
            )

        # Register the active subband weights as persistent model state
        self.register_buffer("subband_weights", initial_weights)

        # Accumulate IER values in float64 during calibration
        self.register_buffer(
            "_ier_sum",
            torch.zeros(num_subbands, dtype=torch.float64),
        )

        # Count batches in batch mode or samples in sample mode
        self.register_buffer(
            "_num_ier_observations",
            torch.zeros((), dtype=torch.long),
        )

        # Record whether weights are ready for optimization
        self.register_buffer(
            "_ier_weights_ready",
            torch.tensor(ier_update_mode == "fixed", dtype=torch.bool),
        )

    # Summary
    # -------
    # filter_bank applies the selected filter-bank backend to an image and returns
    # the SFL subbands.
    #
    # Parameters
    # ----------
    # image : Tensor
    #     Input image tensor in BCHW order.
    #
    # Returns
    # -------
    # List[Tensor]
    #     Subband image tensors in high-to-low spatial-frequency order.
    def filter_bank(self, image: Tensor) -> List[Tensor]:
        return self._filter_bank(image)

    # Summary
    # -------
    # _compute_losses computes batch-level and sample-level reconstruction losses.
    def _compute_losses(
        self,
        generated_image: Tensor,
        target_image: Tensor,
    ) -> Tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        List[Tensor],
        List[Tensor],
    ]:

        # Check the arguments
        assert generated_image.shape == target_image.shape, (
            "The generated and target tensors must have identical shapes: "
            f"generated_shape={tuple(generated_image.shape)}, "
            f"target_shape={tuple(target_image.shape)}."
        )
        assert generated_image.device == target_image.device, (
            "The generated and target tensors must be on the same device: "
            f"generated_device={generated_image.device}, "
            f"target_device={target_image.device}."
        )

        # Decompose the generated and target images into SFL subbands
        generated_subbands = self._filter_bank(generated_image)
        target_subbands = self._filter_bank(target_image)

        # Average over channel, height, and width while retaining the batch axis
        reduction_dimensions = tuple(range(1, generated_image.ndim))

        # Compute one pixel-wise L2 loss per sample
        # Shape: [batch_size]
        sample_pixel_losses = (
            (generated_image - target_image)
            .pow(2)
            .mean(dim=reduction_dimensions)
        )

        # Create one [batch_size] tensor for each subband. The Python-list
        # index corresponds to the subband index, and the list therefore
        # contains num_subbands tensors.
        sample_subband_loss_list = [
            (generated_subband - target_subband)
            .pow(2)
            .mean(dim=reduction_dimensions)
            for generated_subband, target_subband in zip(
                generated_subbands,
                target_subbands,
            )
        ]

        # Insert the new list-element dimension at dim=1. The size of this
        # dimension equals the number of tensors passed to torch.stack(),
        # which is num_subbands. Keeping batch at dim=0 and subband at dim=1
        # enables later broadcasting of [batch_size, 1] pixel losses over
        # [batch_size, num_subbands] subband losses.
        # Shape: num_subbands tensors of [batch_size]
        #     -> [batch_size, num_subbands]
        sample_subband_losses = torch.stack(
            sample_subband_loss_list,
            dim=1,
        )

        # Average sample losses over the current batch
        pixel_loss = sample_pixel_losses.mean()
        subband_losses = sample_subband_losses.mean(dim=0)

        return (
            pixel_loss,
            subband_losses,
            sample_pixel_losses,
            sample_subband_losses,
            generated_subbands,
            target_subbands,
        )

    # Summary
    # -------
    # _compute_ier_contribution computes the current batch's IER contribution.
    def _compute_ier_contribution(
        self,
        pixel_loss: Tensor,
        subband_losses: Tensor,
        sample_pixel_losses: Tensor,
        sample_subband_losses: Tensor,
    ) -> Tuple[Tensor, int]:

        # Compute one ratio from batch-mean losses in batch reduction mode
        if self.ier_reduction == "batch":
            batch_ier_weights = (
                pixel_loss.detach()
                / subband_losses.detach().clamp_min(self.eps)
            )
            ier_sum_contribution = batch_ier_weights.to(torch.float64)
            num_ier_observations_contribution = 1

            return (
                ier_sum_contribution,
                num_ier_observations_contribution,
            )

        # Compute one IER vector per sample in sample reduction mode
        # Shape: [batch_size, num_subbands]
        sample_ier_weights = (
            sample_pixel_losses.detach().unsqueeze(1)
            / sample_subband_losses.detach().clamp_min(self.eps)
        )

        # Compute the exact sample-wise contribution to the accumulated mean
        ier_sum_contribution = sample_ier_weights.sum(
            dim=0,
            dtype=torch.float64,
        )
        num_ier_observations_contribution = sample_ier_weights.shape[0]

        return (
            ier_sum_contribution,
            num_ier_observations_contribution,
        )

    # Summary
    # -------
    # _accumulate_ier_contribution adds one batch to the IER statistics.
    def _accumulate_ier_contribution(
        self,
        pixel_loss: Tensor,
        subband_losses: Tensor,
        sample_pixel_losses: Tensor,
        sample_subband_losses: Tensor,
    ) -> None:

        # Compute the current batch's contribution
        (
            ier_sum_contribution,
            num_ier_observations_contribution,
        ) = self._compute_ier_contribution(
            pixel_loss=pixel_loss,
            subband_losses=subband_losses,
            sample_pixel_losses=sample_pixel_losses,
            sample_subband_losses=sample_subband_losses,
        )

        # Accumulate the contribution
        self._ier_sum.add_(ier_sum_contribution)
        self._num_ier_observations.add_(
            num_ier_observations_contribution
        )

    # Summary
    # -------
    # _set_weights_from_ier_statistics sets mean IER weights and clears
    # the accumulated statistics.
    @torch.no_grad()
    def _set_weights_from_ier_statistics(self) -> None:

        # Check that at least one batch or sample was accumulated
        assert self._num_ier_observations.item() > 0, (
            "IER weights cannot be set because no IER observations were "
            "accumulated. Process the complete calibration pass before "
            "setting the weights."
        )

        # Compute the mean IER weights over batches or samples
        mean_ier_weights = (
            self._ier_sum
            / self._num_ier_observations
        )
        self.subband_weights.copy_(
            mean_ier_weights.to(dtype=self.subband_weights.dtype)
        )

        # Clear the consumed statistics so that no stale calibration
        # observations remain in the persistent module state
        self._ier_sum.zero_()
        self._num_ier_observations.zero_()

    # Summary
    # -------
    # accumulate_calibration_ier accumulates initial calibration statistics.
    @torch.no_grad()
    def accumulate_calibration_ier(
        self,
        generated_image: Tensor,
        target_image: Tensor,
    ) -> None:

        # Check the update mode and calibration state
        assert self.ier_update_mode == "calibration", (
            "accumulate_calibration_ier() requires "
            "ier_update_mode='calibration', but received "
            f"ier_update_mode={self.ier_update_mode!r}."
        )
        assert not bool(self._ier_weights_ready.item()), (
            "IER calibration cannot continue after the initial weights have "
            "been prepared. Create a new loss module to repeat calibration."
        )

        # Compute losses for one forward-only calibration batch
        (
            pixel_loss,
            subband_losses,
            sample_pixel_losses,
            sample_subband_losses,
            _,
            _,
        ) = self._compute_losses(
            generated_image,
            target_image,
        )

        # Accumulate calibration statistics
        self._accumulate_ier_contribution(
            pixel_loss=pixel_loss,
            subband_losses=subband_losses,
            sample_pixel_losses=sample_pixel_losses,
            sample_subband_losses=sample_subband_losses,
        )

    # Summary
    # -------
    # finalize_ier_calibration prepares initial weights after calibration.
    @torch.no_grad()
    def finalize_ier_calibration(self) -> None:

        # Check the update mode and calibration state
        assert self.ier_update_mode == "calibration", (
            "finalize_ier_calibration() requires "
            "ier_update_mode='calibration', but received "
            f"ier_update_mode={self.ier_update_mode!r}."
        )
        assert not bool(self._ier_weights_ready.item()), (
            "Initial IER calibration has already been completed."
        )

        # Set the initial weights from calibration statistics
        self._set_weights_from_ier_statistics()

        # Mark the initial IER weights as ready for optimization
        self._ier_weights_ready.fill_(True)

    # Summary
    # -------
    # forward computes SFL using calibrated or explicitly fixed weights.
    def forward(
        self,
        generated_image: Tensor,
        target_image: Tensor,
    ) -> Tuple[Tensor, Tensor, List[Tensor], List[Tensor], Tensor]:

        # Require calibrated or explicitly fixed weights before optimization
        assert bool(self._ier_weights_ready.item()), (
            "IER weights are not ready. Run a forward-only calibration pass "
            "with accumulate_calibration_ier(), then call "
            "finalize_ier_calibration() before starting optimization."
        )

        # Compute the losses
        (
            pixel_loss,
            subband_losses,
            sample_pixel_losses,
            sample_subband_losses,
            generated_subbands,
            target_subbands,
        ) = self._compute_losses(
            generated_image,
            target_image,
        )


        # Keep the persistent active weights unchanged, and create a temporary
        # dtype-matched tensor only for the current loss computation.
        active_weights = self.subband_weights
        active_weights_for_loss = active_weights.to(
            dtype=subband_losses.dtype
        )

        # Compute the scalar weighted Spatial Frequency Loss
        weighted_subband_loss = torch.sum(
            active_weights_for_loss * subband_losses
        )

        return (
            subband_losses,
            weighted_subband_loss,
            generated_subbands,
            target_subbands,
            active_weights.detach().clone(),
        )
