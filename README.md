# Spatial Frequency Loss

Spatial Frequency Loss (the SFL) is a subband-aware loss designed to mitigate
spectral bias in deep neural networks and improve the learning of fine-scale
image structures. The SFL measures reconstruction errors in overlapping
spatial-frequency subbands produced by a Difference-of-Gaussians filter bank.

- [Project page](https://naoyukiichimura.github.io/spatial-frequency-loss/)
- [Paper](https://doi.org/10.1007/s00138-026-01927-5)
- [View-only full text](https://rdcu.be/XSJ786PqnMca)
- [Supplementary material](https://media.springernature.com/original/springer-static/esm/art%3A10.1007%2Fs00138-026-01927-5/MediaObjects/138_2026_1927_MOESM1_ESM.pdf)
- [Post-publication corrections](https://naoyukiichimura.github.io/spatial-frequency-loss/corrections.html)

## Key features

- **Frequency-selective measurements:** A Difference-of-Gaussians filter bank
  measures discrepancies between corresponding spatial-frequency responses of
  output and target images.
- **Data-driven subband weighting:** The Inverse Error Ratio (IER) method
  determines the subband weights automatically from the training data.
- **Fidelity-oriented learning:** The SFL does not require adversarial training or
  nonlinear features from a pre-trained model. In the evaluated experiments,
  the SFL improved fine-scale image structures without introducing noticeable
  additional artifacts.
- **Minimal integration:** The SFL is implemented as a regular PyTorch module. Existing model
  architectures, data loaders, optimizers, and training loops require only
  small, localized additions.
- **Efficient default backend:** Separable filtering is the recommended default
  backend. The original two-dimensional convolution backend remains available
  for compatibility and numerical validation.

The fidelity-oriented design of the SFL may be particularly valuable when artificially
generated structures are undesirable, including potential applications in
medical imaging and physics-based image processing.

## Requirements

The public implementation uses:

- Python
- PyTorch
- NumPy

Use versions compatible with the PyTorch environment of the target project.

## Installation

Clone the repository and run Python from the repository root, or copy
`spatial_frequency_loss.py` into the target project.

```bash
git clone https://github.com/NaoyukiIchimura/spatial-frequency-loss.git
cd spatial-frequency-loss
```

## Quick start

The following example shows the simplest workflow in which the SFL automatically
determines the subband weights using IER before training.

### 1. Import the SFL

```python
import torch

from spatial_frequency_loss import SpatialFrequencyLoss
```

### 2. Initialize the SFL

```python
sfl = SpatialFrequencyLoss(
    num_image_channels=3,
    base_scale=0.6,
    num_subbands=7,
    ier_update_mode="calibration",
).to(device)

mse_loss = torch.nn.MSELoss()
```

`ier_update_mode="calibration"` is written explicitly for clarity, although
`"calibration"` is the default IER update mode. The default filter backend is
`"separable"`. The values `base_scale=0.6` and `num_subbands=7` match the
published experiments for RGB images.

### 3. Determine the subband weights using IER

```python
model.eval()
sfl.eval()

with torch.no_grad():
    for inputs, targets in training_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)

        sfl.accumulate_calibration_ier(
            outputs,
            targets,
        )

sfl.finalize_ier_calibration()
```

The model forward pass and DataLoader traversal remain responsibilities of the
training program. The same training data are used for this forward-only IER pass,
and model parameters are
not updated. `accumulate_calibration_ier()` collects the IER statistics, and
`finalize_ier_calibration()` sets the subband weights from the accumulated
statistics.

### 4. Train the model with the SFL

```python
model.train()
sfl.train()

for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        l2_loss = mse_loss(
            outputs,
            targets,
        )

        (
            subband_losses,
            weighted_sfl,
            _,
            _,
            _,
        ) = sfl(
            outputs,
            targets,
        )

        total_loss = l2_loss + weighted_sfl

        total_loss.backward()
        optimizer.step()
```

The calibration step determines the subband weights only. During optimization,
the weighted SFL is simply added to the existing pixel-wise L2 loss.

## IER calibration workflow

1. Construct SFL with `ier_update_mode="calibration"`.
2. Set the model and the SFL module to evaluation mode.
3. Traverse the training data without gradient computation or optimizer
   updates.
4. Pass each output-target pair to `accumulate_calibration_ier()`.
5. Call `finalize_ier_calibration()` once after the complete calibration pass.
6. Start normal optimization and call the SFL through `forward()`.

The default `ier_reduction="sample"` computes one IER vector per sample and
averages the vectors over all training samples. The alternative
`ier_reduction="batch"` computes one vector from the batch-mean losses and
averages the vectors across calibration batches.

After `finalize_ier_calibration()` has been called, the calibrated weights are
fixed during optimization. To repeat calibration, construct a new SFL module.

## Initialization parameters

| Parameter | Default | Description |
|---|---:|---|
| `num_image_channels` | Required | Number of image channels. |
| `base_scale` | Required | Base Gaussian scale. The published RGB experiments used `0.6`. |
| `num_subbands` | Required | Number of the SFL subbands. The published experiments used `7`. |
| `scale_radix` | `2.0` | Ratio between adjacent Gaussian scales. |
| `filter_size_factor` | `8.0` | Factor used to determine each finite filter size. |
| `filter_backend` | `"separable"` | `"separable"` is recommended; `"conv2d"` is retained for compatibility and validation. |
| `ier_update_mode` | `"calibration"` | `"calibration"` determines weights from a forward-only pass; `"fixed"` uses supplied weights. |
| `ier_reduction` | `"sample"` | Averages sample-wise or batch-wise IER vectors. |
| `eps` | `1.0e-12` | Positive lower bound for IER denominators. |
| `subband_weights` | `None` | Explicit weights. Required when `ier_update_mode="fixed"`. |
| `normalize_gaussian` | `False` | Whether each finite Gaussian kernel is normalized to sum to one. |

### Explicitly fixed subband weights

Use fixed mode when the subband weights are already known.

```python
sfl = SpatialFrequencyLoss(
    num_image_channels=3,
    base_scale=0.6,
    num_subbands=7,
    ier_update_mode="fixed",
    subband_weights=[
        weight_sb1,
        weight_sb2,
        weight_sb3,
        weight_sb4,
        weight_sb5,
        weight_sb6,
        weight_sb7,
    ],
).to(device)
```

To observe how another loss function changes the subband losses without adding
the SFL to the optimization objective, use fixed mode with all subband weights
set to zero. The returned `weighted_sfl` is then zero, while the returned
`subband_losses` remain available for monitoring.

```python
sfl_for_monitoring = SpatialFrequencyLoss(
    num_image_channels=3,
    base_scale=0.6,
    num_subbands=7,
    ier_update_mode="fixed",
    subband_weights=[
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
).to(device)

(
    subband_losses,
    weighted_sfl,
    _,
    _,
    _,
) = sfl_for_monitoring(
    outputs,
    targets,
)
```

In this monitoring configuration, optimize the model with the other loss
function only and record `subband_losses` to analyze the spatial-frequency error
in each subband.

## Return values

Calling the SFL returns five values:

```python
(
    subband_losses,
    weighted_sfl,
    output_subbands,
    target_subbands,
    active_weights,
) = sfl(
    outputs,
    targets,
)
```

- `subband_losses`: one batch-mean L2 loss per subband.
- `weighted_sfl`: scalar weighted sum of the subband losses.
- `output_subbands`: subband images obtained from the output image, ordered
  from high to low spatial frequency.
- `target_subbands`: subband images obtained from the target image, in the
  same order.
- `active_weights`: detached copy of the active subband weights.

The public `filter_bank()` method may also be used when the subband images are
needed explicitly.

```python
output_subbands = sfl.filter_bank(outputs)
```

## Filter backends

### Separable backend

The default `"separable"` backend applies each isotropic Gaussian as consecutive
horizontal and vertical depthwise convolutions. It is mathematically equivalent
to the corresponding two-dimensional Gaussian filtering operation and is the
recommended backend for normal use.

### Conv2d backend

The `"conv2d"` backend applies two-dimensional depthwise Gaussian convolution.
It was the initial implementation used throughout the paper to maintain fair
comparisons across the reported experiments. It remains available for
compatibility and numerical validation.

```python
sfl = SpatialFrequencyLoss(
    num_image_channels=3,
    base_scale=0.6,
    num_subbands=7,
    filter_backend="conv2d",
    ier_update_mode="calibration",
).to(device)
```

The test suite checks agreement between the backends within floating-point
numerical tolerances.

## Checkpoint save and restore

The active IER weights, accumulated calibration statistics, observation count,
and calibration-ready state are persistent buffers in `state_dict()`.

```python
torch.save(
    {
        "model": model.state_dict(),
        "sfl": sfl.state_dict(),
        "optimizer": optimizer.state_dict(),
    },
    checkpoint_path,
)
```

Reconstruct the model and SFL module with the intended configuration before
loading the checkpoint.

```python
checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
)

model.load_state_dict(checkpoint["model"])
sfl.load_state_dict(checkpoint["sfl"])
optimizer.load_state_dict(checkpoint["optimizer"])
```

`filter_backend` is a construction-time configuration and is not selected by
`state_dict()`. Construct SFL with the backend intended for the restored run.

## Computation time

The paper measured FFHQ training with 40,000 images resized to 256 x 256, batch
size 4, and three complete epochs of 10,000 iterations each. The interval
included GPU transfer, forward propagation, loss computation, backward
propagation, and the optimizer update.

| Loss condition | Training time (ms/iteration) | Relative time |
|---|---:|---:|
| Baseline | 50.50 +/- 0.043 | 1.00 |
| Baseline + FFL | 52.15 +/- 0.033 | 1.03 |
| Baseline + FRL | 59.67 +/- 0.063 | 1.18 |
| Baseline + SFL, conv2d | 267.9 +/- 0.724 | 5.31 |
| **Baseline + SFL, separable** | **66.85 +/- 0.021** | **1.32** |

The first four rows are from Table 6 of the paper, where FFL and FRL stand for
Focal Frequency Loss and Feature Reconstruction Loss, respectively. The
original SFL implementation used the conv2d backend, which was retained
throughout the paper to ensure fair comparisons across experiments. Because
separable filtering is mathematically equivalent and substantially faster, the
separable backend was implemented after publication and is used as the default
in the public software.

A post-publication measurement using the same measurement procedure found an
approximately 4.01-fold speed-up over the original SFL implementation and an
approximately 75.05% reduction in the SFL training time.

## Testing

Run the public test suite from the repository root.

```bash
python3 -m pytest -q
```

This command asks pytest to discover and execute the public test files in the
repository. The `-q` option selects quieter output. The individual test cases
should be reviewed separately before a detailed coverage list is added here.

## Public repository scope

This repository contains the public SFL implementation, documentation, and unit
tests. The public API supports calibrated and explicitly fixed IER weights.

## Results and project information

The published experiments evaluated the SFL with VQ-VAE models on FFHQ and
COCO2017. The SFL improved the reported validation image-quality metrics and reduced
high-frequency subband losses relative to the pixel-wise baseline. See the
[project page](https://naoyukiichimura.github.io/spatial-frequency-loss/) for visual examples, quantitative tables,
computation-time context, and post-publication information.

## Paper

Naoyuki Ichimura, "Spatial frequency loss for mitigating spectral bias in deep
neural networks," *Machine Vision and Applications* 37, 173 (2026).
[DOI: 10.1007/s00138-026-01927-5](https://doi.org/10.1007/s00138-026-01927-5)

## Citation

A machine-readable `CITATION.cff` file will be added to the repository. Until
then, cite the paper using the bibliographic information above.

## License

The Spatial Frequency Loss software is licensed under the GNU General Public
License, version 3 only (`GPL-3.0-only`). Commercial use, modification, and
redistribution are permitted under the terms of the license. If a modified
version of the SFL or a derivative program based on the SFL is distributed, the
corresponding source code must be provided under GPLv3.

See [LICENSE](LICENSE) for the complete license terms.

Copyright (C) 2026 Naoyuki Ichimura
