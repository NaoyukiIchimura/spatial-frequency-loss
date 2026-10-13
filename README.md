# Spatial Frequency Loss

Spatial Frequency Loss (SFL) measures reconstruction errors in overlapping
spatial-frequency subbands produced by a Difference-of-Gaussians filter bank.
The public implementation supports fixed subband weights and forward-only IER
calibration followed by fixed-weight optimization.

## Filter backend

The default backend is `separable`. It applies each isotropic Gaussian as
consecutive horizontal and vertical depthwise convolutions. The `conv2d`
backend remains available for compatibility and numerical validation.

The two backends implement the same mathematical filtering operation. The test
suite checks their agreement within numerical floating-point tolerances.

```python
from spatial_frequency_loss import SpatialFrequencyLoss

sfl = SpatialFrequencyLoss(
    num_image_channels=3,
    base_scale=0.6,
    num_subbands=7,
    ier_update_mode="calibration",
)
sfl = sfl.to(device)
```

To select the compatibility backend explicitly:

```python
sfl = SpatialFrequencyLoss(
    num_image_channels=3,
    base_scale=0.6,
    num_subbands=7,
    filter_backend="conv2d",
    ier_update_mode="calibration",
)
```

## IER calibration

The model forward pass and DataLoader traversal remain responsibilities of the
training program. Before optimization, accumulate IER statistics over the
calibration data and finalize the fixed weights.

```python
model.eval()
with torch.no_grad():
    for inputs, targets in calibration_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        sfl.accumulate_calibration_ier(outputs, targets)

sfl.finalize_ier_calibration()
```

During optimization:

```python
subband_losses, weighted_sfl, _, _, active_weights = sfl(outputs, targets)
total_loss = reconstruction_loss + quantizer_loss + weighted_sfl
```

For explicitly supplied weights, use `ier_update_mode="fixed"` together with
`subband_weights=[...]`.

## Checkpoint behavior

IER weights and calibration state are persistent buffers in `state_dict()`.
Save and restore the SFL module state together with the model and optimizer.
The filter backend itself is a construction-time configuration and must match
the intended checkpoint configuration.

## Computation time

The paper reported computation time using FFHQ with 40,000 images resized to
256 x 256, batch size 4, and three complete epochs of 10,000 iterations each.
The measured interval included GPU transfer, forward propagation, loss
computation, backward propagation, and the optimizer update.

| Loss condition | Training time (ms/iteration) | Relative time |
|---|---:|---:|
| Baseline | 50.50 +/- 0.043 | 1.00 |
| Baseline + FFL | 52.15 +/- 0.033 | 1.03 |
| Baseline + FRL | 59.67 +/- 0.063 | 1.18 |
| Baseline + SFL, conv2d | 267.9 +/- 0.724 | 5.31 |
| **Baseline + SFL, separable** | **66.851 +/- 0.021** | **1.324** |

The first four rows are from Table 6 of the paper. The separable backend was
implemented after publication and was therefore not available for the original
evaluation. The paper used the initial conv2d implementation of SFL to ensure a
fair comparison with the other methods under the implementation available at
that time.

The separable result is an additional post-publication measurement performed
with the same hardware, software environment, measurement interval, and
statistical procedure as the paper. The separable implementation was
approximately 4.01 times faster than the original conv2d implementation and
reduced SFL training time by approximately 75.05%. The two backends implement
the same mathematical filtering operation.

GPU-memory values are not included in this additional comparison because
allocator-based peak-memory measurements are implementation-dependent and
require cautious interpretation.

## Public scope

This public implementation includes `fixed` and `calibration` IER modes.
Dynamic weight control is not included.

## Paper

Naoyuki Ichimura, "Spatial frequency loss for mitigating spectral bias in deep
neural networks," Machine Vision and Applications 37, 173 (2026).
DOI: 10.1007/s00138-026-01927-5

## Tests

```bash
python3 -m pytest -q
```
