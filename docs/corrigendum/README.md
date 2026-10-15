# Supporting Data for the Table 2 Corrigendum

## Purpose

This directory provides the learning-curve data, gnuplot scripts, and reproduced figures supporting the correction of the baseline loss row, denoted by `L_B`, in Table 2 of the article *Spatial Frequency Loss for Mitigating Spectral Bias in Deep Neural Networks*.

The baseline experiment was run with seeds 10, 20, and 30. Supplementary Table S9 identifies seed 10 as the formally selected run because seed 10 had the smallest final pixel-wise L2 loss. The published Table 2 row retained the subband-loss values from seed 20. The corrected row uses the corresponding values from seed 10.

## Verification at Epoch 75

The Table 2 subband-loss values are obtained from columns 8 through 14 at epoch 75. Values below are shown in the same units and precision as Table 2, namely `x 10^-5`.

| Run | SB1 | SB2 | SB3 | SB4 | SB5 | SB6 | SB7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Published values, seed 20 | 17.37 | 23.77 | 26.68 | 17.00 | 7.524 | 4.782 | 17.89 |
| Corrected values, seed 10 | 17.17 | 22.84 | 23.06 | 12.40 | 6.747 | 5.167 | 24.36 |

The seed-20 epoch-75 data reproduce all seven published values after rounding. The seed-10 epoch-75 data reproduce all seven corrected values after rounding.

## Directory Structure

```text
corrigendum/
├── README.md
├── SHA256SUMS.txt
├── data/
│   ├── ffhq_vqvae_l2_seed10_learning_curve.dat
│   └── ffhq_vqvae_l2_seed20_learning_curve.dat
├── scripts/
│   ├── plot_ffhq_vqvae_l2_seed10_learning_curves.plt
│   └── plot_ffhq_vqvae_l2_seed20_learning_curves.plt
└── figures/
    ├── ffhq_vqvae_l2_seed10_l2_loss.eps
    ├── ffhq_vqvae_l2_seed10_subband_losses.eps
    ├── ffhq_vqvae_l2_seed20_l2_loss.eps
    └── ffhq_vqvae_l2_seed20_subband_losses.eps
```

Both learning-curve files contain 75 rows corresponding to epochs 1 through 75.

## Learning-Curve File Format

Each data file is a whitespace-delimited text file with 15 columns.

| Column | Content |
|---:|---|
| 1 | Epoch |
| 2 | Consistency loss |
| 3 | Pixel-wise L2 reconstruction loss, PL2L |
| 4 | Quantizer loss |
| 5 | Codebook loss |
| 6 | Commitment loss |
| 7 | Weighted subband loss |
| 8 | Unweighted subband loss, SB1 |
| 9 | Unweighted subband loss, SB2 |
| 10 | Unweighted subband loss, SB3 |
| 11 | Unweighted subband loss, SB4 |
| 12 | Unweighted subband loss, SB5 |
| 13 | Unweighted subband loss, SB6 |
| 14 | Unweighted subband loss, SB7 |
| 15 | Feature reconstruction loss |

## Reproducing the Figures

The scripts were verified with gnuplot 5.4 patchlevel 2. Run the scripts from `docs/corrigendum/scripts`:

```bash
cd docs/corrigendum/scripts
gnuplot plot_ffhq_vqvae_l2_seed10_learning_curves.plt
gnuplot plot_ffhq_vqvae_l2_seed20_learning_curves.plt
```

Each script reads the corresponding file from `../data` and writes one L2-loss figure and one subband-loss figure to `../figures`.

The correct vertical-axis exponent for both subband-loss figures is `x 10^-4`. The plotted data and interpretation are unchanged by this exponent-label correction.

## Integrity Verification

From `docs/corrigendum`, run:

```bash
sha256sum -c SHA256SUMS.txt
```

Every listed file should report `OK`.
