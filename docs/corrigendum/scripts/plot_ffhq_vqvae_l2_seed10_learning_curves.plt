# Command
# -------
# cd docs/corrigendum/scripts && gnuplot plot_ffhq_vqvae_l2_seed10_learning_curves.plt
#
# Summary
# -------
# Generate the FFHQ VQ-VAE baseline L2-loss and subband-loss figures for
# seed 10 from the public learning-curve data.

DATA_FILE = "../data/ffhq_vqvae_l2_seed10_learning_curve.dat"
L2_OUTPUT = "../figures/ffhq_vqvae_l2_seed10_l2_loss.eps"
SUBBAND_OUTPUT = "../figures/ffhq_vqvae_l2_seed10_subband_losses.eps"

set terminal postscript eps enhanced color font ",30"
set xlabel "Epoch"
set xrange [1:75]

set output L2_OUTPUT
set ylabel "L2 Loss"
set yrange [0:0.01]
set ytics 0,0.001
plot DATA_FILE using 1:3 with lines linewidth 2 notitle
unset output

set output SUBBAND_OUTPUT
set ylabel "Subband Loss ({/Symbol \264}10^{-4})"
set yrange [0:0.00060]
set format y "%.0t"
set ytics 0,0.0001
plot DATA_FILE using 1:8 with lines linewidth 2 title "{SB}_{1}" at graph 0.3, graph 0.93, \
     DATA_FILE using 1:9 with lines linewidth 2 title "{SB}_{2}" at graph 0.3, graph 0.86, \
     DATA_FILE using 1:10 with lines linewidth 2 title "{SB}_{3}" at graph 0.3, graph 0.79, \
     DATA_FILE using 1:11 with lines linewidth 2 title "{SB}_{4}" at graph 0.75, graph 0.93, \
     DATA_FILE using 1:12 with lines linewidth 2 title "{SB}_{5}" at graph 0.75, graph 0.86, \
     DATA_FILE using 1:13 with lines linewidth 2 title "{SB}_{6}" at graph 0.75, graph 0.79, \
     DATA_FILE using 1:14 with lines linewidth 2 title "{SB}_{7}" at graph 0.75, graph 0.72
unset output
