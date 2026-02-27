# Plot Fitness of each iteration to reflect the speed of convergence
# Input: Fitness.data
# Output: Fitness.png

set terminal pngcairo size 1200,800 enhanced font "Times New Roman,14"
set output "Fitness.png"

set title "PSO Convergence on Sphere Function (Normalized)" font "Times New Roman,18"
set xlabel "Iteration" font "Times New Roman,16"
set ylabel "Global Best Fitness Vlaue" font "Times New Roman,16"
set grid linewidth 0.5 linecolor rgb "#888888"
set key top right box width 2 spacing 1.5 font "Times New Roman,12"

set autoscale x
set logscale  y
set format y "10^{%L}"

set style line 1 linewidth 2.5 linecolor rgb "#E69F00" pointtype 7 pointsize 0.8

plot "Fitness.data" using 1:2 with lines title "G_{Best} Fitness"