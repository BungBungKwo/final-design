# Plot Fitness of each iteration to reflect the speed of convergence
# Input: Fitness.dat
# Output: Fitness.png

set terminal pngcairo size 1200,800 enhanced font "Arial,14"
set output "Fitness.png"

set title "PSO Convergence on Sphere Function (Normalized)" font "Arial,18"
set xlabel "Iteration" font "Arial,16"
set ylabel "Global Best Fitness (Log Scale)" font "Arial,16"
set grid linewidth 0.5 linecolor rgb "#888888"
set key top right box width 2 spacing 1.5 font "Arial,12"

set autoscale x
set logscale  y
set yrange [1e-65:1]
set format y "10^{%L}"

set style line 1 linewidth 2.5 linecolor rgb "#E69F00" pointtype 7 pointsize 0.8

plot "Fitness.dat" using 1:2 with linespoints title "G_{Best} Fitness"
