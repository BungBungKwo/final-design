# Plot Fitness of each iteration to reflect the speed of convergence
plot "Fitness.dat" using 1:2 with lines title "Fitness"
set terminal pngcairo
set output "Fitness.png"
replot