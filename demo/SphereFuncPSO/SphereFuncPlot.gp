# SphereFuncPlot.gp - Gnuplot script for visualizing Standard Sphere function
# Input: --
# Output: SphereFunc.png

# Ouput Settings
set terminal pngcairo size 1200,800 enhanced font "Times New Roman,14"
set output "SphereFunc.png"

# Title & Label Settings
set title "Sphere Function: F(x)=x_{1}^{2}+x_{2}^{2}" font "Times New Roman,16"
set xlabel "x_{1}" font "Times New Roman,12"
set ylabel "x_{2}" font "Times New Roman,12"
set format z '%.1t×10^{%T}'
set ztics 0,4000,20000

# Graph Settings
set xyplane relative 0.2
set isosamples 100,100
set hidden3d
set grid

# Graph Plotting
splot [-100:100][-100:100][0:20000] x**2+y**2 with pm3d notitle