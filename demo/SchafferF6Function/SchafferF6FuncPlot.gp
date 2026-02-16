# SchafferF6Plot.gp - Gnuplot script for visualizing Schaffer F6 function
# Usage: 

# Schaffer F6 Function Figure

# Terminal Setting and Output
set terminal pngcairo  transparent enhanced font "arial,10" fontscale 1.0 size 1200, 800 
set output 'SchafferF6Func.png'

# Title & Labels
set title "Schaffer F6 Function"
set xlabel "x_{1}"
set ylabel "x_{2}"
set zlabel "f(x_{1},x_{2})"

# Axis Range Setting
set xrange[-100:100]
set yrange[-100:100]

# 3D Plot Settings
set pm3d
set hidden3d
set view 60,15,1,1
# Schaffer F6 Function Plot
splot "Schaffer_F6_Data.dat" with pm3d notitle