set autoscale
set term post eps enhanced color
set out 'threshold.eps'

set title "Convergence"
set xlabel "Threshold"
set ylabel "Time [s]"

plot  "data/jacobi_thres.dat" using 1:2 title 'Jacobi' w linespoints, \
      "data/gauss_thres.dat" using 1:2 title 'Gauss' w linespoints

