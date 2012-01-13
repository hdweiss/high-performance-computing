set autoscale
set term post eps enhanced color
set out 'datasize.eps'

set title "Performance"
set xlabel "GridSize"
set ylabel "MFLOP/s"

plot  "data/jacobi_dsize.dat" using 1:2 title 'Jacobi' w linespoints, \
      "data/gauss_dsize.dat" using 1:2 title 'Gauss' w linespoints

