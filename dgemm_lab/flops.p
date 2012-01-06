set autoscale
set term post eps enhanced color
set out 'graph_flops.eps'

set title "Floating Point Performance"
set xlabel "Matrix Sizes/KB"
set ylabel "MFLOP/s"
set logscale x

plot  "flops.dat" using 1:2 title 'Simple\_mm' w linespoints, \
      "flops.dat" using 1:4 title 'Block\_mm' w linespoints, \
      "flops.dat" using 1:3 title 'DGEMM\_mm' w linespoints
