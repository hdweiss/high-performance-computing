set autoscale
set out 'matrix_graph.pdf'

set title "Floating Point Performance"
set xlabel "Matrix Sizes/KB"
set ylabel "MFLOP/s"

plot  "measure.dat" using 1:2 title 'Simple_mm' w linespoints, \
      "measure.dat" using 1:4 title 'Block_mm' w linespoints, \
      "measure.dat" using 1:3 title 'DGEMM_mm' w linespoints
