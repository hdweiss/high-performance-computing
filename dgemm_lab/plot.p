set autoscale
set title "Simple vs. DGEMM"
set out 'matrix_graph.pdf'
set xlabel "Matrix Size/kb"
set ylabel "Time/s"
plot  "measure.dat" using 1:2 title 'Simple' w linespoints, \
      "measure.dat" using 1:3 title 'DGEMM' w linespoints
