set autoscale
set term post eps enhanced color
set out 'graph.eps'

set title "Performance"
set xlabel "Threads"
set ylabel "Time [s]"

plot  "data/gauss.dat" using 1:2 title 'Gauss' w linespoints, \
      "data/jacobi.dat" using 1:2 title 'Jacobi' w linespoints, \
      "data/gaussmp.dat" using 1:2 title 'Gauss MP' w linespoints, \
	  "data/jacobimp.dat" using 1:2 title 'Jacobi MP' w linespoints
