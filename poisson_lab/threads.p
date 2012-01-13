set autoscale
set term post eps enhanced color
set out 'threads.eps'

set title "Performance"
set xlabel "Threads"
set ylabel "Iterations/sec"
set style line 1 lt 6 lw 3

plot  "data/gauss_threads.dat" using 1:2 title 'Gauss' w l ls 1, \
      "data/jacobi_threads.dat" using 1:2 title 'Jacobi' w l ls 1, \
      "data/gaussmp_threads.dat" using 1:2 title 'Gauss MP' w linespoints, \
	  "data/jacobimp_threads.dat" using 1:2 title 'Jacobi MP' w linespoints
