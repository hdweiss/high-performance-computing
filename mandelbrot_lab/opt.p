set autoscale
set term post eps enhanced color
set out 'opt.eps'

set title "Optimal Number of Threads"
set xlabel "Threads"
set ylabel "Time [s]"

plot  "opt.dat" using 1:2 title 'openMP' w linespoints
