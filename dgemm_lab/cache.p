set autoscale
set term post eps enhanced color
set out 'graph_cache.eps'

set title "Cache Misses"
set xlabel "Matrix Sizes [KB]"
set ylabel "Cache misses/s [Cache misses / 1000]"

plot  "cache.dat" using 1:2 title 'Simple\_mm' w linespoints, \
      "cache.dat" using 1:4 title 'Block\_mm' w linespoints, \
      "cache.dat" using 1:3 title 'DGEMM\_mm' w linespoints
