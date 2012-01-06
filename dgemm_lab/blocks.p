set autoscale
set term post eps enhanced color
set out 'graph_blocks.eps'

set title "Performance for blocksizes (for large matrices)"
set xlabel "Block dimension"
set ylabel "MFLOP/s"

plot  "blocks.dat" using 1:2 title 'Block\_mm AMD' w linespoints, \
	  "blocks.dat" using 1:3 title 'Block\_mm Intel' w linespoints
