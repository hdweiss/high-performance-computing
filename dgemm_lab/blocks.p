set autoscale
set term post eps enhanced color
set out 'graph_blocks.eps'

set title "Performance for blocksizes (for large matrices)"
set xlabel "Block size"
set ylabel "MFLOP/s"

plot  "blocks.dat" using 1:2 title 'Block\_mm' w linespoints
