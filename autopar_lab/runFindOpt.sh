#!/bin/bash

MATRIX_SIZE=10000

for i in `seq 1 20`
do
    printf "%i" $i
    (time -p OMP_NUM_THREADS=$i ./main_mp $MATRIX_SIZE $MATRIX_SIZE) 2>&1 | grep real | \
        awk {' printf " %f\n", $2'} 

done
