#!/bin/bash

EXP_FILES='experiments/blocks*'
MATRIX_SIZE='^2*3*8/1024'
FLOPS_CALC='flop/time/1024/1024'

for file in $EXP_FILES
do
    ER_OUT="$(mktemp)"
    er_print -func $file 2> /dev/null > $ER_OUT

    echo $file | awk -F. {' printf "%i ", $2'};

    cat $ER_OUT | grep simple_mm | awk {' time=$2; flop=$4; printf "%.1f " , '$FLOPS_CALC};

    printf "\n"
    rm -f $ER_OUT
done

