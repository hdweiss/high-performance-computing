#!/bin/bash

EXP_FILES='experiments/hpc*'
FIND_COMMAND='find experiments/hpc* -maxdepth 0  -printf "%p$IFS"'

for file in $EXP_FILES
do
    echo $file | awk -F. {' printf "%i ", $2^2*3*8/1024 '};
    er_print -func $file 2> /dev/null | grep simple_mm | awk {' time=$2; flop=$4; printf "%.1f " , flop/time/1024/1024'};
    er_print -func $file 2> /dev/null | grep dgemm_mm | awk {' time=$2; flop=$4; printf "%.1f " , flop/time/1024/1024'};
    er_print -func $file 2> /dev/null | grep block_mm | awk {' time=$2; flop=$4; printf "%.1f " , flop/time/1024/1024'};
    printf "\n"
done

