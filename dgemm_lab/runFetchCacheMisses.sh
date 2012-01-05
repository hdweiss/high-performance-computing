#!/bin/bash

EXP_FILES='experiments/cache*'

for file in $EXP_FILES
do
    echo $file | awk -F. {' printf "%i ", $2^2*3*8/1024 '};
    er_print -func $file 2> /dev/null | grep simple_mm | awk {' time=$2; cache=$6; printf "%.1f " , cache'};
    er_print -func $file 2> /dev/null | grep dgemm_mm | awk {' time=$2; cache=$6; printf "%.1f " , cache'};
    er_print -func $file 2> /dev/null | grep block_mm | awk {' time=$2; cache=$6; printf "%.1f " , cache'};
    printf "\n"
done

