#!/bin/bash

EXP_FILES='experiments/cache*'
CACHE_CALC='cache'

for file in $EXP_FILES
do
    ER_OUT="$(mktemp)"
    er_print -func $file 2> /dev/null > $ER_OUT

    echo $file | awk -F. {' printf "%i ", $2^2*3*8/1024 '};

    cat $ER_OUT | grep simple_mm | awk {' time=$2; cache=$6; printf "%.1f " , '$CACHE_CALC};
    cat $ER_OUT | grep dgemm_mm | awk {' time=$2; cache=$6; printf "%.1f " , '$CACHE_CALC};
    cat $ER_OUT | grep block_mm | awk {' time=$2; cache=$6; printf "%.1f " , '$CACHE_CALC};

    printf "\n"
    rm $ER_OUT
done

