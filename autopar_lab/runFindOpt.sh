#!/bin/bash

float_scale=2

function float_eval()
{
    local stat=0
    local result=0.0
    if [[ $# -gt 0 ]]; then
        result=$(echo "scale=$float_scale; $*" | bc -q 2>/dev/null)
        stat=$?
        if [[ $stat -eq 0  &&  -z "$result" ]]; then stat=1; fi
    fi
    echo $result
    return $stat
}

MATRIX_SIZE=10000

for i in `seq 1 20`
do
    printf "%i " $i
	r=0
	r_mp=0
	for n in `seq 1 10`
	do
		tmp=$((time -p OMP_NUM_THREADS=$i ./main $MATRIX_SIZE $MATRIX_SIZE) 2>&1 | grep real | awk {' print $2'} )
		tmp_mp=$((time -p OMP_NUM_THREADS=$i ./main_mp $MATRIX_SIZE $MATRIX_SIZE) 2>&1 | grep real | awk {' print $2'} )
		r=$(float_eval "$r+$tmp")
		r_mp=$(float_eval "$r_mp+$tmp_mp")
	done
	ans=$(float_eval "$r / 10.0")
	ans_mp=$(float_eval "$r_mp / 10.0")
	printf "%f %f\n" $ans $ans_mp

done
