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

IMAGE_SIZE=2000

for i in `seq 1 20`
do
    printf "%i " $i
	r=0
	for n in `seq 1 1`
	do
		tmp=$((time -p OMP_NUM_THREADS=$i ./mandelbrot $IMAGE_SIZE) 2>&1 | grep real | awk {' print $2'} )
		r=$(float_eval "$r+$tmp")

	done
	ans=$(float_eval "$r / 1.0")

	printf "%f\n" $ans

done
