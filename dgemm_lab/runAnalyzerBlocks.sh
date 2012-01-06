#!/bin/bash
module swap studio/12u3b

EXP_NAME=experiments/blocks
COLLECT_CMD='/opt/oracle/solstudiodev/lib/analyzer/lib/../../../bin/collect'

BLOCK_SIZE=32

rm -Rf $EXP_NAME.*

for m in `seq 1 50`
do
    $COLLECT_CMD -o $EXP_NAME.$m.er -p on -h PAPI_fp_ops,on,l2dm,on,dcm,on,cycles,on -S on -A on \
        ./mm 500 500 500 $m
done
