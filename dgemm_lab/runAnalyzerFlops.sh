#!/bin/bash
module swap studio/12u3b

EXP_NAME=experiments/flops
COLLECT_CMD='/opt/oracle/solstudiodev/lib/analyzer/lib/../../../bin/collect'

BLOCK_SIZE=60

rm -Rf $EXP_NAME.*

for m in 10 15 20 25 30 35 40 50 70 90 110 140 170 200 250 300 400 500 750 1000 #`seq 10 20 1000`
do
    $COLLECT_CMD -o $EXP_NAME.$m.er -p on -h PAPI_fp_ops,on,l2dm,on,dcm,on,cycles,on -S on -A on \
        ./mm $m $m $m $BLOCK_SIZE
done
