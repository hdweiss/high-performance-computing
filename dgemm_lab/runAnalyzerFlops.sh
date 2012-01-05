#!/bin/bash
module swap studio/12u3b

EXP_NAME=experiments/flops
COLLECT_CMD='/opt/oracle/solstudiodev/lib/analyzer/lib/../../../bin/collect'

BLOCK_SIZE=16

rm -Rf $EXP_NAME.*

for m in `seq 10 10 100`
do
    $COLLECT_CMD -o $EXP_NAME.$m.er -p on -h PAPI_fp_ops,on,l2dm,on,dcm,on,cycles,on -S on -A on \
        ./mm $m $m $m $BLOCK_SIZE
done
