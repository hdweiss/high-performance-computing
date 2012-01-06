#!/bin/bash
module swap studio/12u3b

EXP_NAME=experiments/flops

rm -Rf $EXP_NAME.*

BLOCK_SIZE=16

for m in `seq 10 10 100`
do
    /opt/oracle/solstudiodev/lib/analyzer/lib/../../../bin/collect -o $EXP_NAME.$m.er -p on -h PAPI_fp_ops,on,l2dm,on,dcm,on,cycles,on -S on -A on ./mm $m $m $m $BLOCK_SIZE
done
