#!/bin/bash
module swap studio/12u3b

EXP_NAME=experiments/hpc

rm -rf $EXP_NAME.*

for m in `seq 10 100 1000`
do
    /opt/oracle/solstudiodev/lib/analyzer/lib/../../../bin/collect -o $EXP_NAME.$m.er -p on -h PAPI_fp_ops,on,l2dm,on,dcm,on,cycles,on -S on -A on ./mm $m $m $m 
done
