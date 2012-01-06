#!/bin/bash
module swap studio/12u3b

EXP_NAME=experiments/blocks
COLLECT_CMD='/opt/oracle/solstudiodev/lib/analyzer/lib/../../../bin/collect'

MATRIX_SIZE=300

rm -Rf $EXP_NAME.*

for blocks in `seq 10 10 100`
do
    $COLLECT_CMD -o $EXP_NAME.$blocks.er -p on -h PAPI_fp_ops,on,l2dm,on,dcm,on,cycles,on -S on -A on \
        ./mm $MATRIX_SIZE $MATRIX_SIZE $MATRIX_SIZE $blocks
done
