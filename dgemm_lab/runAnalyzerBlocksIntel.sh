#!/bin/bash
module swap studio/12u3b

EXP_NAME=experiments/blocks_intel
COLLECT_CMD='/opt/oracle/solstudiodev/lib/analyzer/lib/../../../bin/collect'

MATRIX_SIZE=300

rm -Rf $EXP_NAME.*

for blocks in `seq 10 5 150`
do
    $COLLECT_CMD -o $EXP_NAME.$blocks.er -p on -h fp_comp_ops_exe.sse_fp,on,fp_comp_ops_exe.x87,on,sse_mem_exec.nta,on,cycles,on -S on -A on \
        ./mm $MATRIX_SIZE $MATRIX_SIZE $MATRIX_SIZE $blocks
done
