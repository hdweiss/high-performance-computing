#!/bin/bash
module swap studio/12u3b

EXP_NAME=data/flops
COLLECT_CMD='/opt/oracle/solstudiodev/lib/analyzer/lib/../../../bin/collect'

rm -Rf $EXP_NAME.*

MATRIX_SIZE=1000
MAX_ITERATIONS=1000

MATRIX_SIZE_UNIT='^2*3*8/1024'
FLOPS_CALC='flop/time/1024/1024'


JACOBI=0
GAUSS=1
JACOBIMP=2
GAUSSMP=3

rm data/*_dsize.dat

echo "Running Jacobi Serial"
for i in `seq 100 50 $MATRIX_SIZE`
do
	 $COLLECT_CMD -o $EXP_NAME.jacobi.$i.er -p on -h PAPI_fp_ops,on,l2dm,on,dcm,on,cycles,on -S on -A on \
        ./poisson $i $JACOBI $MAX_ITERATIONS >> /dev/null
	echo -n $i >> data/jacobi_dsize.dat;
    er_print -func $EXP_NAME.jacobi.$i.er | grep jacobi | awk {' time=$2; flop=$4; printf " %.1f\n" , '$FLOPS_CALC} >> data/jacobi_dsize.dat;
	#echo $i $fl >> data/jacobi_dsize.dat

done

echo "Running Gauss Serial"
for i in `seq 100 50 $MATRIX_SIZE`
do
	 $COLLECT_CMD -o $EXP_NAME.gauss.$i.er -p on -h PAPI_fp_ops,on,l2dm,on,dcm,on,cycles,on -S on -A on \
        ./poisson $i $GAUSS $MAX_ITERATIONS >> /dev/null
	echo -n $i >> data/gauss_dsize.dat;
    er_print -func $EXP_NAME.gauss.$i.er | grep gauss | awk {' time=$2; flop=$4; printf " %.1f\n" , '$FLOPS_CALC} >> data/gauss_dsize.dat;
done

echo "Plotting"
gnuplot datasize.p
