
MATRIX_SIZE=1000
MAX_ITERATIONS=100
MAX_THREAD=32


JACOBI=0
GAUSS=1
JACOBIMP=2
GAUSSMP=3

rm data/*.dat

(time -p ./poisson $MATRIX_SIZE $JACOBI $MAX_ITERATIONS) 2>&1 | \
    grep -iE 'real|NumThreads' | awk {'printf "%s ", $2} END {printf "\n"'}  > data/gauss.dat

(time -p ./poisson $MATRIX_SIZE $GAUSS $MAX_ITERATIONS) 2>&1 | \
    grep -iE 'real|NumThreads' | awk {'printf "%s ", $2} END {printf "\n"'} > data/jacobi.dat

for i in `seq 1 $MAX_THREAD`
do
    (time -p OMP_NUM_THREADS=$i ./poisson $MATRIX_SIZE $JACOBIMP $MAX_ITERATIONS) 2>&1 | \
        grep -iE 'real|NumThreads' |  awk {'printf "%s ", $2} END {printf "\n"'} >> data/jacobimp.dat
done

for i in `seq 1 $MAX_THREAD`
do
    (time -p OMP_NUM_THREADS=$i ./poisson $MATRIX_SIZE $GAUSSMP $MAX_ITERATIONS) 2>&1 | \
        grep -iE 'real|NumThreads' |  awk {'printf "%s ", $2} END {printf "\n"'} >> data/gaussmp.dat
done


gnuplot graph.p
