
MATRIX_SIZE=1000
MAX_ITERATIONS=10000


JACOBI=0
GAUSS=1
JACOBIMP=2
GAUSSMP=3

rm data/*_dsize.dat

echo "Running Jacobi Serial"
for i in `seq 100 50 $MATRIX_SIZE`
do
(time -p ./poisson $i $JACOBI $MAX_ITERATIONS) 2>&1 | \
    grep -iE real | \
	awk -v i=$i -v d=$MAX_ITERATIONS ' {printf "%d %.2f\n", i, d/$2} ' \
	>> data/jacobi_dsize.dat
done

echo "Running Gauss Serial"
for i in `seq 100 50 $MATRIX_SIZE`
do
(time -p ./poisson $i $GAUSS $MAX_ITERATIONS) 2>&1 | \
    grep -iE real | \
	awk -v i=$i -v d=$MAX_ITERATIONS ' {printf "%d %.2f\n", i, d/$2} ' \
 	>> data/gauss_dsize.dat
done

echo "Plotting"
gnuplot datasize.p
