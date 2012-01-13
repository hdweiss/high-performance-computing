
MATRIX_SIZE=840
MAX_ITERATIONS=1
MAX_THRESHOLD=2


JACOBI=0
GAUSS=1
JACOBIMP=2
GAUSSMP=3

rm data/*_thres.dat


echo "Running Jacobi"
for i in `seq 0.1 0.1 $MAX_THRESHOLD`
do
	(time -p ./poisson $MATRIX_SIZE $JACOBI $MAX_ITERATIONS $i) 2>&1 | \
    grep -iE real | awk -v t=$i '{printf " %.2f %s\n", t, $2}' >> data/jacobi_thres.dat
done


echo "Running Gauss"
for i in `seq 0.1 0.1 $MAX_THRESHOLD`
do
	(time -p ./poisson $MATRIX_SIZE $GAUSS $MAX_ITERATIONS $i) 2>&1 | \
    grep -iE real | awk -v t=$i '{printf " %.2f %s\n", t, $2}' >> data/gauss_thres.dat
done

echo "Plotting"
gnuplot threshold.p
