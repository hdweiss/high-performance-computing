
MATRIX_SIZE=840
MAX_ITERATIONS=10000
MAX_THREAD=8


JACOBI=0
GAUSS=1
JACOBIMP=2
GAUSSMP=3

rm data/*_threads.dat

echo "Running Jacobi Serial"
(time -p ./poisson $MATRIX_SIZE $JACOBI $MAX_ITERATIONS) 2>&1 | \
    grep -iE 'real|NumThreads' | \
	awk -v d=$MAX_ITERATIONS -v t=$MAX_THREAD \
		'$1=="real" { printf "%.2f\n%d %.2f", d/$2, t, d/$2} $1=="NumThreads" {printf "%s ", $2} END {printf "\n"}' \
	> data/jacobi_threads.dat

echo "Running Gauss Serial"
(time -p ./poisson $MATRIX_SIZE $GAUSS $MAX_ITERATIONS) 2>&1 | \
    grep -iE 'real|NumThreads' | \
	awk -v d=$MAX_ITERATIONS -v t=$MAX_THREAD \
		'$1=="real" { printf "%.2f\n%d %.2f", d/$2, t, d/$2} $1=="NumThreads" {printf "%s ", $2} END {printf "\n"}' \
	> data/gauss_threads.dat

echo "Running Jacobi Parallel"
for i in `seq 1 $MAX_THREAD`
do
	echo $i
    (time -p OMP_NUM_THREADS=$i ./poisson $MATRIX_SIZE $JACOBIMP $MAX_ITERATIONS) 2>&1 | \
        grep -iE 'real|NumThreads' |  \
		awk -v d=$MAX_ITERATIONS ' $1=="real" { printf "%.2f ", d/$2} $1=="NumThreads" {printf "%s ", $2} END {printf "\n"}'\
		>> data/jacobimp_threads.dat
done

echo "Running Gauss Parallel"
for i in `seq 1 $MAX_THREAD`
do
	echo $i
    (time -p OMP_NUM_THREADS=$i ./poisson $MATRIX_SIZE $GAUSSMP $MAX_ITERATIONS) 2>&1 | \
        grep -iE 'real|NumThreads' | \
		awk -v d=$MAX_ITERATIONS '$1=="real" { printf "%.2f ", d/$2} $1=="NumThreads" {printf "%s ", $2} END {printf "\n"}' \
		>> data/gaussmp_threads.dat
done


echo "Plotting"
gnuplot threads.p
