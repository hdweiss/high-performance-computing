
echo "OpenMP:"
time OMP_NUM_THREADS=$1 ./main_mp $2

echo "Serial:"
time ./main $2
