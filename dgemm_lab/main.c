
#include <stdio.h>
#include "matrix.h"

int debug;

void print_timing(double simple_count, double dgemm_count, double block_count, int m, int n, int k) {
    if(debug) {
        int matrix_size = ((m*n + n*k + k*m) * sizeof(double)) / 1024;

        /* FILE *file; */
        /* file = fopen("experiments/counts.txt","a+"); */
        
        /* fprintf(file, "%i %i %i %i\n", m, simple_count, dgemm_count, block_count); */
        /* fclose(file); */
    }
}

void run_matrix_calc(int m, int n, int k, int s) {
    double** A = create_matrix(m, k);
    double** B = create_matrix(k, n);
    init_matrix(m, k, 10, A);
    init_matrix(k, n, 20, B);

    print_matrix(A, m, k, "A after init");
    print_matrix(B, k, n, "B after init");

    double simple_mm_count = 0;
    double gemm_mm_count = 0;
	double block_mm_count = 0;
    
    double** C;

    simple_mm_count = simple_mm(m, n, k, A, B, C);
    print_matrix(C, m, n, "C after simple_mm");

	gemm_mm_count = dgemm_mm(m, n, k, A, B, C);
    print_matrix(C, m, n, "C after dgemm_mm");

    block_mm_count = block_mm(m, n, k, A, B, C, s);
    print_matrix(C, m, n, "C after block_mm");

	print_timing(simple_mm_count, gemm_mm_count, block_mm_count, m, n, k);
}

int main(int argc, char** argv) {
    debug = 0;

    if(argc < 4) {
        printf("Invaled number (%i) of arguments!\n", argc-1);
        return 0;
    }
        
    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);

    int s = 1;
    if(argc >= 5)
        s = atoi(argv[4]);
    
    int loop_count = 5;

    run_matrix_calc(m, n, k, s);
    //printf("Ran matrix calculations with %i %i %i %i\n", m, n, k, s);

    return 0;
}
