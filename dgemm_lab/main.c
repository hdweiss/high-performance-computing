
#include <stdio.h>
#include "matrix.h"

int debug;

void print_timing(double time_simple, double time_dgemm, double time_block, int loopcount, int m, int n, int k) {
    if(debug) {
        int matrix_size = ((m*n + n*k + k*m) * sizeof(double)) / 1024;
        printf("%i %f %f %f\n", matrix_size, time_simple/loopcount, time_dgemm/loopcount, time_block/loopcount);
    }
}

void run_matrix_calc(int m, int n, int k, int loopcount, int s) {
    double** A = create_matrix(m, k);
    double** B = create_matrix(k, n);
    init_matrix(m, k, 10, A);
    init_matrix(k, n, 20, B);

    print_matrix(A, m, k, "A after init");
    print_matrix(B, k, n, "B after init");

    double simple_mm_time = 0;
    double gemm_mm_time = 0;
	double block_mm_time = 0;
    
    for(int i = 0; i < loopcount; i++) {
        double** C = create_matrix(m, n);
        simple_mm_time += simple_mm(m, n, k, A, B, C);
        print_matrix(C, m, n, "C after simple_mm");

        C = create_matrix(m, n);
        gemm_mm_time += dgemm_mm(m, n, k, A, B, C);
        print_matrix(C, m, n, "C after dgemm_mm");

		C = create_matrix(m, n);
        block_mm_time += block_mm(m, n, k, A, B, C, s);
        print_matrix(C, m, n, "C after block_mm");
    }

    //print_timing(simple_mm_time, gemm_mm_time, block_mm_time, loopcount+1, m, n, k);
}

int main(int argc, char** argv) {
    debug = 1;

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

    run_matrix_calc(m, n, k, loop_count, s);
    printf("Ran matrix calculations with %i %i %i\n", m, n, k);
    return 0;
}
