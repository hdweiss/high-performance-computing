
#include <stdio.h>
#include "matrix.h"

int debug;

void print_timing(double time_simple, double time_dgemm, double time_block, int m, int n, int k) {
    if(debug) {
        int matrix_size = ((m*n + n*k + k*m) * sizeof(double)) / 1024;
        printf("%i %f %f %f\n", matrix_size, time_simple, time_dgemm, time_block);
    }
}

void run_matrix_calc(int m, int n, int k, int s) {
    double** A = create_matrix(m, k);
    double** B = create_matrix(k, n);
    init_matrix(m, k, 10, A);
    init_matrix(k, n, 20, B);

    print_matrix(A, m, k, "A after init");
    print_matrix(B, k, n, "B after init");

    double simple_mm_time = 0;
    double gemm_mm_time = 0;
	double block_mm_time = 0;
	int loopcount = 0;

    
    double** C;
	while(simple_mm_time < 3.0 && loopcount <= 2){
		C = create_matrix(m, n);
        simple_mm_time += simple_mm(m, n, k, A, B, C);
		loopcount++;
		free(C[0]);
		free(C);
	}
	simple_mm_time = simple_mm_time/loopcount;
	loopcount = 0;
    print_matrix(C, m, n, "C after simple_mm");

	while(gemm_mm_time < 3.0 && loopcount <= 2){
        C = create_matrix(m, n);
        gemm_mm_time += dgemm_mm(m, n, k, A, B, C);
		loopcount++;
		free(C[0]);
		free(C);
	}
	gemm_mm_time = gemm_mm_time/loopcount;
	loopcount=0;
    print_matrix(C, m, n, "C after dgemm_mm");

	while(block_mm_time < 3.0 && loopcount <= 2){
		C = create_matrix(m, n);
        block_mm_time += block_mm(m, n, k, A, B, C, s);
		loopcount++;
		free(C[0]);
		free(C);
	}
	block_mm_time = block_mm_time/loopcount;
    print_matrix(C, m, n, "C after block_mm");
    //}

    //print_timing(simple_mm_time, gemm_mm_time, block_mm_time, loopcount+1, m, n, k);
	print_timing(simple_mm_time, gemm_mm_time, block_mm_time, m, n, k);
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
    printf("Ran matrix calculations with %i %i %i %i\n", m, n, k, s);
    return 0;
}
