
#include <stdio.h>
#include "matrix.h"


void print_timing(double time_simple, double time_dgemm) {
    printf("%f %f \n", time_simple, time_dgemm);
}

void run_matrix(int m, int n, int k) {

}

int main(int argc, char** argv) {

    int m = 1000;
    int n = 1000;
    int k = 1000;

    double** A = create_matrix(m, k);
    double** B = create_matrix(k, n);
    init_matrix(m, k, 10, A);
    init_matrix(k, n, 20, B);

//    print_matrix(A, m, k, "A after init");
//    print_matrix(B, k, n, "B after init");

    double** C = create_matrix(m, n);
    double simple_mm_time = simple_mm(m, n, k, A, B, C);
//    print_matrix(C, m, n, "C after simple_mm");

    C = create_matrix(m, n);
    double gemm_mm_time = dgemm_mm(m, n, k, A, B, C);
//    print_matrix(C, m, n, "C after dgemm_mm");
    print_timing(simple_mm_time, gemm_mm_time);
    
    
    return 0;
}
