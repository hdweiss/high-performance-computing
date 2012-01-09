
#ifdef _OPENMP
#include <omp.h>
#endif

#include <stdio.h>
#include <math.h>

#include "mxv.h"

int debug;

double f(double x) {
    return 4.0/(1 + pow(x,2));
}

double run_f(int n) {
    double h = 1.0 / (double)n;
    double sum = 0.0;

#pragma omp parallel for \
    reduction(+: sum)
    for(int i=1; i<=n; i++) {
        double x = h * ((double)i - 0.5);
        sum += f(x);
    } /* end parallel */

    return sum/n;
}

void mvm(int m, int n) {
    
    double** A = create_matrix(m, n);
    init_matrix(m, n, 1.0, A);
    double** B = create_matrix(1, n);
    init_matrix(1, n, 2.0, B);

    double** C = create_matrix(1, m);

    mxv(m, n, A, B, C);
}

int main(int argc, char** argv) {
    if(argc < 3) {
        perror("Need more arguments\n");
        return -1;
    }
        
    
    int n = atoi(argv[1]);
    int m = atoi(argv[2]);
    debug = 0;

    mvm(m, n);
    
    /* double result = run_f(n); */
    /* printf("Result: %f\n", result); */
    return 0;
}

