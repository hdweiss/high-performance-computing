
#include <stdio.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "integrand.h"
#include "mxv.h"

int debug;

double run_f(int n) {
    double h = 1.0 / (double)n;
    double sum = 0.0;

#pragma omp parallel for default(none) \
    shared(n,h) private(i,x)           \
    reduction(+: sum)
    for(int i=1; i<=n; i++) {
        double x = h * ((double)i + 0.5);
        sum += f(x);
    } /* end parallel */

    return sum;
}


int main(int argc, char** argv) {
    if(argc < 2) {
        perror("Need more arguments\n");
        return -1;
    }
        
    
    int n = atoi(argv[1]);
    debug = 0;
    double result = run_f(n);

    printf("Result: %f\n", result/n);
    return 0;
}

