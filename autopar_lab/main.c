
#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif


int main(int argc, char** argv) {
    int n;

    double h = 1.0 / (double)n;
    double sum = 0.0;

#pragma omp parallel for default(none) \
    shared(n,h) private(i,x)           \
    reduction(+: sum)

    for(int i=1; i<=n; i++) {
        double x = h * ((double)i + 0.5);
        sum += f(x);
    }
}

