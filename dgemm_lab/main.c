
#include <stdio.h>
//#include <cstdlib>
#include "matrix.h"


int main() {

    int m = 3;
    int n = 2;
    int k = 5;

    double** A = create_matrix(m, k);
    double** B = create_matrix(k, n);
    init_matrix(m, k, 10, A);
    init_matrix(k, n, 20, B);

    print_matrix(A, m, k, "A after init");
    print_matrix(B, k, n, "B after init");
    
    double** C = simple_mm(m, n, k, A, B);
    print_matrix(C, m, n, "C after simple_mm");

    C = dgemm_mm(m, n, k, A, B);
    print_matrix(C, m, n, "C after dgemm_mm");
    
    return 0;
}
