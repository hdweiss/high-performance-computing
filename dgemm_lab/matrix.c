
#include "matrix.h"

double** create_matrix(int row, int column) {
    double** matrix = (double**) malloc(row * sizeof(double*));
    double* space = (double*) calloc(row * column, sizeof(double));

    for(int p = 0; p < row; p++) {
        matrix[p] = space + p * column;
    }

    return matrix;
}

void init_matrix(int row, int column, int scale_factor, double** matrix) {
    for(int i = 0; i < row; i++) {
        for(int j = 0; j < column; j++) {
            matrix[i][j] = scale_factor * (i+1) + (j+1);
        }
    }
}

void print_matrix(double** matrix, int row, int column, const char* message) {
    if(matrix == NULL) {
        printf("Print got null matrix %s", message);
        return;
    }
        
    printf("%s Matrix:\n", message);
    for(int i = 0; i < row; i++) {
        for(int j = 0; j < column; j++) {
            printf("%.1f\t", matrix[i][j]);
        }
        printf("\n");
    }
}


double simple_mm(int m, int n, int k, double** a, double** b, double** c) {
    init_timer();
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            for(int q = 0; q < k; q++) {
                c[i][j] = c[i][j] + a[i][q] * b[q][j];
            }
        }
    }

    return xtime();
}

double dgemm_mm(int m, int n, int k, double** a, double** b, double** c) {
    double alpha = 1;
    double beta = 0;

    int lda = k; // 10. Parameter
    int ldb = n; // 8. Parameter

    int ldc = n; // 13. Parameter

    init_timer();
    dgemm_64('N', 'N',
          n, m, k,
          alpha,
          b[0], ldb,
          a[0], lda,
          beta,
          c[0], ldc);

    return xtime();
}
