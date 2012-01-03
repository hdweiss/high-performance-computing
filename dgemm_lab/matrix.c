
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

double** simple_mm(int m, int n, int k, double** a, double** b) {
    double** c = create_matrix(m, n);
    
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            for(int q = 0; q < k; q++) {
                c[i][j] = c[i][j] + a[i][q] * b[q][j];
            }
        }
    }

    return c;
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

/* void dgemm(char transa, char transb, int m, int n, int k, double alpha, */
/*                  double  *a,  int lda, double *b, int ldb, double beta, double */
/*                  *c, int ldc); */


double** dgemm_mm(int m, int n, int k, double** a, double** b) {
    double** c = create_matrix(m, n);

    double alpha = 1;
    double beta = 0;

    int lda = k; // 10. Parameter
    int ldb = m; // 8. Parameter

    int ldc = k; // 13. Parameter

    char trans = 'N';
    
    dgemm_64(trans, trans,
          n, m, k,
          alpha,
          b[0], ldb,
          a[0], lda,
          beta,
          c[0], ldc);

    return c;
}
