
#include "matrix.h"

double** create_matrix(int row, int column) {
    double** matrix = (double**) calloc(row, sizeof(double*));

    for(int p = 0; p < row; p++) {
        matrix[p] = (double*) calloc(column, sizeof(double));
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

void print_matrix(double** matrix, int row, int column, char* message) {
    printf("%s Matrix:\n", message);
    for(int i = 0; i < row; i++) {
        for(int j = 0; j < column; j++) {
            printf("%f\t", matrix[i][j]);
        }
        printf("\n");
    }
}
