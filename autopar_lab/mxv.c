#include "matrix.h"

extern int debug;

double** create_matrix(int row, int column) {
    double** matrix = (double**) malloc(row * sizeof(double*));
    double* space = (double*) calloc(row * column, sizeof(double));

    for(int p = 0; p < row; p++) {
        matrix[p] = space + p * column;
    }

    return matrix;
}

void destroy_matrix(double** matrix) {
    free(matrix[0]);
    free(matrix);
}

void init_matrix(int row, int column, int scale_factor, double** matrix) {
    for(int i = 0; i < row; i++) {
        for(int j = 0; j < column; j++) {
            matrix[i][j] = scale_factor * (i+1) + (j+1);
        }
    }
}

void print_matrix(double** matrix, int row, int column, const char* message) {
    if(debug == 0)
        return;

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

void mxv (int m, int n, double **a, double **b, double **c) {
	int i, j;
	double sum;
	for (i = 0; i < m; i++) {
		sum = 0.0;
		for (j = 0; j< n; j++) {
			sum += a[i][j]*b[j];
		}
		c[i] = sum;
	}
}

