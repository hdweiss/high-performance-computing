
#include <stdio.h>
#include <stdlib.h>

double** create_matrix(int row, int column);
void init_matrix(int row, int column, int scale_factor, double** matrix);
void destroy_matrix(double** matrix);
void print_matrix(double** matrix, int row, int column, const char* message);

void mxv (int m, int n, double **a, double **b, double **c);

