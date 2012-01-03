
#include <stdio.h>
#include <stdlib.h>
#include <sunperf.h>

void print_matrix(double** matrix, int row, int column, const char* message);
void init_matrix(int row, int column, int scale_factor, double** matrix);
double** create_matrix(int row, int column);

double** simple_mm(int m, int n, int k, double** a, double** b);
double** dgemm_mm(int m, int n, int k, double** a, double** b);
