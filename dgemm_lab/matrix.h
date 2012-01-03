
#include <cstdlib>
#include <stdio.h>

void print_matrix(double** matrix, int row, int column, char* message);
double** simple_mm(int m, int n, int k, double** a, double** b);
void init_matrix(int row, int column, int scale_factor, double** matrix);
double** create_matrix(int row, int column);

