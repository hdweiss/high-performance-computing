#include <stdio.h>
#include <stdlib.h>

double f(int i, int j, int n, double delta) {
	double n2 = (double)(n+2)/2.0;
	double x = (i - n2) * delta;
	double y = (j - n2) * delta;
	if ( (x >= 0) && (x <= 1/3) && (y >= -2/3) && (y <= -1/3)) {
		return 200.0;
	return 0.0;
}

double threshold (double *u, double *u_old, int n) {

	double sum = 0.0;
	for (int i = 1; i <= n; i++) {
		for (int j = 0; j <= n; j++) {
			sum += sqrt( u[i*(n+2)+j] - u_old[i*(n+2)+j] );
		}
	}
	return sqrt(sum);
}

void jacobi(int n, double *grid, double *grid_old) {
	for (int i = 1; i <= n; i++) {
		for (int j = 1; j <= n; j++) {
			double f_ij = f( i, j, n+2, delta );
			double bot = grid_old[(i-1)*(n+2)+j];
			double top = grid_old[(i+1)*(n+2)+j];
			double left = grid_old[i*(n+2)+j-1];
			double right = grid_old[i*(n+2)+j+1];
			grid[i*n+j] = h*(bottom + top + left + right + delta2*f_ij);
		}
	}
}

void poisson(int n, double *grid, double th/*, int kmax*/) {

	double delta = 2.0/(double)(n+2);
	double delta2 = pow(delta,2);
	double h = 1.0/4.0;
	double *grid_old = (double *)malloc( (n+2) * (n+2) * sizeof(double));

	double d = 1;
	while (d > th) {
	//for (int k = 0; k < kmax; k++) {
		double tmp = grid_old;
		grid_old = grid;
		grid = tmp;

		jacobi(grid, grid_old, n);

		d = threshold(grid, grid_old, n);

	}
}

