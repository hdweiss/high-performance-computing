#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "poisson.h"

double f(int i, int j, int n, double delta) {
	double n2 = (double)(n+2)/2.0;
	double x = (i - n2) * delta;
	double y = (j - n2) * delta;
	if ( (x >= 0) && (x <= 1/3) && (y >= -2/3) && (y <= -1/3))
		return 200.0;
	return 0.0;
}

double threshold (double *u, double *u_old, int n) {

	double sum = 0.0;
	for (int i = 1; i <= n; i++) {
		for (int j = 0; j <= n; j++) {
			sum += sqrt ( abs (( u[i*(n+2)+j] - u_old[i*(n+2)+j] )) );
		}
	}
	return sqrt(sum);
}

void init_array(double* image, double value, int N){
	for(int i = 0; i <= N+1; i++){
		for(int j = 0; j <= N+1; j++){
			double tmp_value = value;
			if((j == 0)||(j == N+1)){
				tmp_value = 20.0;
			}
			if(i == 0){
				tmp_value = 20.0;
			} else if(i == N+1){
				tmp_value = 0.0;
			}
			image[i*(N+2)+j] = tmp_value;
		}
	}
}

void jacobi(double *grid, double *grid_old, int n, int kmax) {
	double delta = 2.0/(double)(n+2);
	double delta2 = pow(delta,2);
	for (int k = 0; k < kmax; k++) {
		for (int i = 1; i <= n; i++) {
			for (int j = 1; j <= n; j++) {
				double f_ij = f( i, j, n+2, delta );
				double bot = grid_old[(i-1)*(n+2)+j];
				double top = grid_old[(i+1)*(n+2)+j];
				double left = grid_old[i*(n+2)+j-1];
				double right = grid_old[i*(n+2)+j+1];
				grid[i*n+j] = 0.25*(bot + top + left + right + delta2*f_ij);
			}
		}
	}
	return;
}

void gauss(double* img,double *img_old, int N, int max_iter){
	double delta = 2.0/(double)(N+2);
	double delta2 = pow(delta,2);
	for(int k = 0; k < max_iter; k++){
		for(int i = 1; i <= N; i++){
			for(int j = 1; j <= N; j++){
/*				img[i*(N+2)+j] = 0.25*(   img[(i-1)*(N+2)+ j   ] \
										+ img[(i+1)*(N+2)+ j   ] \
										+ img[ i   *(N+2)+(j-1)] \
										+ img[ i   *(N+2)+(j+1)] \
										+ delat2*f( i, j, N+2, delta ));
*/
				img[i*(N+2)+j] = 0.25*(   img[(i-1)*(N+2)+ j   ] \
										+ img_old[(i+1)*(N+2)+ j   ] \
										+ img[ i   *(N+2)+(j-1)] \
										+ img_old[ i   *(N+2)+(j+1)] \
										+ delta2*f( i, j, N+2, delta ));
			}
		}
	}
	return;
}

void poisson(int n, double *grid, double th, int kmax, int choice) {

	double *grid_old = (double *)malloc( (n+2) * (n+2) * sizeof(double));
	init_array(grid_old, 10, n);

	double d = 1.0;
	while (d > th) {

		double *tmp = grid_old;
		grid_old = grid;
		grid = tmp;		

		if (choice == 0) {

			jacobi(grid, grid_old, n, kmax);
		}
		else {
			gauss(grid, grid_old, n, kmax);
		}

		d = threshold(grid, grid_old, n);
		printf("d %f\n", d);

	}
}

