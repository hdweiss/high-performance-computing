#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "poisson.h"

#ifdef _OPENMP
#include <omp.h>
#endif

double f(int i, int j, int n, double delta) {
	double n2 = (double)(n+2)/2.0;
	double x = (i - n2) * delta;
	double y = (j - n2) * delta;
	if ( (x >= 0.0) && (x <= 1.0/3.0) && (y >= -2.0/3.0) && (y <= -1.0/3.0))
		return 500.0;
	return 0.0;
}

double threshold (double *u, double *u_old, int n) {

	double sum = 0.0;
	for (int i = 1; i <= n; i++) {
		for (int j = 0; j <= n; j++) {
			sum += pow ( ( u[i*(n+2)+j] - u_old[i*(n+2)+j] ),2 );
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

    printf("NumThreads 1\n");

    for (int k = 0; k < kmax; k++) {
		for (int i = 1; i <= n; i++) {
			for (int j = 1; j <= n; j++) {
				double f_ij = f( i, j, n+2, delta );
				double bot = grid_old[(i-1)*(n+2)+j];
				double top = grid_old[(i+1)*(n+2)+j];
				double left = grid_old[i*(n+2)+j-1];
				double right = grid_old[i*(n+2)+j+1];
				grid[i*(n+2)+j] = 0.25*(bot + top + left + right + delta2*f_ij);
				//printf("%.1f ",grid[i*(n+2)+j]);
			}
            //printf("\n");
		}
        double *tmp = grid_old;
        grid_old = grid;
        grid = tmp;		
        
	}
	return;
}

void gauss(double* img,double *img_old, int N, int max_iter){
	double delta = 2.0/(double)(N+2);
	double delta2 = pow(delta,2);

    printf("NumThreads 1\n");

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
				//printf("%.1f ",f(i,j,N+2,delta)*delta2);
			}
			//printf("\n");
		}
	}
	return;
}

void jacobi_mp(double *grid, double *grid_old, int n, int kmax) {
	double delta = 2.0/(double)(n+2);
	double delta2 = pow(delta,2);

#pragma omp parallel
    {
#pragma omp master
        printf("Numthreads %i\n", omp_get_num_threads());
        for (int k = 0; k < kmax; k++) {

#pragma omp for
            for (int i = 1; i <= n; i++) {
                for (int j = 1; j <= n; j++) {
                    double f_ij = f( i, j, n+2, delta );
                    double bot = grid_old[(i-1)*(n+2)+j];
                    double top = grid_old[(i+1)*(n+2)+j];
                    double left = grid_old[i*(n+2)+j-1];
                    double right = grid_old[i*(n+2)+j+1];
                    grid[i*(n+2)+j] = 0.25*(bot + top + left + right + delta2*f_ij);
                    //printf("%.1f ",grid[i*(n+2)+j]);
                }
                //printf("\n");
            } /* End for */

#pragma omp master {
            double *tmp = grid_old;
            grid_old = grid;
            grid = tmp;
        } /* End master */

        }
            
	} /* End parallel */
	return;
}

void gauss_mp(double* img,double *img_old, int N, int max_iter){
	double delta = 2.0/(double)(N+2);
	double delta2 = pow(delta,2);
    int* flags;
    // TODO Check for corner cases with regards to flags
    // TODO Make threads ordered (or think of other crazy solution)
    
#pragma omp parallel
    {

        int thread_id = omp_get_thread_num();
#pragma omp master
            {
        printf("Numthreads %i\n", omp_get_num_threads());
        flags = malloc(sizeof(int) * omp_get_num_threads() + 2);
        }

        for(int k = 0; k < max_iter; k++){

            while(flags[thread_id+1]+1 == k && flags[thread_id-1] == k+1);
            // TODO Potential speedup by kicking of the previous thread after calculated current line
                
#pragma omp for nowait
            for(int i = 1; i <= N; i++){
                for(int j = 1; j <= N; j++){
/*				img[i*(N+2)+j] = 0.25*(   img[(i-1)*(N+2)+ j   ] \
										+ img[(i+1)*(N+2)+ j   ] \
										+ img[ i   *(N+2)+(j-1)] \
										+ img[ i   *(N+2)+(j+1)] \
										+ delat2*f( i, j, N+2, delta ));
*/
				img[i*(N+2)+j] = 0.25*(       img[(i-1)*(N+2)+ j   ] \
										+ img_old[(i+1)*(N+2)+ j   ] \
										+     img[ i   *(N+2)+(j-1)] \
										+ img_old[ i   *(N+2)+(j+1)] \
										+ delta2*f( i, j, N+2, delta ));
				//printf("%.1f ",f(i,j,N+2,delta)*delta2);
                } /* for j */
            } /* parallel for i */
            
            flags[thread_id] = k;
#pragma omp flush(flags)
			//printf("\n");
		} /* for k*/
	}
	return;
}


void poisson(int n, double *grid, double th, int kmax, int choice) {

	double *grid_old = (double *)malloc( (n+2) * (n+2) * sizeof(double));
	init_array(grid_old, 10, n);

	/* double d = 1.0; */
	/* while (d > th) { */

    double *tmp = grid_old;
    grid_old = grid;
    grid = tmp;		

    if (choice == 0) {
//        printf("Poisson calculating. Using the jacobi method.");
        jacobi(grid, grid_old, n, kmax);
    }
    else if (choice == 1) {
//        printf("Poisson calculating. Using the gauss method.");
        gauss(grid, grid_old, n, kmax);
    }
    else if(choice == 2) {
//        printf("Poisson calculating. Using the jacobi_mp method.");
        jacobi_mp(grid, grid_old, n, kmax);
    }
    else {
//        printf("Poisson calculating. Using the gauss_mp method.");
        gauss_mp(grid, grid_old, n, kmax);
    }
        
		/* d = threshold(grid, grid_old, n); */
		/* printf("d %f\n", d); */

	/* } */
}

