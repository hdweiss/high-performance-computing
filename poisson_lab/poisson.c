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
		return 200.0;
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

void jacobi(double *grid, double *grid_old, int n, int kmax, double th) {
	double delta = 2.0/(double)(n+2);
	double delta2 = pow(delta,2);

    printf("NumThreads 1\n");

	double d = 1.0; 
	while (d > th) { 

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
//			printf("k:%d\n",k);
		}

		if ( th < 0.0 )
			break;

		d = threshold(grid, grid_old, n);
        printf("d %f\n", d);
	}
	return;
}

void gauss(double* img,double *img_old, int N, int max_iter, double th){
	double delta = 2.0/(double)(N+2);
	double delta2 = pow(delta,2);

    printf("NumThreads 1\n");

	double d = 1.0; 
	while (d > th) { 

	    for(int k = 0; k < max_iter; k++){
			for(int i = 1; i <= N; i++){
				for(int j = 1; j <= N; j++){
/*					img[i*(N+2)+j] = 0.25*(   img[(i-1)*(N+2)+ j   ] \
										+ img[(i+1)*(N+2)+ j   ] \
										+ img[ i   *(N+2)+(j-1)] \
										+ img[ i   *(N+2)+(j+1)] \
										+ delta2*f( i, j, N+2, delta ));
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
        	double *tmp = img_old;
        	img_old = img;
        	img = tmp;
//			printf("k:%d\n",k);
		}
		if (th < 0.0)
			break;

		d = threshold(img, img_old, N);
        printf("d %f\n", d);
	}
	return;
}

void jacobi_mp(double *grid, double *grid_old, int n, int kmax) {
	double delta = 2.0/(double)(n+2);
	double delta2 = pow(delta,2);

#pragma omp parallel
    {
#pragma omp master
        printf("NumThreads %i\n", omp_get_num_threads());
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

#pragma omp single
            {
            double *tmp = grid_old;
            grid_old = grid;
            grid = tmp;
        } /* End master */

        }
            
	} /* End parallel */
	return;
}

inline void do_work(double* img, double delta2, double delta, int N, int i) {
    for(int j = 1; j <= N; j++){
//        img[i*(N+2)+j] = 0.25*(       img[(i-1)*(N+2)+ j ]
  //                                    + img_old[(i+1)*(N+2)+ j ]
    //                                  +     img[ i   *(N+2)+(j-1)]          
      //                                + img_old[ i   *(N+2)+(j+1)]          
        //                              + delta2*f( i, j, N+2, delta ));
		double fij = f(i,j,N+2,delta);
    		img[i*(N+2)+j] = 0.25*(   img[(i-1)*(N+2)+ j   ]        \
                + img[(i+1)*(N+2)+ j   ]                                \
                + img[ i   *(N+2)+(j-1)]                                \
                + img[ i   *(N+2)+(j+1)]                                \
                + delta2*fij);
		//if(fij>0.0){printf("f: %.1f\n",fij);}
		//else if(delta == 0.0){printf("delta: %f\n",delta);}
	}
}

void gauss_mp(double* img, int N, int max_iter){
	double delta = 2.0/(double)(N+2);
	double delta2 = pow(delta,2);
    int* start;
    int* finish;
    
#pragma omp parallel shared(start, finish, delta, delta2,img,N,max_iter)
    {
        
#pragma omp single
        {
            printf("NumThreads %i\n", omp_get_num_threads());
            start = malloc(sizeof(int) * (omp_get_num_threads() + 2));
            finish = malloc(sizeof(int) * (omp_get_num_threads() + 2));
            start[omp_get_num_threads() + 1] = max_iter + 1;
            finish[0] = max_iter + 1;

        }

        int thread_id = omp_get_thread_num() + 1;
        start[thread_id] = 0;
        finish[thread_id] = 0;

        const int grid_size = N / omp_get_num_threads();
        int i_lower = (grid_size * (thread_id - 1)) + 1;
        int i_upper = grid_size * thread_id;
        // TODO Fix rounding errors
        
#pragma omp barrier
        
        for(int k = 1; k <= max_iter; k++){

            //printf("%i waiting for first guard on %i'th iteration finish[%i] \n", thread_id, k, finish[thread_id-1]);
            while(finish[thread_id - 1] < k) { // Guard 1
                #pragma omp flush(finish)
            }
            //printf("%i passed first guard on %i'th iteration\n", thread_id, k);


#pragma omp for schedule(static) nowait
            for(int i = 1; i <= N; i++) {

                if(i == i_lower) {
                    do_work(img, delta2, delta, N, i); // TODO Check if this is inlined
                    start[thread_id] = k;
                    #pragma omp flush(start)
                }

                else if(i == i_upper) {

                   // printf("%i waiting on second guard on %i'th iteration start[%i] \n", thread_id, k, start[thread_id+1]);
                    while(start[thread_id + 1] < k-1) { // Guard 2
                     #pragma omp flush(start)
                    }
                    //printf("%i passed second guard on %i'th iteration\n", thread_id, k);

                    do_work(img, delta2, delta ,N, i);
                    finish[thread_id] = k;
                    #pragma omp flush(finish)
                }

                else {
                    do_work(img, delta2, delta, N, i);
                    //printf("%.1f ",f(i,j,N+2,delta)*delta2);
                    }
                } /* for j */
            } /* parallel for i */
            
			//printf("\n");
		} /* for k*/
	return;
	}

/*				img[i*(N+2)+j] = 0.25*(   img[(i-1)*(N+2)+ j   ]        \
                + img[(i+1)*(N+2)+ j   ]                                \
                + img[ i   *(N+2)+(j-1)]                                \
                + img[ i   *(N+2)+(j+1)]                                \
                + delat2*f( i, j, N+2, delta ));
*/


void poisson(int n, double *grid, double th, int kmax, int choice) {

	double *grid_old = (double *)malloc( (n+2) * (n+2) * sizeof(double));
	init_array(grid_old, 10, n);

	/* double d = 1.0; */
	/* while (d > th) { */

/*    double *tmp = grid_old;
    grid_old = grid;
    grid = tmp;	*/	

    if (choice == 0) {
//        printf("Poisson calculating. Using the jacobi method.");
        jacobi(grid, grid_old, n, kmax, th);
    }
    else if (choice == 1) {
//        printf("Poisson calculating. Using the gauss method.");
        gauss(grid, grid_old, n, kmax, th);
    }
    else if(choice == 2) {
//        printf("Poisson calculating. Using the jacobi_mp method.");
        jacobi_mp(grid, grid_old, n, kmax);
    }
    else {
//        printf("Poisson calculating. Using the gauss_mp method.");
        gauss_mp(grid, n, kmax);
    }
        
		/* d = threshold(grid, grid_old, n); */
		/* printf("d %f\n", d); */

	/* } */
}

