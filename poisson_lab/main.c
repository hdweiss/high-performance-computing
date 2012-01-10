// gaussseidel.cpp : Defines the entry point for the console application.
//
#include <stdio.h>
#include <stdlib.h>
#include "writepng.h"
#include "gauss.h"

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

int main(int argc, char *argv[])
{
    int N=1000;
    int	max_iter;
    double *image;

    max_iter = 400;

    // command line argument sets the dimensions of the image
    if ( argc == 2 ) N = atoi(argv[1]);

    image = (double *)malloc( (N+2) * (N+2) * sizeof(double));
    if ( image == NULL ) {
       fprintf(stderr, "memory allocation failed!\n");
       return(1);
    }

	init_array(image,10,N);

    gauss(N, image, max_iter);

    writepng("poisson.png", image, N, N);

	return 0;
}



