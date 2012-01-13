// gaussseidel.cpp : Defines the entry point for the console application.
//
#include <stdio.h>
#include <stdlib.h>
#include "writepng.h"
#include "poisson.h"



int main(int argc, char *argv[])
{
    int N=1000;
    int	max_iter = 1;
    double *image;
	int choice = 0;
	double threshold = -1.0;

    //max_iter = 400;

    // command line argument sets the dimensions of the image
    if ( argc >= 2 )
        N = atoi(argv[1]);
    
    if ( argc >= 3 )
		choice = atoi(argv[2]);

    if ( argc >= 4 )
        max_iter = atoi(argv[3]);

	if ( argc >= 5)
		threshold = atof(argv[4]);

    image = (double *)malloc( (N+2) * (N+2) * sizeof(double));
    if ( image == NULL ) {
       fprintf(stderr, "memory allocation failed!\n");
       return(1);
    }

	init_array(image,10,N);

	// 0: jacobi
	// 1: gauss
    poisson(N, image, threshold, max_iter, choice);

    writepng("poisson.png", image, N+2, N+2);

	return 0;
}



