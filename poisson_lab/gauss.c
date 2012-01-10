#include "poisson.h"

void gauss(int N, double* img, int max_iter){
	for(int k = 0; i < max_iter; k++){
		for(int i = 1; i <= N; i++){
			for(int j = 1; j <= N; j++){
				img[i*(N+2)+j] = 0.25*(   img[(i-1)*(N+2)+ j   ] \
										+ img[(i+1)*(N+2)+ j   ] \
										+ img[ i   *(N+2)+(j-1)] \
										+ img[ i   *(N+2)+(j+1)]+deltaf());
			}
		}
	}
	return;
}
