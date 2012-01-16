#include <stdio.h>
#include "VecAdd_kernel.cu"

int main(int argc, char *argv[])
{

 int N = 100;

 unsigned int size;
 float *d_A, *d_B, *d_C;
 float *h_A, *h_B, *h_C;

/****************************
 * Initialization of memory *
 ****************************/

 size = N * sizeof(float);
 h_A = (float *) malloc(size);
 h_B = (float *) malloc(size);
 h_C = (float *) malloc(size);
 for (unsigned i=0; i<N; i++) {
   h_A[i] = 1.0f;
   h_B[i] = 2.0f;
   h_C[i] = 0.0f;
 }

 // YOUR TASKS:
 // - Allocate below device arrays d_A, d_B and d_C
 // - Transfer array data from host to device arrays
 // Insert code below this line.
 cudaMalloc((void **)&d_A, size);
 cudaMalloc((void **)&d_B, size);
 cudaMalloc((void **)&d_C, size);

 cudaMemcpy(d_A,h_A,size,cudaMemcpyHostToDevice);
 cudaMemcpy(d_B,h_B,size,cudaMemcpyHostToDevice);
 

/****************************
 * GPU execution            *
 ****************************/

 // YOUR TASK:
 // - Define below the number of threads per block and blocks per grid
 // Update the two lines below this line.

 int threadsPerBlock = 16; 
 int blocksPerGrid = 8; 

 VecAdd_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A,d_B,d_C,N);
 cudaThreadSynchronize();

 // YOUR TASK:
 // - Transfer data results stored in d_C to host array
 // Insert code below this line.

 cudaMemcpy(h_C,d_C,size,cudaMemcpyDeviceToHost);

/****************************
 * Verification             *
 ****************************/

 float sum = 0.0f;
 for (unsigned i=0; i<N; i++) {
    sum += h_C[i];
 }
 printf("Vector addition\n");
 if (abs(sum-3.0f*(float) N)<=1e-10) 
 {
    printf("PASSED!\n");
 }
 else
 {
    printf("FAILED!\n");
 }

/****************************
 * Cleaning memory          *
 ****************************/

 // YOUR TASK:
 // - Free device memory for the allocated d_A, d_B and d_C arrays
 // Insert code below this line.

 cudaFree(d_A);
 cudaFree(d_B);
 cudaFree(d_C);

 free(h_A);
 free(h_B);
 free(h_C);

 return 0;

}