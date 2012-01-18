// This program takes the dot product between two vectors of size N.

// Included C libraries
#include <stdio.h>
#include <vector_types.h>

// Macro for determining minimum
#ifndef MIN
#define MIN(a,b) ((a)>(b)?(b):(a))
#endif
#ifndef MAX
#define MAX(a,b) ((a)>(b)?(a):(b))
#endif

// Included CUDA libraries
#include <cutil.h> 

// Included files
#include "DotProd_Gold.c"
#include "DotProd_kernel.cu"


__constant__ float* Vec1_d; 
__constant__ float* Vec2_d;

int main(int argc, char* argv[]) {
  // Screen output
  printf("DotProd\n");
  printf("Computation of the dot product between two vectors of size N.\n");
  printf("  ./DotProd <N:default=1000> <THREADS_PR_BLOCK:default=MaxOnDevice>\n\n");

  // Check limitation of available device
  int dev = 0; // assumed only one device
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  printf("Device 0: Maximum number of threads per block is %d.\n", deviceProp.maxThreadsPerBlock);
  int MAX_THREADS_PR_BLOCK = deviceProp.maxThreadsPerBlock;
  int MAX_BLOCKS = deviceProp.maxGridSize[0];

  int N, THREADS_PR_BLOCK;
  if (argc>1 ? N = atoi(argv[1]) : N = 1000);
  if (argc>2 ? THREADS_PR_BLOCK = atoi(argv[2]) : THREADS_PR_BLOCK = MAX_THREADS_PR_BLOCK);
  if (THREADS_PR_BLOCK > MAX_THREADS_PR_BLOCK ? THREADS_PR_BLOCK = MAX_THREADS_PR_BLOCK : 0.0);
  printf("N: %d\n", N); 
  printf("Threads per block = %d. \n",THREADS_PR_BLOCK);

  int BLOCKS = (N + THREADS_PR_BLOCK)/THREADS_PR_BLOCK; 
  BLOCKS = MIN(BLOCKS,MAX_BLOCKS);
  printf("Blocks allocated = %d\n\n",BLOCKS);

/**************************************************
 * Create timers                                  *
 **************************************************/
    unsigned timer_cpu;
    unsigned timer_gpu_computation;
	unsigned timer_gpu_transfer;
    CUT_SAFE_CALL(cutCreateTimer(&timer_cpu));
    CUT_SAFE_CALL(cutCreateTimer(&timer_gpu_computation));
	CUT_SAFE_CALL(cutCreateTimer(&timer_gpu_transfer));

/****************************
 * Initialization of memory *
 ****************************/

  // Pointers to CPU (host) data
  float* Vec1_h = new float[N]; 
  float* Vec2_h = new float[N]; 
  float* cpuResult; 

  // initialize the two vectors
  for (int i = 0; i < N; ++i) {
     Vec1_h[i] = 2.0;
     Vec2_h[i] = (float)(i % 10); 
  }

  // Pointers for GPU (device) data 
  float* gpuResult; 

 // YOUR TASKS:
 // - Allocate below device arrays Vec1_d, Vec2_d and gpuResult
 // Insert code below this line.
  int size = N*sizeof(float);

	CUDA_SAFE_CALL(cudaMalloc((void **)&Vec1_d, size));
	CUDA_SAFE_CALL(cudaMalloc((void **)&Vec2_d, size));
	CUDA_SAFE_CALL(cudaMalloc((void **)&gpuResult, size));

  // Uncomment below and safely allocate memory for data on device
	
  // allocate an array to download the results of all threads 
  cpuResult = new float[ BLOCKS * THREADS_PR_BLOCK ]; 

/***************************
 * CPU execution           *
 ***************************/

  float finalDotProduct = 0.0;
  float* pfinalDotProduct = &finalDotProduct;

  CUT_SAFE_CALL(cutStartTimer(timer_cpu));

  for (int iter = 0; iter < 100; ++iter) 
  {
    finalDotProduct = 0.0; 
    DotProd_Gold(Vec1_h,Vec2_h,pfinalDotProduct,N);
  }

  CUT_SAFE_CALL(cutStopTimer(timer_cpu));

/***************************
 * GPU execution           *
 ***************************/

  // Split problem into threads
  dim3 blockGrid( BLOCKS ); 
  dim3 threadBlock( THREADS_PR_BLOCK ); 

  float finalDotProductGPU = 0.f;

  

  for (int iter = 0; iter < 100; ++iter) 
  {

 // YOUR TASKS:
 // - Transfer arrays Vec1_h and Vec2_h to device
 // Insert code below this line.
	CUT_SAFE_CALL(cutStartTimer(timer_gpu_transfer));
    // Uncomment below and copy vectors from host to device
    CUDA_SAFE_CALL(cudaMemcpy(Vec1_d,Vec1_h,size,cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(Vec2_d,Vec2_h,size,cudaMemcpyHostToDevice));
	
	CUT_SAFE_CALL(cutStopTimer(timer_gpu_transfer));



 // YOUR TASKS:
 // - Define Kernel execution configuration
 // Insert code below this line.
	CUT_SAFE_CALL(cutStartTimer(timer_gpu_computation));
    // Uncomment below and invoce kernel
    DotProd_kernel<<< blockGrid, threadBlock >>>(gpuResult, Vec1_d, Vec2_d, N); 

    // Error check
    CUT_CHECK_ERROR("Dot product kernel execution failed\n");
    CUDA_SAFE_CALL( cudaThreadSynchronize() );	

 // YOUR TASKS:
 // - Transfer array gpuResult from device to host cpuResult
 // Insert code below this line.

    // Uncomment below and copy results from device to host 
    CUDA_SAFE_CALL(cudaMemcpy(cpuResult,gpuResult,size,cudaMemcpyDeviceToHost));

    // Use host to condense results to a scalar
    finalDotProductGPU = 0.; 
    for (int i = 0; i < MIN(N,BLOCKS*THREADS_PR_BLOCK); ++i)
	 finalDotProductGPU += cpuResult[i]; 

	CUT_SAFE_CALL(cutStopTimer(timer_gpu_computation));
  }

  

/***************************
 * Output timings          *
 ***************************/
	double timer_gpu_total = cutGetTimerValue(timer_gpu_transfer) + cutGetTimerValue(timer_gpu_computation);	

    printf("  CPU time               : %.4f (ms)\n",cutGetTimerValue(timer_cpu));
    printf("  GPU time transfer      : %.4f (ms)\n\n",cutGetTimerValue(timer_gpu_transfer));
	printf("  GPU time computation   : %.4f (ms)\n\n",cutGetTimerValue(timer_gpu_computation));
	printf("  GPU time total         : %.4f (ms) , speedup %.2fx\n\n",timer_gpu_total,cutGetTimerValue(timer_cpu)/timer_gpu_total);

/***************************
 * Verification            *
 ***************************/

if (abs(finalDotProduct - finalDotProductGPU)<1e-4 ? printf("PASSED!\n") : printf("FAILED \n")  )

/***************************
 * Cleaning memory         *
 ***************************/

 // YOUR TASKS:
 // - Free allocated device memory
 // Insert code below this line.
  
  // Uncomment below and cleanup device memory
  CUDA_SAFE_CALL( cudaFree( Vec1_d ) ); 
  CUDA_SAFE_CALL( cudaFree( Vec2_d) ); 
  CUDA_SAFE_CALL( cudaFree( gpuResult ) ); 

  delete[] cpuResult; 
  delete[] Vec1_h; 
  delete[] Vec2_h; 

}








