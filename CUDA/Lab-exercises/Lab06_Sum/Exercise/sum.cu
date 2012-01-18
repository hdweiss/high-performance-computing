// This program completes parallel reduction on a data set.

// Included C libraries
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

// Included CUDA libraries
#include <cutil.h>

// Included files
#include "sum_kernel.cu"
#include "sum_gold.cpp"

int main( int argc, char* argv[]) 
{
  // Screen output
  printf("sum\n");
  printf("Parallel sum reduction.\n");
  printf("  ./sum <N:default=64> <THREADS_PR_BLOCK:default=MaxOnDevice>\n\n");

  // Check limitation of available device
  int dev = 0; // assumed only one device
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  printf("Device 0: Maximum number of threads per block is %d.\n", deviceProp.maxThreadsPerBlock);
  int MAX_THREADS_PR_BLOCK = deviceProp.maxThreadsPerBlock;

  int N, THREADS_PR_BLOCK;
  if (argc>1 ? N = atoi(argv[1]) : N = 64);
  if (argc>2 ? THREADS_PR_BLOCK = atoi(argv[2]) : THREADS_PR_BLOCK = MAX_THREADS_PR_BLOCK);
  if (THREADS_PR_BLOCK > MAX_THREADS_PR_BLOCK ? THREADS_PR_BLOCK = MAX_THREADS_PR_BLOCK : 0.0);
  if (THREADS_PR_BLOCK > N ? THREADS_PR_BLOCK = N : 0.0);
  printf("N: %d\n", N); 
  if (N % 32 > 0) { 
     printf("N is not a integer multiple of warpsize (32). %d \n",N%32);
     exit(1);
  }
  printf("Threads per block = %d. \n",THREADS_PR_BLOCK);

  int BLOCKS = (N + THREADS_PR_BLOCK)/THREADS_PR_BLOCK;
  if ((BLOCKS-1)*THREADS_PR_BLOCK >= N ? BLOCKS = BLOCKS-1 : 0.0 );
  printf("Blocks allocated = %d\n\n",BLOCKS);

/**************************************************
 * Create timers                                  *
 **************************************************/
    unsigned timer_cpu;
    unsigned timer_gpu;
    unsigned timer_gpu2;
    unsigned timer_gpu3;
    unsigned timer_gpu4;
    unsigned timer_gpu5;
    CUT_SAFE_CALL(cutCreateTimer(&timer_cpu));
    CUT_SAFE_CALL(cutCreateTimer(&timer_gpu));
    CUT_SAFE_CALL(cutCreateTimer(&timer_gpu2));
    CUT_SAFE_CALL(cutCreateTimer(&timer_gpu3));
    CUT_SAFE_CALL(cutCreateTimer(&timer_gpu4));
    CUT_SAFE_CALL(cutCreateTimer(&timer_gpu5));

/****************************
 * Initialization of memory *
 ****************************/

  // Pointers to CPU (host) data
  float* DataSet_h = new float[N]; 
  float* partialsums_h = new float[BLOCKS]; 

  // initialize the Data Set
  for (int i = 0; i < N; ++i) {
     DataSet_h[i] = 1.5f;
  }

  // Pointers for GPU (device) data
  float* DataSet_d; 
  float* partialsums_d; 

  // Safely allocate memory for data on device
  CUDA_SAFE_CALL( cudaMalloc( (void**)&DataSet_d, N * sizeof(float) ) );   
  CUDA_SAFE_CALL( cudaMalloc( (void**)&partialsums_d, BLOCKS*sizeof(float) ) );   

/***************************
 * CPU execution           *
 ***************************/

  float* sumGold_h = new float[1];
  sumGold_h[0] = 0.0f;

  CUT_SAFE_CALL(cutStartTimer(timer_cpu));

  for (int iter = 0; iter < 100; ++iter) 
  {
    sum_gold(sumGold_h,DataSet_h,N);
  }

  CUT_SAFE_CALL(cutStopTimer(timer_cpu));

/***************************
 * GPU execution (naive)   *
 ***************************/

  // Split problem into threads
  dim3 blockGrid( BLOCKS ); 
  dim3 threadBlock( THREADS_PR_BLOCK ); 

  float* sum_h = new float[1];

  CUT_SAFE_CALL(cutStartTimer(timer_gpu));

  for (int iter = 0; iter < 100; ++iter) 
  {

    // Copy vectors from host to device
    CUDA_SAFE_CALL( cudaMemcpy( DataSet_d, DataSet_h, N * sizeof(float), cudaMemcpyHostToDevice) );

    // Kernel invocation
    sum_kernel<<< blockGrid, threadBlock, THREADS_PR_BLOCK*sizeof(float) >>>(partialsums_d, DataSet_d, N); 

    // Error check
    CUT_CHECK_ERROR("parallel reduction kernel execution failed\n");
    CUDA_SAFE_CALL( cudaThreadSynchronize() );	

    // Copy results from device to host 
    CUDA_SAFE_CALL( cudaMemcpy( partialsums_h, partialsums_d, BLOCKS*sizeof(float), cudaMemcpyDeviceToHost) );
    CUDA_SAFE_CALL( cudaThreadSynchronize() );	

    // Use host to condense block results to a scalar
    sum_h[0] = 0.;  
    sum_gold(sum_h,partialsums_h,BLOCKS);
  }

  CUT_SAFE_CALL(cutStopTimer(timer_gpu));
	CUDA_SAFE_CALL( cudaFree(partialsums_d) );
	CUDA_SAFE_CALL( cudaMalloc( (void**)&partialsums_d, BLOCKS*sizeof(float) ) ); 

/***************************
 * GPU execution (v2)      *
 ***************************/

  float* sum2_h = new float[1];

  CUT_SAFE_CALL(cutStartTimer(timer_gpu2));

  for (int iter = 0; iter < 100; ++iter) 
  {

    // Copy vectors from host to device
    CUDA_SAFE_CALL( cudaMemcpy( DataSet_d, DataSet_h, N * sizeof(float), cudaMemcpyHostToDevice) );

    // Kernel invocation
    sum_kernel2<<< blockGrid, threadBlock, THREADS_PR_BLOCK*sizeof(float) >>>(partialsums_d, DataSet_d, N); 

    // Error check
    CUT_CHECK_ERROR("parallel reduction kernel execution failed\n");
    CUDA_SAFE_CALL( cudaThreadSynchronize() );	

    // Copy results from device to host 
    CUDA_SAFE_CALL( cudaMemcpy( partialsums_h, partialsums_d, BLOCKS*sizeof(float), cudaMemcpyDeviceToHost) );
    CUDA_SAFE_CALL( cudaThreadSynchronize() );	

    // Use host to condense block results to a scalar
    sum2_h[0] = 0.;  
    sum_gold(sum2_h,partialsums_h,BLOCKS);
  }

  CUT_SAFE_CALL(cutStopTimer(timer_gpu2));
  CUDA_SAFE_CALL( cudaFree(partialsums_d) );
  CUDA_SAFE_CALL( cudaMalloc( (void**)&partialsums_d, BLOCKS*sizeof(float) ) ); 
  for (int i = 0; i < BLOCKS; i++)
	  partialsums_h[i] = 0.0;

  CUDA_SAFE_CALL( cudaMemcpy( partialsums_d, partialsums_h, BLOCKS*sizeof(float), cudaMemcpyHostToDevice) );

/***************************
 * GPU execution (v3)      *
 ***************************/

  float* sum3_h = new float[1];

  CUT_SAFE_CALL(cutStartTimer(timer_gpu3));

  for (int iter = 0; iter < 100; ++iter) 
  {

    // Copy vectors from host to device
    CUDA_SAFE_CALL( cudaMemcpy( DataSet_d, DataSet_h, N * sizeof(float), cudaMemcpyHostToDevice) );

    // Kernel invocation
    sum_kernel3<<< blockGrid, threadBlock, THREADS_PR_BLOCK*sizeof(float) >>>(partialsums_d, DataSet_d, N); 

    // Error check
    CUT_CHECK_ERROR("parallel reduction kernel execution failed\n");
    CUDA_SAFE_CALL( cudaThreadSynchronize() );	

    // Copy results from device to host 
    CUDA_SAFE_CALL( cudaMemcpy( partialsums_h, partialsums_d, BLOCKS*sizeof(float), cudaMemcpyDeviceToHost) );
    CUDA_SAFE_CALL( cudaThreadSynchronize() );	

    // Use host to condense block results to a scalar
    sum3_h[0] = 0.;  
    sum_gold(sum3_h,partialsums_h,BLOCKS);
  }

  CUT_SAFE_CALL(cutStopTimer(timer_gpu3));

/***************************
 * GPU execution (v4)      *
 ***************************/

  float* sum4_h = new float[1];

  CUT_SAFE_CALL(cutStartTimer(timer_gpu4));

  for (int iter = 0; iter < 100; ++iter) 
  {

    // Copy vectors from host to device
    CUDA_SAFE_CALL( cudaMemcpy( DataSet_d, DataSet_h, N * sizeof(float), cudaMemcpyHostToDevice) );

    // Kernel invocation
    sum_kernel4<<< blockGrid, threadBlock, THREADS_PR_BLOCK*sizeof(float) >>>(partialsums_d, DataSet_d, N); 

    // Error check
    CUT_CHECK_ERROR("parallel reduction kernel execution failed\n");
    CUDA_SAFE_CALL( cudaThreadSynchronize() );	

    // Copy results from device to host 
    CUDA_SAFE_CALL( cudaMemcpy( partialsums_h, partialsums_d, BLOCKS*sizeof(float), cudaMemcpyDeviceToHost) );
    CUDA_SAFE_CALL( cudaThreadSynchronize() );	

    // Use host to condense block results to a scalar
    sum4_h[0] = 0.;  
    sum_gold(sum4_h,partialsums_h,BLOCKS);
  }

  CUT_SAFE_CALL(cutStopTimer(timer_gpu4));

/***************************
 * GPU execution (v5)      *
 ***************************/

  float* sum5_h = new float[1];

  CUT_SAFE_CALL(cutStartTimer(timer_gpu5));

  for (int iter = 0; iter < 100; ++iter) 
  {

    // Copy vectors from host to device
    CUDA_SAFE_CALL( cudaMemcpy( DataSet_d, DataSet_h, N * sizeof(float), cudaMemcpyHostToDevice) );

    // Kernel invocation
    sum_kernel5<<< blockGrid, threadBlock, THREADS_PR_BLOCK*sizeof(float) >>>(partialsums_d, DataSet_d, N); 

    // Error check
    CUT_CHECK_ERROR("parallel reduction kernel execution failed\n");
    CUDA_SAFE_CALL( cudaThreadSynchronize() );	

    // Copy results from device to host 
    CUDA_SAFE_CALL( cudaMemcpy( partialsums_h, partialsums_d, BLOCKS*sizeof(float), cudaMemcpyDeviceToHost) );
    CUDA_SAFE_CALL( cudaThreadSynchronize() );	

    // Use host to condense block results to a scalar
    sum5_h[0] = 0.;  
    sum_gold(sum5_h,partialsums_h,BLOCKS);
  }

  CUT_SAFE_CALL(cutStopTimer(timer_gpu5));


/*********************************
 * Output timings & verification *
 *********************************/

    printf("  CPU time           : %.4f (ms)\n\n",cutGetTimerValue(timer_cpu));
    printf("  GPU time (naive)   : %.4f (ms) , speedup %.2fx\n",cutGetTimerValue(timer_gpu),cutGetTimerValue(timer_cpu)/cutGetTimerValue(timer_gpu));
    if (abs(sum_h[0] - sumGold_h[0])<1e-4 ? printf("  PASSED\n\n") : printf("  FAILED \n")  )
    printf("  GPU time (v2)      : %.4f (ms) , speedup %.2fx\n",cutGetTimerValue(timer_gpu2),cutGetTimerValue(timer_cpu)/cutGetTimerValue(timer_gpu2));
    if (abs(sum2_h[0] - sumGold_h[0])<1e-4 ? printf("  PASSED\n\n") : printf("  FAILED \n")  )
    printf("  GPU time (v3)      : %.4f (ms) , speedup %.2fx\n",cutGetTimerValue(timer_gpu3),cutGetTimerValue(timer_cpu)/cutGetTimerValue(timer_gpu3));
    if (abs(sum3_h[0] - sumGold_h[0])<1e-4 ? printf("  PASSED\n\n") : printf("  FAILED \n")  )
    printf("  GPU time (v4)      : %.4f (ms) , speedup %.2fx\n",cutGetTimerValue(timer_gpu4),cutGetTimerValue(timer_cpu)/cutGetTimerValue(timer_gpu4));
    if (abs(sum4_h[0] - sumGold_h[0])<1e-4 ? printf("  PASSED\n\n") : printf("  FAILED \n")  )
    printf("  GPU time (v5)      : %.4f (ms) , speedup %.2fx\n",cutGetTimerValue(timer_gpu5),cutGetTimerValue(timer_cpu)/cutGetTimerValue(timer_gpu5));
    if (abs(sum5_h[0] - sumGold_h[0])<1e-4 ? printf("  PASSED\n\n") : printf("  FAILED \n")  )

/***************************
 * Verification            *
 ***************************/

  printf("sumCPU         = %.2f\n",sumGold_h[0]);
  printf("sumGPU (naive) = %.2f\n",sum_h[0]);
  printf("sumGPU (v2)    = %.2f\n",sum2_h[0]);
  printf("sumGPU (v3)    = %.2f\n",sum3_h[0]);
  printf("sumGPU (v4)    = %.2f\n",sum4_h[0]);
  printf("sumGPU (v5)    = %.2f\n",sum5_h[0]);

/***************************
 * Cleaning memory         *
 ***************************/

  // cleanup device memory
  CUDA_SAFE_CALL( cudaFree(partialsums_d) ); 
  CUDA_SAFE_CALL( cudaFree(DataSet_d) ); 

  delete[] partialsums_h; 
  delete[] DataSet_h; 

}
