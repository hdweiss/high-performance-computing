// This program completes parallel multiplication of Matricies

// Included C libraries
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

// Included CUDA libraries
#include <cutil.h>

// Included files
#include "mm_kernel.cu"
#include "mm_gold.cpp"

int main( int argc, char* argv[]) 
{
	// Screen output
	printf("MM\n");
	printf("Parallel Matrix multiplication.\n");
	printf("  ./mm <N:default=64> <THREADS_PR_BLOCK:default=MaxOnDevice>\n\n");

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
	float* A_h = new float[N*N];
	float* B_h = new float[N*N]; 
	float* C_h = new float[N*N]; 

	srand(time(NULL));
	// initialize the Data Set
	for (int i = 0; i < N*N; ++i) {
		A_h[i] = (float)(rand()&(32767));
		B_h[i] = (float)(rand()&(32767));
		C_h[i] = 0.0;
	}

	// Pointers for GPU (device) data
	float* A_d; 
	float* B_d; 
	float* C_d; 

	// Safely allocate memory for data on device
	CUDA_SAFE_CALL( cudaMalloc( (void**)&A_d, N * N * sizeof(float) ) );   
	CUDA_SAFE_CALL( cudaMalloc( (void**)&B_d, N * N * sizeof(float) ) );   
	CUDA_SAFE_CALL( cudaMalloc( (void**)&C_d, N * N * sizeof(float) ) );   

	/***************************
	* CPU execution           *
	***************************/

	float* mmGold_h = new float[N*N];
	for (int i = 0; i < N*N; ++i) {
		mmGold_h[i] = 0.0;
	}

	CUT_SAFE_CALL(cutStartTimer(timer_cpu));

	for (int iter = 0; iter < 100; ++iter) 
	{
		mm_gold(A_h,B_h,mmGold_h,N);
	}

	CUT_SAFE_CALL(cutStopTimer(timer_cpu));

	/***************************
	* GPU execution (naive)   *
	***************************/

	// Split problem into threads
	dim3 blockGrid( BLOCKS ); 
	dim3 threadBlock( THREADS_PR_BLOCK ); 

	float* mm_h = new float[N*N];

	CUT_SAFE_CALL(cutStartTimer(timer_gpu));

	for (int iter = 0; iter < 100; ++iter) 
	{

		// Copy vectors from host to device
		CUDA_SAFE_CALL( cudaMemcpy( A_d, A_h, N*N * sizeof(float), cudaMemcpyHostToDevice) );
		CUDA_SAFE_CALL( cudaMemcpy( B_d, B_h, N*N * sizeof(float), cudaMemcpyHostToDevice) );

		// Kernel invocation
		mm_kernel<<< blockGrid, threadBlock, THREADS_PR_BLOCK*sizeof(float) >>>(C_d, A_d, B_d, N); 

		// Error check
		CUT_CHECK_ERROR("parallel reduction kernel execution failed\n");
		CUDA_SAFE_CALL( cudaThreadSynchronize() );	

		// Copy results from device to host 
		CUDA_SAFE_CALL( cudaMemcpy( mm_h, C_d, N*N*sizeof(float), cudaMemcpyDeviceToHost) );
		CUDA_SAFE_CALL( cudaThreadSynchronize() );	
		
	}

	CUT_SAFE_CALL(cutStopTimer(timer_gpu));
	CUDA_SAFE_CALL( cudaMemcpy( C_d, C_h, N*N*sizeof(float), cudaMemcpyHostToDevice) );

	/***************************
	* GPU execution (v2)      *
	***************************/

	float* mm2_h = new float[N*N];

	CUT_SAFE_CALL(cutStartTimer(timer_gpu2));

	for (int iter = 0; iter < 100; ++iter) 
	{

		// Copy vectors from host to device
		CUDA_SAFE_CALL( cudaMemcpy( A_d, A_h, N * N * sizeof(float), cudaMemcpyHostToDevice) );
		CUDA_SAFE_CALL( cudaMemcpy( B_d, B_h, N * N * sizeof(float), cudaMemcpyHostToDevice) );

		// Kernel invocation
		mm_kernel2<<< blockGrid, threadBlock, THREADS_PR_BLOCK*sizeof(float) >>>(C_d, A_d, B_d, N); 

		// Error check
		CUT_CHECK_ERROR("parallel reduction kernel execution failed\n");
		CUDA_SAFE_CALL( cudaThreadSynchronize() );	

		// Copy results from device to host 
		CUDA_SAFE_CALL( cudaMemcpy( mm2_h, C_d, N*N*sizeof(float), cudaMemcpyDeviceToHost) );
		CUDA_SAFE_CALL( cudaThreadSynchronize() );	
	}

	CUT_SAFE_CALL(cutStopTimer(timer_gpu2));
	CUDA_SAFE_CALL( cudaMemcpy( C_d, C_h, N*N*sizeof(float), cudaMemcpyHostToDevice) );

	/***************************
	* GPU execution (v3)      *
	***************************/

	float* mm3_h = new float[N*N];

	CUT_SAFE_CALL(cutStartTimer(timer_gpu3));

	for (int iter = 0; iter < 100; ++iter) 
	{

		// Copy vectors from host to device
		CUDA_SAFE_CALL( cudaMemcpy( A_d, A_h, N * N * sizeof(float), cudaMemcpyHostToDevice) );
		CUDA_SAFE_CALL( cudaMemcpy( B_d, B_h, N * N * sizeof(float), cudaMemcpyHostToDevice) );

		// Kernel invocation
		mm_kernel3<<< blockGrid, threadBlock, THREADS_PR_BLOCK*sizeof(float) >>>(C_d, A_d, B_d, N); 

		// Error check
		CUT_CHECK_ERROR("parallel reduction kernel execution failed\n");
		CUDA_SAFE_CALL( cudaThreadSynchronize() );	

		// Copy results from device to host 
		CUDA_SAFE_CALL( cudaMemcpy( mm3_h, C_d, N*N*sizeof(float), cudaMemcpyDeviceToHost) );
		CUDA_SAFE_CALL( cudaThreadSynchronize() );	

	}

	CUT_SAFE_CALL(cutStopTimer(timer_gpu3));
	CUDA_SAFE_CALL( cudaMemcpy( C_d, C_h, N*N*sizeof(float), cudaMemcpyHostToDevice) );

	/***************************
	* GPU execution (v4)      *
	***************************/

	float* mm4_h = new float[N*N];

	CUT_SAFE_CALL(cutStartTimer(timer_gpu4));

	for (int iter = 0; iter < 100; ++iter) 
	{

		// Copy vectors from host to device
		CUDA_SAFE_CALL( cudaMemcpy( A_d, A_h, N * N * sizeof(float), cudaMemcpyHostToDevice) );
		CUDA_SAFE_CALL( cudaMemcpy( B_d, B_h, N * N * sizeof(float), cudaMemcpyHostToDevice) );

		// Kernel invocation
		mm_kernel4<<< blockGrid, threadBlock, THREADS_PR_BLOCK*sizeof(float) >>>(C_d, A_d, B_d, N); 

		// Error check
		CUT_CHECK_ERROR("parallel reduction kernel execution failed\n");
		CUDA_SAFE_CALL( cudaThreadSynchronize() );	

		// Copy results from device to host 
		CUDA_SAFE_CALL( cudaMemcpy( mm4_h, C_d, N*N*sizeof(float), cudaMemcpyDeviceToHost) );
		CUDA_SAFE_CALL( cudaThreadSynchronize() );	
	}

	CUT_SAFE_CALL(cutStopTimer(timer_gpu4));
	CUDA_SAFE_CALL( cudaMemcpy( C_d, C_h, N*N*sizeof(float), cudaMemcpyHostToDevice) );

	/***************************
	* GPU execution (v5)      *
	***************************/

	float* mm5_h = new float[N*N];

	CUT_SAFE_CALL(cutStartTimer(timer_gpu5));

	for (int iter = 0; iter < 100; ++iter) 
	{

		// Copy vectors from host to device
		CUDA_SAFE_CALL( cudaMemcpy( A_d, A_h, N * N * sizeof(float), cudaMemcpyHostToDevice) );
		CUDA_SAFE_CALL( cudaMemcpy( B_d, B_h, N * N * sizeof(float), cudaMemcpyHostToDevice) );

		// Kernel invocation
		mm_kernel5<<< blockGrid, threadBlock, THREADS_PR_BLOCK*sizeof(float) >>>(C_d, A_d,B_d, N); 

		// Error check
		CUT_CHECK_ERROR("parallel reduction kernel execution failed\n");
		CUDA_SAFE_CALL( cudaThreadSynchronize() );	

		// Copy results from device to host 
		CUDA_SAFE_CALL( cudaMemcpy( mm5_h, C_d, N*N*sizeof(float), cudaMemcpyDeviceToHost) );
		CUDA_SAFE_CALL( cudaThreadSynchronize() );	
	}

	CUT_SAFE_CALL(cutStopTimer(timer_gpu5));
	CUDA_SAFE_CALL( cudaMemcpy( C_d, C_h, N*N*sizeof(float), cudaMemcpyHostToDevice) );


	/*********************************
	* Output timings & verification *
	*********************************/

	printf("  CPU time           : %.4f (ms)\n\n",cutGetTimerValue(timer_cpu));
	printf("  GPU time (naive)   : %.4f (ms) , speedup %.2fx\n",cutGetTimerValue(timer_gpu),cutGetTimerValue(timer_cpu)/cutGetTimerValue(timer_gpu));
	if (abs(mm_h[0] - mmGold_h[0])<1e-4 ? printf("  PASSED\n\n") : printf("  FAILED \n")  )
		printf("  GPU time (v2)      : %.4f (ms) , speedup %.2fx\n",cutGetTimerValue(timer_gpu2),cutGetTimerValue(timer_cpu)/cutGetTimerValue(timer_gpu2));
	if (abs(mm2_h[0] - mmGold_h[0])<1e-4 ? printf("  PASSED\n\n") : printf("  FAILED \n")  )
		printf("  GPU time (v3)      : %.4f (ms) , speedup %.2fx\n",cutGetTimerValue(timer_gpu3),cutGetTimerValue(timer_cpu)/cutGetTimerValue(timer_gpu3));
	if (abs(mm3_h[0] - mmGold_h[0])<1e-4 ? printf("  PASSED\n\n") : printf("  FAILED \n")  )
		printf("  GPU time (v4)      : %.4f (ms) , speedup %.2fx\n",cutGetTimerValue(timer_gpu4),cutGetTimerValue(timer_cpu)/cutGetTimerValue(timer_gpu4));
	if (abs(mm4_h[0] - mmGold_h[0])<1e-4 ? printf("  PASSED\n\n") : printf("  FAILED \n")  )
		printf("  GPU time (v5)      : %.4f (ms) , speedup %.2fx\n",cutGetTimerValue(timer_gpu5),cutGetTimerValue(timer_cpu)/cutGetTimerValue(timer_gpu5));
	if (abs(mm5_h[0] - mmGold_h[0])<1e-4 ? printf("  PASSED\n\n") : printf("  FAILED \n")  )

	/***************************
	* Verification            *
	***************************/

	printf("mmCPU         = %.2f\n",mmGold_h[0]);
	printf("mmGPU (naive) = %.2f\n",mm_h[0]);
	printf("mmGPU (v2)    = %.2f\n",mm2_h[0]);
	printf("mmGPU (v3)    = %.2f\n",mm3_h[0]);
	printf("mmGPU (v4)    = %.2f\n",mm4_h[0]);
	printf("mmGPU (v5)    = %.2f\n",mm5_h[0]);

	/***************************
	* Cleaning memory         *
	***************************/

	// cleanup device memory
	CUDA_SAFE_CALL( cudaFree(A_d) ); 
	CUDA_SAFE_CALL( cudaFree(B_d) ); 
	CUDA_SAFE_CALL( cudaFree(C_d) ); 

	delete[] C_h; 
	delete[] A_h; 
	delete[] B_h;

}
