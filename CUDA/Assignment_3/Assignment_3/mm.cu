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
#include <cublas.h>

// Included files
#include "mm_util.cu"
#include "mm_kernel.cu"
#include "mm_gold.cpp"

#define ITERATIONS 50


int main( int argc, char* argv[]) 
{
	int mm1_BLOCKS = 16;
	// Screen output
	printf("MM %i\n", mm1_BLOCKS);
	printf("Parallel Matrix multiplication.\n");

	// Check limitation of available device
	int dev = 0; // assumed only one device
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	printf("Device 0: Maximum number of threads per block is %d.\n", deviceProp.maxThreadsPerBlock);


	//	int MAX_THREADS_PR_BLOCK = deviceProp.maxThreadsPerBlock;
	//int N, THREADS_PR_BLOCK;
	//if (argc>1 ? N = atoi(argv[1]) : N = 64);
	//if (argc>2 ? THREADS_PR_BLOCK = atoi(argv[2]) : THREADS_PR_BLOCK = MAX_THREADS_PR_BLOCK);
	//if (THREADS_PR_BLOCK > MAX_THREADS_PR_BLOCK ? THREADS_PR_BLOCK = MAX_THREADS_PR_BLOCK : 0.0);
	//if (THREADS_PR_BLOCK > N ? THREADS_PR_BLOCK = N : 0.0);
	//printf("N: %d\n", N); 
	//if (N % 32 > 0) { 
	//	printf("N is not a integer multiple of warpsize (32). %d \n",N%32);
	//	exit(1);
	//}
	//printf("Threads per block = %d. \n",THREADS_PR_BLOCK);

	//int BLOCKS = (N + THREADS_PR_BLOCK)/THREADS_PR_BLOCK;
	//if ((BLOCKS-1)*THREADS_PR_BLOCK >= N ? BLOCKS = BLOCKS-1 : 0.0 );
	//printf("Blocks allocated = %d\n\n",BLOCKS);

	/**************************************************
	* Create timers                                  *
	**************************************************/
	unsigned timer_cpu;
	unsigned timer_gpu1;
	unsigned timer_gpu2;
	unsigned timer_gpu3;
	unsigned timer_gpu4;
	unsigned timer_gpu5;
	unsigned timer_gpu6;
	CUT_SAFE_CALL(cutCreateTimer(&timer_cpu));
	CUT_SAFE_CALL(cutCreateTimer(&timer_gpu1));
	CUT_SAFE_CALL(cutCreateTimer(&timer_gpu2));
	CUT_SAFE_CALL(cutCreateTimer(&timer_gpu3));
	CUT_SAFE_CALL(cutCreateTimer(&timer_gpu4));
	CUT_SAFE_CALL(cutCreateTimer(&timer_gpu5));
	CUT_SAFE_CALL(cutCreateTimer(&timer_gpu6));

	/****************************
	* Initialization of memory *
	****************************/

	int matrix_blocks = 16;
	int m = matrix_blocks*20;
	int k = matrix_blocks*40;
	int n = matrix_blocks*60;
	//int m, k, n;
	//n = m = k = 1280;

	// Pointers to CPU (host) data
	Matrix A_h;
	Matrix B_h;
	Matrix C_h;

	create_matrix(&A_h, m, k);
	create_matrix(&B_h, k, n);
	create_matrix(&C_h, m, n);

	srand((unsigned int)time(NULL));

	int A_size = A_h.width * A_h.height;
	for(int i = 0; i < A_size; i++)
		A_h.elements[i] = (float) rand() / RAND_MAX;

	int B_size = B_h.width * B_h.height;
	for(int i = 0; i < B_size; i++)
		B_h.elements[i] = (float) rand() / RAND_MAX;

	int C_size = C_h.width * C_h.height;
	for(int i = 0; i < C_size; i++)
		C_h.elements[i] = 0.0f;

	/***************************
	* CPU execution           *
	***************************/

	Matrix mmGold_h = clone_matrix(&C_h);
	for (int i = 0; i < C_size; ++i) {
		mmGold_h.elements[i] = 0.0f;
	}

	CUT_SAFE_CALL(cutStartTimer(timer_cpu));

	for (int iter = 0; iter < ITERATIONS; ++iter) 
	{
		mm_gold(m,n,k,A_h.elements,B_h.elements,mmGold_h.elements);
	}

	CUT_SAFE_CALL(cutStopTimer(timer_cpu));

	/***************************
	* GPU execution (naive)   *
	***************************/

	// Split problem into threads

	dim3 mm1_threadBlock( mm1_BLOCKS, mm1_BLOCKS );
	unsigned int blocky = (unsigned int) ceil(((float)m)/mm1_BLOCKS);
	unsigned int blockx = (unsigned int)ceil(((float)n)/mm1_BLOCKS);

	dim3 mm1_blockGrid(blockx, blocky);
	printf("Allocated grid (%u,%u)\n", blockx, blocky);

	//Matrix mm1_h = clone_matrix(&C_h);

	//CUT_SAFE_CALL(cutStartTimer(timer_gpu1));

	Matrix A_d = alloc_matrix_on_device( &A_h);
	Matrix B_d = alloc_matrix_on_device( &B_h);
	Matrix C_d = alloc_matrix_on_device(&C_h);

	//for (int iter = 0; iter < ITERATIONS; ++iter) 
	//{
	//	copy_matrix_to_device( &A_h, &A_d);
	//	copy_matrix_to_device( &B_h, &B_d);

	//	// Kernel invocation
	//	mm_kernel1<<< mm1_blockGrid, mm1_threadBlock>>>(C_d, A_d, B_d); 

	//	// Error check
	//	CUT_CHECK_ERROR("parallel reduction kernel execution failed\n");
	//	CUDA_SAFE_CALL( cudaThreadSynchronize() );	

	//	copy_matrix_from_device(&mm1_h, &C_d);

	//	CUDA_SAFE_CALL( cudaThreadSynchronize() );	

	//	
	//}

	//CUDA_SAFE_CALL( cudaFree(A_d.elements));
	//CUDA_SAFE_CALL( cudaFree(B_d.elements));
	//CUDA_SAFE_CALL( cudaFree(C_d.elements));

	//CUT_SAFE_CALL(cutStopTimer(timer_gpu1));

	/***************************
	* GPU execution (v2)      *
	***************************/

	//Matrix mm2_h = clone_matrix(&C_h);

	//CUT_SAFE_CALL(cutStartTimer(timer_gpu2));

	//A_d = alloc_matrix_on_device( &A_h);
	//B_d = alloc_matrix_on_device( &B_h);
	//C_d = alloc_matrix_on_device(&C_h);

	//for (int iter = 0; iter < ITERATIONS; ++iter) 
	//{
	//	copy_matrix_to_device( &A_h, &A_d);
	//	copy_matrix_to_device( &B_h, &B_d);

	//	// Kernel invocation
	//	mm_kernel2<<< mm1_blockGrid, mm1_threadBlock>>>(C_d, A_d, B_d); 

	//	// Error check
	//	CUT_CHECK_ERROR("parallel reduction kernel execution failed\n");
	//	CUDA_SAFE_CALL( cudaThreadSynchronize() );	

	//	copy_matrix_from_device(&mm2_h, &C_d);

	//	CUDA_SAFE_CALL( cudaThreadSynchronize() );	

	//}

	//CUDA_SAFE_CALL( cudaFree(A_d.elements));
	//CUDA_SAFE_CALL( cudaFree(B_d.elements));
	//CUDA_SAFE_CALL( cudaFree(C_d.elements));

	//CUT_SAFE_CALL(cutStopTimer(timer_gpu2));

	/***************************
	* GPU execution (v3)      *
	***************************/

	dim3 mm3_threadBlock( 16 , 1 );

	Matrix mm3_h = clone_matrix(&C_h);

	CUT_SAFE_CALL(cutStartTimer(timer_gpu3));

	A_d = alloc_matrix_on_device( &A_h);
	B_d = alloc_matrix_on_device( &B_h);
	C_d = alloc_matrix_on_device(&C_h);

	for (int iter = 0; iter < ITERATIONS; ++iter) 
	{
		copy_matrix_to_device( &A_h, &A_d);
		copy_matrix_to_device( &B_h, &B_d);

		// Kernel invocation
		mm_kernel3<<< mm1_blockGrid, mm3_threadBlock>>>(C_d, A_d, B_d); 

		// Error check
		CUT_CHECK_ERROR("parallel reduction kernel execution failed\n");
		CUDA_SAFE_CALL( cudaThreadSynchronize() );	

		copy_matrix_from_device(&mm3_h, &C_d);

		CUDA_SAFE_CALL( cudaThreadSynchronize() );	

	}

	CUDA_SAFE_CALL( cudaFree(A_d.elements));
	CUDA_SAFE_CALL( cudaFree(B_d.elements));
	CUDA_SAFE_CALL( cudaFree(C_d.elements));

	CUT_SAFE_CALL(cutStopTimer(timer_gpu3));

	/***************************
	* GPU execution (v4)      *
	***************************/

	//dim3 mm4_threadBlock( 64, 1 );
	//unsigned int blocky4 = (unsigned int) ceil(((float)m)/16);
	//unsigned int blockx4 = (unsigned int)ceil(((float)n)/64);

	//dim3 mm4_blockGrid(blockx4, blocky4);

	//Matrix mm4_h = clone_matrix(&C_h);

	//CUT_SAFE_CALL(cutStartTimer(timer_gpu4));

	//A_d = alloc_matrix_on_device( &A_h);
	//B_d = alloc_matrix_on_device( &B_h);
	//C_d = alloc_matrix_on_device(&C_h);

	//for (int iter = 0; iter < ITERATIONS; ++iter) 
	//{
	//	copy_matrix_to_device( &A_h, &A_d);
	//	copy_matrix_to_device( &B_h, &B_d);

	//	// Kernel invocation
	//	mm_kernel4<<< mm4_blockGrid, mm4_threadBlock>>>(C_d, A_d, B_d); 

	//	// Error check
	//	CUT_CHECK_ERROR("parallel reduction kernel execution failed\n");
	//	CUDA_SAFE_CALL( cudaThreadSynchronize() );	

	//	copy_matrix_from_device(&mm4_h, &C_d);

	//	CUDA_SAFE_CALL( cudaThreadSynchronize() );	
	//}

	//CUDA_SAFE_CALL( cudaFree(A_d.elements));
	//CUDA_SAFE_CALL( cudaFree(B_d.elements));
	//CUDA_SAFE_CALL( cudaFree(C_d.elements));

	//CUT_SAFE_CALL(cutStopTimer(timer_gpu4));

	/***************************
	* GPU execution (v5)      *
	***************************/

	//dim3 mm5_threadBlock( 64, 1 );
	//unsigned int blocky5 = (unsigned int) ceil(((float)m)/16);
	//unsigned int blockx5 = (unsigned int)ceil(((float)n)/64);

	//dim3 mm5_blockGrid(blockx5, blocky5);

	//Matrix mm5_h = clone_matrix(&C_h);

	//CUT_SAFE_CALL(cutStartTimer(timer_gpu5));

	//A_d = alloc_matrix_on_device( &A_h);
	//B_d = alloc_matrix_on_device( &B_h);
	//C_d = alloc_matrix_on_device(&C_h);

	//for (int iter = 0; iter < ITERATIONS; ++iter) 
	//{
	//	copy_matrix_to_device( &A_h, &A_d);
	//	copy_matrix_to_device( &B_h, &B_d);

	//	// Kernel invocation
	//	mm_kernel5<<< mm5_blockGrid, mm5_threadBlock>>>(C_d, A_d, B_d); 

	//	// Error check
	//	CUT_CHECK_ERROR("parallel reduction kernel execution failed\n");
	//	CUDA_SAFE_CALL( cudaThreadSynchronize() );	

	//	copy_matrix_from_device(&mm5_h, &C_d);

	//	CUDA_SAFE_CALL( cudaThreadSynchronize() );	

	//}

	//CUDA_SAFE_CALL( cudaFree(A_d.elements));
	//CUDA_SAFE_CALL( cudaFree(B_d.elements));
	//CUDA_SAFE_CALL( cudaFree(C_d.elements));

	//CUT_SAFE_CALL(cutStopTimer(timer_gpu5));

	/***************************
	* GPU execution (cublas)      *
	***************************/

	//Matrix mmcu_h = clone_matrix(&C_h);

	//CUT_SAFE_CALL(cutStartTimer(timer_gpu6));

	//A_d = alloc_matrix_on_device( &A_h);
	//B_d = alloc_matrix_on_device( &B_h);
	//C_d = alloc_matrix_on_device(&C_h);

	//for (int iter = 0; iter < ITERATIONS; ++iter) 
	//{
	//	copy_matrix_to_device( &A_h, &A_d);
	//	copy_matrix_to_device( &B_h, &B_d);

	//	float alpha = 1.0f;
	//	float beta = 0.0f;

	//	int lda = k; // 10. Parameter
	//	int ldb = n; // 8. Parameter

	//	int ldc = n; // 13. Parameter

	//	// Invocation of the 
	//	cublasSgemm('N', 'N',
	//		n, m, k,
	//		alpha,
	//		B_d.elements, ldb,
	//		A_d.elements, lda,
	//		beta,
	//		C_d.elements, ldc);

	//	// Error check
	//	CUT_CHECK_ERROR("parallel reduction kernel execution failed\n");
	//	CUDA_SAFE_CALL( cudaThreadSynchronize() );	

	//	copy_matrix_from_device(&mmcu_h, &C_d);

	//	CUDA_SAFE_CALL( cudaThreadSynchronize() );	

	//}

	//CUDA_SAFE_CALL( cudaFree(A_d.elements));
	//CUDA_SAFE_CALL( cudaFree(B_d.elements));
	//CUDA_SAFE_CALL( cudaFree(C_d.elements));

	//CUT_SAFE_CALL(cutStopTimer(timer_gpu6));

	/*********************************
	* Output timings & verification *
	*********************************/

	printf("  CPU time                : %.4f (ms)\n\n",cutGetTimerValue(timer_cpu));

//	print_matrix_result(&mm1_h,  "Naive       ", timer_gpu1, timer_cpu, &mmGold_h);
//	print_matrix_result(&mm2_h,  "Shared      ", timer_gpu2, timer_cpu, &mmGold_h);
	print_matrix_result(&mm3_h,  "4a          ", timer_gpu3, timer_cpu, &mmGold_h);
//	print_matrix_result(&mm4_h,  "4b          ", timer_gpu4, timer_cpu, &mmGold_h);
//	print_matrix_result(&mm5_h,  "Pimped      ", timer_gpu5, timer_cpu, &mmGold_h);
//	print_matrix_result(&mmcu_h, "cublasSgemm ", timer_gpu6, timer_cpu, &mmGold_h);

	/***************************
	* Cleaning memory         *
	***************************/

	free(C_h.elements); 
	free(A_h.elements); 
	free(B_h.elements); 

	//free(mm1_h.elements); 
	//free(mm2_h.elements); 
	free(mm3_h.elements); 
	//free(mm4_h.elements); 
	//free(mm5_h.elements); 
	//free(mmcu_h.elements); 
}
