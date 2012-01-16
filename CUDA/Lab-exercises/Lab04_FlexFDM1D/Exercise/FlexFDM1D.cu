//
// Written by Allan Engsig-Karup, October 20, 2010.
//

// Included C libraries
#include <stdio.h>
#include <ctime>
#include <math.h>
#include <assert.h>

// Included CUDA libraries
#include <cutil_inline.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char* msg);

// Included C subroutines
#include "fdcoeffF.c"

// Macro defining maximum stencil array size
#define STENCILARRAYSIZE_MAX 81

// number of times timed calculation is done for averaging
#define ITERATIONS 100

// Global scope
__constant__ float weightsconstant[STENCILARRAYSIZE_MAX]; // FIX linear memory size and choose maximum size here (FIXME: define maximum size somewhere else!!!)

// Included CUDA C subroutines
#include "FlexFDM1D_kernel.cu"
#include "FlexFDM1D_Gold.c"

// External routine for computing finite difference stencils
extern void fdcoeffF(int k, float xbar, float x[], float c[], int n); // external function prototype

int main(int argc, char *argv[])
{
    // Screen output
    printf("FlexFDM1D\n");
    printf("Approximation of first derivative in 1D using the finite difference method.\n");
    printf("  ./FlexFDM1D <Nx:default=1000> <alpha:default=3> <THREADS_PR_BLOCK:default=MaxOnDevice>\n\n");

    // Check limitation of available device
    int dev = 0; // assumed only one device
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Device 0: Maximum number of threads per block is %d.\n", deviceProp.maxThreadsPerBlock);
    int MAX_THREADS_PR_BLOCK = deviceProp.maxThreadsPerBlock;
    int MAX_BLOCKS = deviceProp.maxGridSize[0];

    // Assume default parameters or define based on input argument list
    int Nx, alpha, THREADS_PR_BLOCK;
    if (argc>1 ? Nx = strtol(argv[1],NULL,10) : Nx = 1000);
    printf("Number of points in x-direction, Nx = %d. \n",Nx);
    if (argc>2 ? alpha = strtol(argv[2],NULL,10) : alpha = 3);
    printf("Halfwidth of finite difference stencil, alpha = %d. \n",alpha);
    if (argc>3 ? THREADS_PR_BLOCK = strtol(argv[3],NULL,10) : THREADS_PR_BLOCK = MAX_THREADS_PR_BLOCK);
    printf("Threads per block = %d. \n\n",THREADS_PR_BLOCK);

    // Compute useful flops to be computed
    int flops = Nx*2*(2*alpha+1);

    /**************************************************
     * Create timers
     **************************************************/
    unsigned timer_cpu;
    unsigned timer_gpu1;
    unsigned timer_gpu2;
    unsigned timer_gpu3;
    unsigned timer_mem1;
    unsigned timer_mem2;
    unsigned timer_mem3;

    CUT_SAFE_CALL(cutCreateTimer(&timer_cpu));
    CUT_SAFE_CALL(cutCreateTimer(&timer_gpu1));
    CUT_SAFE_CALL(cutCreateTimer(&timer_gpu2));
    CUT_SAFE_CALL(cutCreateTimer(&timer_gpu3));
    CUT_SAFE_CALL(cutCreateTimer(&timer_mem1));
    CUT_SAFE_CALL(cutCreateTimer(&timer_mem2));
    CUT_SAFE_CALL(cutCreateTimer(&timer_mem3));

    /**************************************************
     * Pre-processing
     **************************************************/

    // define order of derivative
    int q = 1;

    // define total stencil size
    int rank = 2*alpha+1;

    // Setup mesh
    float* grid = (float*) malloc(sizeof(float)*Nx);
    for (int i=0; i<Nx; ++i)
	   grid[i] = (float) (i) / (float) (Nx-1);

    // Setup finite difference weights table
    float xbar;
    float* weights_h = (float*) malloc(sizeof(float)*STENCILARRAYSIZE_MAX); // stencil 2d-array
    float* c = (float*) malloc(sizeof(float)*rank);
    float* x = (float*) malloc(sizeof(float)*rank);

    // local grid for use in computing stencils
    for (int i=0; i<rank; ++i)
	   x[i] = grid[i];

    for (int i=0; i<rank; ++i) {
	   xbar = grid[i];
	   fdcoeffF(q,xbar,x,c,rank);
	   for (int j=0; j<rank; ++j)
	      weights_h[i*rank+j] = c[j]; // array storage according to row major order
	}

    /**************************************************
     * Host execution
     **************************************************/

    printf("Average timings per kernel invocation: \n\n");

    // Function and derivative definitions
    float* U_h   = (float*) malloc(sizeof(float)*Nx);
    float* Ux_h  = (float*) malloc(sizeof(float)*Nx);
    float* UxCPU = (float*) malloc(sizeof(float)*Nx);

    // Initialize input array
    for (int i=0; i<Nx; ++i)
    {
        U_h[i] = 0.5*grid[i]*grid[i]; // u(x) = 0.5*x^2
        Ux_h[i] = 0.0;
        UxCPU[i] = 0.0;
    }

    CUT_SAFE_CALL(cutStartTimer(timer_cpu));

    for (unsigned i = 0; i < ITERATIONS; i++) {
        FlexFDM1D_Gold(U_h, Ux_h, Nx, alpha, weights_h, rank); 
    }
    
    // Transfer data to CPU array to be used for comparisons with GPU results
    for (int i=0; i<Nx; ++i)
	    UxCPU[i] = Ux_h[i];
    
    CUT_SAFE_CALL(cutStopTimer(timer_cpu));

    // output a few of the calculated gradients...
//    printf("A few gradients computed by the CPU version\n");
//    for (int i=Nx-10; i<Nx; ++i)
//       printf("x[%d]=%f,   Ux[%d]=%f   \n",i,grid[i],i,Ux_h[i]);
//    printf("\n");

    /**************************************************
     * Allocate GPU memory
     **************************************************/

    float *stencils_d;
    float *U_d;
    float *Ux_d;

    // Transfer stencil weights to device memory
    int size;
    size = (rank*rank)*sizeof(float);
    CUDA_SAFE_CALL(cudaMalloc((void**) &stencils_d, size));

    // Allocate memory for result on device
    size = Nx*sizeof(float);
    CUDA_SAFE_CALL(cudaMalloc((void**) &U_d, size));
    CUDA_SAFE_CALL(cudaMalloc((void**) &Ux_d, size));
      
    /**************************************************
     * Device execution
     **************************************************/

    /**************************************************
     * GPU execution v1
     **************************************************/

    // make sure that we have at least one block 
    int blocksPerGrid = (Nx + THREADS_PR_BLOCK)/THREADS_PR_BLOCK; 
    blocksPerGrid = min(blocksPerGrid,MAX_BLOCKS);

//    printf("blocksPerGrid=%d \n",blocksPerGrid);

    CUT_SAFE_CALL(cutStartTimer(timer_mem1));
       size = (rank*rank)*sizeof(float);
       CUDA_SAFE_CALL(cudaMemcpy(stencils_d,weights_h,size,cudaMemcpyHostToDevice));
       size = Nx*sizeof(float);
       CUDA_SAFE_CALL(cudaMemcpy(U_d,U_h,size,cudaMemcpyHostToDevice));
    CUT_SAFE_CALL(cutStopTimer(timer_mem1));

    CUT_SAFE_CALL(cutStartTimer(timer_gpu1));
    for (unsigned i = 0; i < ITERATIONS; i++) {
	   FlexFDM1D_naive<<< blocksPerGrid, THREADS_PR_BLOCK>>>(U_d, Ux_d, Nx, alpha, stencils_d);
    }
    cudaThreadSynchronize();

    // check if kernel execution generated an error
    // Check for any CUDA errors
    checkCUDAError("kernel invocation v1");
    
    CUT_SAFE_CALL(cutStopTimer(timer_gpu1));

    CUT_SAFE_CALL(cutStartTimer(timer_mem1));
       size = Nx*sizeof(float);
       CUDA_SAFE_CALL(cudaMemcpy(Ux_h,Ux_d,size,cudaMemcpyDeviceToHost));
    CUT_SAFE_CALL(cutStopTimer(timer_mem1));

    // output a few of the calculated gradients...
//    printf("A few gradients computed by the GPU version 1\n");
//    for (int i=Nx-10; i<Nx; ++i)
//       printf("x[%d]=%f,   Ux[%d]=%f   \n",i,grid[i],i,Ux_h[i]);
//    printf("\n");

    // Verification
    float sum1 = 0.0f;
    for (int n=0; n<Nx; ++n)
    	sum1 = max(sum1,abs(Ux_h[n]-UxCPU[n]));

	// Reset device vector
	CUDA_SAFE_CALL(cudaMemset(Ux_d, 0, size));

    /**************************************************
     * GPU execution v2
     **************************************************/

    // make sure that we have at least one block 
    blocksPerGrid = (Nx + THREADS_PR_BLOCK - 2*alpha)/(THREADS_PR_BLOCK - 2*alpha); 
    blocksPerGrid = min(blocksPerGrid,MAX_BLOCKS);

//    printf("blocksPerGrid=%d \n",blocksPerGrid);

    CUT_SAFE_CALL(cutStartTimer(timer_mem2));
       size = (rank*rank)*sizeof(float);
       CUDA_SAFE_CALL(cudaMemcpy(stencils_d,weights_h,size,cudaMemcpyHostToDevice));
       size = Nx*sizeof(float);
       CUDA_SAFE_CALL(cudaMemcpy(U_d,U_h,size,cudaMemcpyHostToDevice));
    CUT_SAFE_CALL(cutStopTimer(timer_mem2));

    CUT_SAFE_CALL(cutStartTimer(timer_gpu2));
printf("THREADS_PR_BLOCK = %d\n",THREADS_PR_BLOCK);
    size = THREADS_PR_BLOCK*sizeof(float)*2; // FIXME: factor of two too much
    for (unsigned i = 0; i < ITERATIONS; i++) {
	   FlexFDM1D_v2<<< blocksPerGrid, THREADS_PR_BLOCK, size>>>(U_d, Ux_d, Nx, alpha, stencils_d);
    }
    cudaThreadSynchronize();

    // check if kernel execution generated an error
    // Check for any CUDA errors
    checkCUDAError("kernel invocation v2");
    
    CUT_SAFE_CALL(cutStopTimer(timer_gpu2));

    CUT_SAFE_CALL(cutStartTimer(timer_mem2));
       size = Nx*sizeof(float);
       CUDA_SAFE_CALL(cudaMemcpy(Ux_h,Ux_d,size,cudaMemcpyDeviceToHost));
    CUT_SAFE_CALL(cutStopTimer(timer_mem2));

    // output a few of the calculated gradients...
//    printf("A few gradients computed by the GPU version 2\n");
//    for (int i=0; i<Nx; ++i)
//       printf("x[%d]=%f,   Ux[%d]=%f   \n",i,grid[i],i,Ux_h[i]);
//    printf("\n");

    // Verification
    float sum2 = 0.0f;
    for (int n=0; n<Nx; ++n)
    	sum2 = max(sum2,abs(Ux_h[n]-UxCPU[n]));

	// Reset device vector
	CUDA_SAFE_CALL(cudaMemset(Ux_d, 0, size));

    /**************************************************
     * GPU execution v3
     **************************************************/

    // make sure that we have at least one block 
    blocksPerGrid = (Nx + THREADS_PR_BLOCK - 2*alpha)/(THREADS_PR_BLOCK - 2*alpha); 
    blocksPerGrid = min(blocksPerGrid,MAX_BLOCKS);

//    printf("blocksPerGrid=%d \n",blocksPerGrid);

    CUT_SAFE_CALL(cutStartTimer(timer_mem3));
       size = (rank*rank)*sizeof(float);
       CUDA_SAFE_CALL(cudaMemcpy(stencils_d,weights_h,size,cudaMemcpyHostToDevice));
       size = Nx*sizeof(float);
       CUDA_SAFE_CALL(cudaMemcpy(U_d,U_h,size,cudaMemcpyHostToDevice));
       size = (rank*rank)*sizeof(float);
       CUDA_SAFE_CALL(cudaMemcpyToSymbol(weightsconstant,weights_h,size,0)); // block data transfer
    CUT_SAFE_CALL(cutStopTimer(timer_mem3));

    CUT_SAFE_CALL(cutStartTimer(timer_gpu3));
    size = THREADS_PR_BLOCK*sizeof(float);
    for (unsigned i = 0; i < ITERATIONS; i++) {
	   FlexFDM1D_v3<<< blocksPerGrid, THREADS_PR_BLOCK, size>>>(U_d, Ux_d, Nx, alpha, stencils_d);
    }
    cudaThreadSynchronize();
 
    // check if kernel execution generated an error
    // Check for any CUDA errors
    checkCUDAError("kernel invocation v3");
    
    CUT_SAFE_CALL(cutStopTimer(timer_gpu3));

    CUT_SAFE_CALL(cutStartTimer(timer_mem3));
       size = Nx*sizeof(float);
       CUDA_SAFE_CALL(cudaMemcpy(Ux_h,Ux_d,size,cudaMemcpyDeviceToHost));
    CUT_SAFE_CALL(cutStopTimer(timer_mem3));

    // output a few of the calculated gradients...
//    printf("A few gradients computed by the GPU version 3\n");
//    for (int i=0; i<Nx; ++i)
//       printf("x[%d]=%f,   Ux[%d]=%f   \n",i,grid[i],i,Ux_h[i]);
//    printf("\n");

    // Verification
    float sum3 = 0.0f;
    for (int n=0; n<Nx; ++n)
    	sum3 = max(sum3,abs(Ux_h[n]-UxCPU[n]));

	// Reset device vector
	CUDA_SAFE_CALL(cudaMemset(Ux_d, 0, size));

    /**************************************************
     * Print timing results
     **************************************************/

    printf("  CPU time           : %.4f (ms)\n",
            cutGetTimerValue(timer_cpu)/ITERATIONS);
    printf("  CPU flops          : %.4f (Gflops) \n\n",flops/(cutGetTimerValue(timer_cpu)/ITERATIONS*(float)(1 << 30)));

    printf("  GPU v1 time compute: %.4f (ms)\n",
            cutGetTimerValue(timer_gpu1)/ITERATIONS);
    printf("  GPU v1 time memory : %.4f (ms)\n",
            cutGetTimerValue(timer_mem1));
    printf("  GPU v1 time total  : %.4f (ms): speedup %.2fx\n",
            cutGetTimerValue(timer_gpu1)/ITERATIONS + cutGetTimerValue(timer_mem1),
            cutGetTimerValue(timer_cpu)/ITERATIONS/(cutGetTimerValue(timer_gpu1)/ITERATIONS + cutGetTimerValue(timer_mem1)));
    printf("  GPU v1 flops       : %.4f (Gflops) \n",flops/(cutGetTimerValue(timer_gpu1)/ITERATIONS*(float)(1 << 30)));
    if (sum1<1e-2) {
    	printf("  PASSED\n\n");
    	} else {
    	printf("  FAILED %.4f \n\n",sum1);
    	}

    printf("  GPU v2 time compute: %.4f (ms)\n",
            cutGetTimerValue(timer_gpu2)/ITERATIONS);
    printf("  GPU v2 time memory : %.4f (ms)\n",
            cutGetTimerValue(timer_mem2));
    printf("  GPU v2 time total  : %.4f (ms): speedup %.2fx\n",
            cutGetTimerValue(timer_gpu2)/ITERATIONS + cutGetTimerValue(timer_mem2),
            cutGetTimerValue(timer_cpu)/ITERATIONS/(cutGetTimerValue(timer_gpu2)/ITERATIONS + cutGetTimerValue(timer_mem2)));
    printf("  GPU v2 flops       : %.4f (Gflops) \n",flops/(cutGetTimerValue(timer_gpu2)/ITERATIONS*(float)(1 << 30)));
    if (sum2<1e-2) {
    	printf("  PASSED\n\n");
    	} else {
    	printf("  FAILED %.4f \n\n",sum2);
       	}

    printf("  GPU v3 time compute: %.4f (ms)\n",
            cutGetTimerValue(timer_gpu3)/ITERATIONS);
    printf("  GPU v3 time memory : %.4f (ms)\n",
            cutGetTimerValue(timer_mem3));
    printf("  GPU v3 time total  : %.4f (ms): speedup %.2fx\n",
            cutGetTimerValue(timer_gpu3)/ITERATIONS + cutGetTimerValue(timer_mem3),
            cutGetTimerValue(timer_cpu)/ITERATIONS/(cutGetTimerValue(timer_gpu3)/ITERATIONS + cutGetTimerValue(timer_mem3)));
    printf("  GPU v3 flops       : %.4f (Gflops) \n",flops/(cutGetTimerValue(timer_gpu3)/ITERATIONS*(float)(1 << 30)));
    if (sum3<1e-2) {
    	printf("  PASSED\n\n");
    	} else {
    	printf("  FAILED %.4f \n\n",sum3);   	
    	}

    /**************************************************
     * Free data structures
     **************************************************/
    CUDA_SAFE_CALL(cudaFree(U_d));
    CUDA_SAFE_CALL(cudaFree(Ux_d));
    CUDA_SAFE_CALL(cudaFree(stencils_d));

    return 0;
}
void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, 
                                  cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }                         
}

