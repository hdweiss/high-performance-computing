#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cutil.h>

#include "FDM3D_kernel.cu"

#define DIMX 32
#define DIMY 32
#define DIMZ 64

#define ITERATIONS 25

void initVolumes(float * a, float * b, unsigned dimX, unsigned dimY, unsigned dimZ)
{
    unsigned size = dimX * dimY * dimZ * sizeof(float);
    memset(a, 0, size);
    memset(b, 0, size);
    for (unsigned k = dimZ/2 - 2; k <= dimZ/2 + 2; k++) {
        for (unsigned j = dimY/2 - 2; j <= dimY/2 + 2; j++) {
            for (unsigned i = dimX/2 - 2; i <= dimX/2 + 2; i++) {
                unsigned idx = k * dimX * dimY + j * dimX + i;
                a[idx] = 100.f;
            }
        }
    }
}

int main(int argc, char *argv[])
{
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

    float * u1_d, * u2_d;
    float * u1_h, * u2_h;

    unsigned elems = (DIMX + 2 * RADIUS) *
                     (DIMY + 2 * RADIUS) *
                     (DIMZ + 2 * RADIUS);

    size_t size = elems * sizeof(float);

    u1_h = (float *) malloc(size);
    u2_h = (float *) malloc(size);

    /**************************************************
     * Host execution
     **************************************************/
    dim3 Db(BLOCK_DIM_X, BLOCK_DIM_Y); // define number of threads per block
    dim3 Dg(DIMX/BLOCK_DIM_X, DIMY/BLOCK_DIM_Y); // define number of blocks in the grid
    initVolumes(u1_h, u2_h, DIMX + 2 * RADIUS,
                            DIMY + 2 * RADIUS,
                            DIMZ + 2 * RADIUS);

    CUT_SAFE_CALL(cutStartTimer(timer_cpu));

    for (unsigned i = 0; i < ITERATIONS; i++) {
        kernelStencil_gold(u1_h, u2_h,
                           DIMX + 2 * RADIUS,
                           DIMY + 2 * RADIUS,
                           DIMZ + 2 * RADIUS);

        float * tmp = u2_h;
        u2_h = u1_h;
        u1_h = tmp;
    }
    
    CUT_SAFE_CALL(cutStopTimer(timer_cpu));

    /**************************************************
     * Allocate GPU memory
     **************************************************/
	CUDA_SAFE_CALL(cudaMalloc((void **) &u1_d, size));
	CUDA_SAFE_CALL(cudaMalloc((void **) &u2_d, size));

    /**************************************************
     * GPU execution v1
     **************************************************/
    initVolumes(u1_h, u2_h, DIMX + 2 * RADIUS,
                            DIMY + 2 * RADIUS,
                            DIMZ + 2 * RADIUS);


    CUT_SAFE_CALL(cutStartTimer(timer_mem1));
	CUDA_SAFE_CALL(cudaMemcpy(u1_d, u1_h, size, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(u2_d, u2_h, size, cudaMemcpyHostToDevice));
    CUT_SAFE_CALL(cutStopTimer(timer_mem1));

    CUT_SAFE_CALL(cutStartTimer(timer_gpu1));
    for (unsigned i = 0; i < ITERATIONS; i++) {
        kernelStencil_v1<<<Dg, Db>>>(u1_d, u2_d,
                                     DIMX + 2 * RADIUS,
                                     DIMY + 2 * RADIUS,
                                     DIMZ + 2 * RADIUS);

        float * tmp = u2_d;
        u2_d = u1_d;
        u1_d = tmp;
		cudaThreadSynchronize();
    }
    
    CUT_SAFE_CALL(cutStopTimer(timer_gpu1));

    CUT_SAFE_CALL(cutStartTimer(timer_mem1));
	CUDA_SAFE_CALL(cudaMemcpy(u2_h, u2_d, size, cudaMemcpyDeviceToHost));
    CUT_SAFE_CALL(cutStopTimer(timer_mem1));

    /**************************************************
     * GPU execution v2
     **************************************************/
    initVolumes(u1_h, u2_h, DIMX + 2 * RADIUS,
                            DIMY + 2 * RADIUS,
                            DIMZ + 2 * RADIUS);

    CUT_SAFE_CALL(cutStartTimer(timer_mem2));
	CUDA_SAFE_CALL(cudaMemcpy(u1_d, u1_h, size, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(u2_d, u2_h, size, cudaMemcpyHostToDevice));
    CUT_SAFE_CALL(cutStopTimer(timer_mem2));

    CUT_SAFE_CALL(cutStartTimer(timer_gpu2));

    for (unsigned i = 0; i < ITERATIONS; i++) {
        kernelStencil_v2<<<Dg, Db>>>(u1_d, u2_d,
                                     DIMX + 2 * RADIUS,
                                     DIMY + 2 * RADIUS,
                                     DIMZ + 2 * RADIUS);

        float * tmp = u2_d;
        u2_d = u1_d;
        u1_d = tmp;
		cudaThreadSynchronize();
    }
    
    CUT_SAFE_CALL(cutStopTimer(timer_gpu2));

    CUT_SAFE_CALL(cutStartTimer(timer_mem2));
	CUDA_SAFE_CALL(cudaMemcpy(u2_h, u2_d, size, cudaMemcpyDeviceToHost));
    CUT_SAFE_CALL(cutStopTimer(timer_mem2));

    /**************************************************
     * GPU execution v3
     **************************************************/
    initVolumes(u1_h, u2_h, DIMX + 2 * RADIUS,
                            DIMY + 2 * RADIUS,
                            DIMZ + 2 * RADIUS);

    CUT_SAFE_CALL(cutStartTimer(timer_mem3));
	CUDA_SAFE_CALL(cudaMemcpy(u1_d, u1_h, size, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(u2_d, u2_h, size, cudaMemcpyHostToDevice));
    CUT_SAFE_CALL(cutStopTimer(timer_mem3));

    CUT_SAFE_CALL(cutStartTimer(timer_gpu3));

    for (unsigned i = 0; i < ITERATIONS; i++) {
        kernelStencil_v3<<<Dg, Db>>>(u1_d, u2_d,
                                     DIMX + 2 * RADIUS,
                                     DIMY + 2 * RADIUS,
                                     DIMZ + 2 * RADIUS);

        float * tmp = u2_d;
        u2_d = u1_d;
        u1_d = tmp;
		cudaThreadSynchronize();
    }

    CUT_SAFE_CALL(cutStopTimer(timer_gpu3));

    CUT_SAFE_CALL(cutStartTimer(timer_mem3));
	CUDA_SAFE_CALL(cudaMemcpy(u2_h, u2_d, size, cudaMemcpyDeviceToHost));
    CUT_SAFE_CALL(cutStopTimer(timer_mem3));

    /**************************************************
     * Print timing results
     **************************************************/
    printf("  CPU time           : %.2f (ms)\n\n",
            cutGetTimerValue(timer_cpu));
    printf("  GPU v1 time compute: %.2f (ms)\n",
            cutGetTimerValue(timer_gpu1));
    printf("  GPU v1 time memory : %.2f (ms)\n",
            cutGetTimerValue(timer_mem1));
    printf("  GPU v1 time total  : %.2f (ms): speedup %.2fx\n\n",
            cutGetTimerValue(timer_gpu1) + cutGetTimerValue(timer_mem1),
            cutGetTimerValue(timer_cpu)/(cutGetTimerValue(timer_gpu1) + cutGetTimerValue(timer_mem1)));
    printf("  GPU v2 time compute: %.1f (ms)\n",
            cutGetTimerValue(timer_gpu2));
    printf("  GPU v2 time memory : %.2f (ms)\n",
            cutGetTimerValue(timer_mem2));
    printf("  GPU v2 time total  : %.2f (ms): speedup %.2fx\n\n",
            cutGetTimerValue(timer_gpu2) + cutGetTimerValue(timer_mem2),
            cutGetTimerValue(timer_cpu)/(cutGetTimerValue(timer_gpu2) + cutGetTimerValue(timer_mem2)));
    printf("  GPU v3 time compute: %.2f (ms)\n",
            cutGetTimerValue(timer_gpu3));
    printf("  GPU v3 time memory : %.2f (ms)\n",
            cutGetTimerValue(timer_mem3));
    printf("  GPU v3 time total  : %.2f (ms): speedup %.2fx\n",
            cutGetTimerValue(timer_gpu3) + cutGetTimerValue(timer_mem3),
            cutGetTimerValue(timer_cpu)/(cutGetTimerValue(timer_gpu3) + cutGetTimerValue(timer_mem3)));

    /**************************************************
     * Free data structures
     **************************************************/
    CUDA_SAFE_CALL(cudaFree(u1_d));
    CUDA_SAFE_CALL(cudaFree(u2_d));
    free(u1_h);
    free(u2_h);

    return 0;
}
