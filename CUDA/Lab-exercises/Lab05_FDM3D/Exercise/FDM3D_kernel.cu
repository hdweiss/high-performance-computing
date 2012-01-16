#ifndef __STENCIL_COMMON
#define __STENCIL_COMMON

#include "FDM3D.h"

#define C0 (0.f)
#define C1 (4.f/5.f)
#define C2 (-1.f/5.f)
#define C3 (4.f/105.f)
#define C4 (-1.f/280.f)

    void
kernelStencil_gold(const float * input,
        float * output,
        unsigned dimX,
        unsigned dimY,
        unsigned dimZ)
{
    unsigned dimXY = dimX * dimY;

    // Loop over all interior points in the domain (i.e. omit halo zones)
    for (int k = RADIUS; k < dimZ - RADIUS; k++) {
        for (int j = RADIUS; j < dimY - RADIUS; j++) {
            for (int i = RADIUS; i < dimX - RADIUS; i++) {
                int idx = k * dimXY + j * dimX + i;
                float value = C0 * input[idx];

                // Compute values with distance 1 respect to the central point
                value += C1 * (input[idx -         1] + input[idx +         1] +
                        input[idx -  dimX * 1] + input[idx +  dimX * 1] +
                        input[idx - dimXY * 1] + input[idx + dimXY * 1]);
                // Compute values with distance 2 respect to the central point
                value += C2 * (input[idx -         2] + input[idx +         2] +
                        input[idx -  dimX * 2] + input[idx +  dimX * 2] +
                        input[idx - dimXY * 2] + input[idx + dimXY * 2]);
                // Compute values with distance 3 respect to the central point
                value += C3 * (input[idx -         3] + input[idx +         3] +
                        input[idx -  dimX * 3] + input[idx +  dimX * 3] +
                        input[idx - dimXY * 3] + input[idx + dimXY * 3]);
                // Compute values with distance 4 respect to the central point
                value += C4 * (input[idx -         4] + input[idx +         4] +
                        input[idx -  dimX * 4] + input[idx +  dimX * 4] +
                        input[idx - dimXY * 4] + input[idx + dimXY * 4]);
                // Write the result
                output[idx] = value;
            }
        }
    }
}


__global__
    void
kernelStencil_v1(const float * input,
        float * output,
        unsigned dimX,
        unsigned dimY,
        unsigned dimZ)
{

 // YOUR TASKS:
 // - Compute unique threads indexes in 2D grid of threads
 // Insert code below this line.

    // Compute thread indexes
    unsigned ix = 0; // ?????
    unsigned iy = 0; // ?????

    int dimXY = dimX * dimY;

    // Compute global thread index
    int idx = RADIUS * dimXY +
        RADIUS * dimX  + RADIUS +
        iy * dimX + ix;

    // Advance in 2D planes
    for (int k = RADIUS; k < dimZ - RADIUS; k++) {
        // Read the central point
        float value   = C0 * input[idx];
        ///////////////////////
        // Code segment 1
        ///////////////////////

 // YOUR TASKS:
 // - Compute values by doing a naive parallelization of gold gode.
 // Insert code below this line.

        // Compute values with distance 1 respect to the central point
        //value += ?????
		output[idx] = value;
        idx += dimXY;
    }
}

__global__
    void
kernelStencil_v2(const float * input,
        float * output,
        unsigned dimX,
        unsigned dimY,
        unsigned dimZ)
{
    // Declare shared data for the thread block
    __shared__ float s_data[BLOCK_DIM_Y + 2 * RADIUS][BLOCK_DIM_X + 2 * RADIUS];

 // YOUR TASKS:
 // - Compute unique threads indexes in 2D grid of threads
 // Insert code below this line.

    // Compute thread indexes
    unsigned ix = 0; //?????
    unsigned iy = 0; //?????

    // size of plane including halo regions
    int dimXY = dimX * dimY;

    // Compute global thread index in cube with halo regions
    int idx = RADIUS * dimXY +
              RADIUS * dimX  +
              RADIUS + iy * dimX + ix;

    ///////////////////////
    // Code segment 2
    ///////////////////////

    // Compute thread indexes for the shared memory
    int tx = threadIdx.x + RADIUS;
    int ty = threadIdx.y + RADIUS;

    float current;

    // Advance in 2D planes
    for (int i = RADIUS; i < dimZ - RADIUS; i++) {
        // Read the current value
        current = input[idx];

        ///////////////////////
        // Code segment 3
        ///////////////////////

        // Load the halos into the shared memory
        if (threadIdx.y < RADIUS) { // above/below
		   s_data[threadIdx.y][tx]             = input[idx+dimX*blockDim.y];
		   s_data[threadIdx.y+blockDim.y+RADIUS][tx] = input[idx-dimX];
        }
        if (threadIdx.x < RADIUS) { // left/right
		   s_data[ty][threadIdx.x]             = input[idx-RADIUS];
		   s_data[ty][threadIdx.x+blockDim.x+RADIUS] = input[idx+blockDim.x];		
        }

        ///////////////////////
        // Code segment 4
        ///////////////////////

        // Load the central point into the shared memory
        s_data[ty][tx] = current;
        __syncthreads();

        ///////////////////////
        // Code segment 5
        ///////////////////////

        float value  = C0 * current;
        // Compute values with distance 1 respect to the central point
        value += C1*( s_data[ty-1][tx] + s_data[ty+1][tx] + s_data[ty][tx-1] + s_data[ty][tx+1] +
		              input[ idx-dimXY ] + input[ idx+dimXY ] );
        // Compute values with distance 2 respect to the central point
        value += C2*( s_data[ty-2][tx] + s_data[ty+2][tx] + s_data[ty][tx-2] + s_data[ty][tx+2] +
		              input[ idx-2*dimXY ] + input[ idx+2*dimXY ] );
        // Compute values with distance 3 respect to the central point
        value += C3*( s_data[ty-3][tx] + s_data[ty+3][tx] + s_data[ty][tx-3] + s_data[ty][tx+3] +
		              input[ idx-3*dimXY ] + input[ idx+3*dimXY ] );
        // Compute values with distance 4 respect to the central point
        value += C4*( s_data[ty-4][tx] + s_data[ty+4][tx] + s_data[ty][tx-4] + s_data[ty][tx+4] +
		              input[ idx-4*dimXY ] + input[ idx+4*dimXY ] );		
        // Write the result
        output[idx] = value;
        // Wait for the rest of threads in the thread block
        __syncthreads();
        idx += dimXY;
    }
}


__global__
    void
kernelStencil_v3(const float * input,
        float * output,
        unsigned dimX,
        unsigned dimY,
        unsigned dimZ)
{
    // Declare shared data for the thread block
    __shared__ float s_data[BLOCK_DIM_Y + 2 * RADIUS][BLOCK_DIM_X + 2 * RADIUS];
    // Compute thread indexes

 // YOUR TASKS:
 // - Compute unique threads indexes in 2D grid of threads
 // Insert code below this line.

    // Compute thread indexes
    unsigned ix = 0; //?????
    unsigned iy = 0; //?????

    int dimXY = dimX * dimY;
    int inIdx = RADIUS * dimX +
        RADIUS + iy * dimX + ix;
    int outIdx;

    ///////////////////////
    // Code segment 6
    ///////////////////////

    // Compute thread indexes for the shared memory
    int tx = threadIdx.x + RADIUS;
    int ty = threadIdx.y + RADIUS;

    // Registers used to extend the 3rd dimension
    float infront1, infront2, infront3, infront4;
    float behind1, behind2, behind3, behind4;
    float current;

    // Fill the "in-front" and "behind" data
    behind3  = input[inIdx]; inIdx += dimXY;
    behind2  = input[inIdx]; inIdx += dimXY;
    behind1  = input[inIdx]; inIdx += dimXY;

    current  = input[inIdx]; outIdx = inIdx; inIdx += dimXY;

    infront1 = input[inIdx]; inIdx += dimXY;
    infront2 = input[inIdx]; inIdx += dimXY;
    infront3 = input[inIdx]; inIdx += dimXY;
    infront4 = input[inIdx]; inIdx += dimXY;

    // Advance in 2D planes
    for (int i = RADIUS; i < dimZ - RADIUS; i++) {
        ///////////////////////
        // Code segment 7
        ///////////////////////

        // Advance the slice (move the thread-front)
        behind4  = behind3;
        behind3  = behind2;
        behind2  = behind1;
        behind1  = current;
        current  = infront1;
        infront1 = infront2;
        infront2 = infront3;
        infront3 = infront4;
        infront4 = input[inIdx]; // Read the current value

        inIdx  += dimXY;
        outIdx += dimXY;

        ///////////////////////
        // Code segment 8
        ///////////////////////

        // Load the halos into the shared memory
        if (threadIdx.y < RADIUS) { // above/below
		   s_data[threadIdx.y][tx]             = input[inIdx+dimX*blockDim.y];
		   s_data[threadIdx.y+blockDim.y+RADIUS][tx] = input[inIdx-dimX];
        }
        if (threadIdx.x < RADIUS) { // left/right
		   s_data[ty][threadIdx.x]             = input[inIdx-RADIUS];
		   s_data[ty][threadIdx.x+blockDim.x+RADIUS] = input[inIdx+blockDim.x];		
        }

        ///////////////////////
        // Code segment 9
        ///////////////////////

        // Load the central point into the shared memory
        s_data[ty][tx] = current;
        __syncthreads();

        ///////////////////////
        // Code segment 10
        ///////////////////////

        float value  = C0 * current;
        // Compute values with distance 1 respect to the central point
        value += C1*( s_data[ty-1][tx] + s_data[ty+1][tx] + s_data[ty][tx-1] + s_data[ty][tx+1] + 
		              behind1 + infront1 );
        // Compute values with distance 2 respect to the central point
        value += C2*( s_data[ty-2][tx] + s_data[ty+2][tx] + s_data[ty][tx-2] + s_data[ty][tx+2] +
		              behind2 + infront2 );
        // Compute values with distance 3 respect to the central point
        value += C3*( s_data[ty-3][tx] + s_data[ty+3][tx] + s_data[ty][tx-3] + s_data[ty][tx+3] +
		              behind3 + infront3 );		
        // Compute values with distance 4 respect to the central point
        value += C4*( s_data[ty-4][tx] + s_data[ty+4][tx] + s_data[ty][tx-4] + s_data[ty][tx+4] +
		              behind4 + infront4 );		
        // Write the result
        output[outIdx] = value;
        // Wait for the rest of threads in the thread block
        __syncthreads();
    }
}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
