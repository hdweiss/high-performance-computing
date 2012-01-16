__global__ void FlexFDM1D_naive(float* U, float* Ux, int N, int alpha, float* stencils)
//
// Naive version where only global memory and automatic variables are accessed.
//
{

 // YOUR TASKS:
 // - Write body of kernel for computing Finite Difference Approksimations for
 //   threads in the grid. 
 // - Arbitrary sizes of N should be allowed (N can be larger than total threads).
 // Insert code below this line.

}

//
// Kernel v2
//

__global__ void FlexFDM1D_v2(float* U, float* Ux, int N, int alpha, float* stencils)
//
// Improved version where shared memory is used to reduce global memory accesses.
//
{

 // YOUR TASKS:
 // - Write body of kernel for computing Finite Difference Approksimations for
 //   threads in the grid. 
 // - Arbitrary sizes of N should be allowed (N can be larger than total threads).
 // - Utilize shared memory
 // Insert code below this line.

}

//
// Kernel v3
//

__global__ void FlexFDM1D_v3(float* U, float* Ux, int N, int alpha, float* stencils)
//
// Improved version where shared memory is used to reduce global memory accesses.
//
{

 // YOUR TASKS:
 // - Write body of kernel for computing Finite Difference Approksimations for
 //   threads in the grid. 
 // - Arbitrary sizes of N should be allowed (N can be larger than total threads).
 // - Utilize shared memory
 // - Utilize constant memory for stencils coefficients
 // Insert code below this line.

}
