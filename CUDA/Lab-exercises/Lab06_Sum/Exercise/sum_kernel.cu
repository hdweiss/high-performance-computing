__global__ void sum_kernel(float *g_odata, float *g_idata, int N)
// Naive kernel
{

 // YOUR TASKS:
 // - Write a naive kernel where parallel sum reduction is done on a per block basis
 //   and reduction sum is returned in g_odata.
 // - For simplicity, assume kernel only considers a dataset within each block of size 2^p, p=1,2,É
 //

  // access thread id within block
  //unsigned int t = ???;

  // Reduction per block in global memory
  //?????

  // Output partial sum
  //if (t==0) g_odata[ ????? ] = g_idata[ ????? ];

}

__global__ void sum_kernel2(float *g_odata, float *g_idata, int N)
// Shared memory kernel
{

 // YOUR TASKS:
 // - Improve naive kernel to use shared memory per block for reduction
 // - Employ dynamic allocation of shared memory where mem block size is determined by host
 // - Threads within a block should collaborate on loading data from device to shared memory
 // - For simplicity, assume kernel only considers a dataset within each block of size 2^p, p=1,2,É
 //

  // shared mem array
  // the size is determined by the host application

  // access thread id
  //unsigned int t = ????? ;

  // read in input data to shared memory from global memory
  //?????
  __syncthreads();

  // Reduction per block in shared memory
  //?????

  // Output partial sum
  //if (t==0) g_odata[blockIdx.x] = ?????;
}

__global__ void sum_kernel3(float *g_odata, float *g_idata, int N)
{

 // YOUR TASKS:
 // - Change stride pattern in sum_kernel2 for reduction step.

}

__global__ void sum_kernel4(float *g_odata, float *g_idata, int N)
{

 // YOUR TASKS:
 // - Change stride pattern in sum_kernel3 for reduction step.

}

__global__ void sum_kernel5(float *g_odata, float *g_idata, int N)
{

 // YOUR TASKS:
 // - Optimize as much as possible

}

