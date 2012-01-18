__global__ void sum_kernel(float *g_odata, float *g_idata, int N)
// Naive kernel
{

 // YOUR TASKS:
 // - Write a naive kernel where parallel sum reduction is done on a per block basis
 //   and reduction sum is returned in g_odata.
 // - For simplicity, assume kernel only considers a dataset within each block of size 2^p, p=1,2,É
 //

  // access thread id within block
  unsigned int t = threadIdx.x;

  for(int i = 2; i <= (blockDim.x*blockDim.y); i*=2){
	if (t&(i-1) == 0) 
		g_idata[t + blockDim.x*blockIdx.x] = g_idata[t + blockDim.x*blockIdx.x] + g_idata[t+(i/2) + blockDim.x*blockIdx.x];

	__syncthreads();

  }

  if (t==0) g_odata[blockIdx.x] = g_idata[blockDim.x*blockIdx.x];

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
  unsigned int tid = threadIdx.x;
  unsigned int utid = threadIdx.x + blockDim.x * blockIdx.x;

  __shared__ float sum[128];
	
  sum[tid] = g_idata[utid];

  __syncthreads();

  for(int i = 2; i <= (blockDim.x*blockDim.y); i*=2){
	if (tid &(i-1) == 0) 
		sum[tid] = sum[tid] + sum[(i/2) + tid];

	__syncthreads();

  }

  if (tid==0)
	  g_odata[blockIdx.x] = sum[tid];

  // Reduction per block in shared memory
  //?????

  // Output partial sum
  //if (t==0) g_odata[blockIdx.x] = ?????;
}

__global__ void sum_kernel3(float *g_odata, float *g_idata, int N)
{

 // YOUR TASKS:
 // - Change stride pattern in sum_kernel2 for reduction step.
  unsigned int tid = threadIdx.x;
  unsigned int size = blockDim.x * blockDim.y;
  unsigned int utid = threadIdx.x + blockDim.x * blockIdx.x;

  __shared__ float sum[128];
	
  sum[tid] = g_idata[utid];

  __syncthreads();

  /*for(int i = 0; i < (int) __log2f(size); i++){
	  if(tid < size/(2*i))
		sum[tid] = sum[tid] + sum[size/i - tid];
*/
  for (int i = size; i > 0; i = i/2 ) {
	  if (tid < i/2 )
		sum[tid] = sum[tid] + sum[i - tid - 1];

	__syncthreads();

  }

  if (tid==0)
	  g_odata[blockIdx.x] = sum[tid];
}

__global__ void sum_kernel4(float *g_odata, float *g_idata, int N)
{

 // YOUR TASKS:
 // - Change stride pattern in sum_kernel3 for reduction step.
	  unsigned int tid = threadIdx.x;
  unsigned int size = blockDim.x * blockDim.y;
  unsigned int utid = threadIdx.x + blockDim.x * blockIdx.x;

  __shared__ float sum[128];
	
  sum[tid] = g_idata[utid];

  __syncthreads();

  /*for(int i = 0; i < (int) __log2f(size); i++){
	  if(tid < size/(2*i))
		sum[tid] = sum[tid] + sum[size/i - tid];
*/
  for (int i = size; i > 0; i = i/2 ) {
	  if (tid < i/2 )
		sum[tid] = sum[tid] + sum[i - tid -1];

	__syncthreads();

  }

  if (tid==0)
	  g_odata[blockIdx.x] = sum[tid];

}

__global__ void sum_kernel5(float *g_odata, float *g_idata, int N)
{

 // YOUR TASKS:
 // - Optimize as much as possible

}

