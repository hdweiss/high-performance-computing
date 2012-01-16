//
// kernel routine
//

__global__ void VecAdd_kernel(const float* A, const float* B, float* C, int N) 
/* Naive kernel */
{
// Uncomment line below and define global index form block and thread indexes
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if(i < N){
		C[i] = A[i] + B[i];
	}

} 


