__global__ void DotProd_kernel(float *result, const float* vec1, const float* vec2, int N) 
{

  // YOUR TASKS:
  // - Write kernel body to compute element-wise product between elements of vec1 and vec2 and return result in
  //   new vec. 
  // - Make sure that arbitrary sizes of N can be used.
  // Insert code below this line.

 const unsigned int outputIdx = threadIdx.x + blockDim.x*blockIdx.x ;
 
 //?????
 if(outputIdx < N)
	result[outputIdx] = vec1[outputIdx] * vec2[outputIdx];

}
