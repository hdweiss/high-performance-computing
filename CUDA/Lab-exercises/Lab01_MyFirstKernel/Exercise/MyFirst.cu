//
// include files
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cutil_inline.h>

#include "MyFirst_kernel.cu"

//
// main host code 
//

int main(int argc, char **argv)
{
  float *h_x, *d_x;
  int   nblocks, nthreads, nsize, n; 

  // set number of blocks, and threads per block
  nblocks  = 2;
  nthreads = 16;
  nsize    = nblocks*nthreads;

  // allocate memory for array
  h_x = (float *)malloc(nsize*sizeof(float));
  cudaMalloc((void **)&d_x, nsize*sizeof(float));

  // execute kernel
  my_first_kernel<<<nblocks,nthreads>>>(d_x);

  // copy results from device to host
  cudaMemcpy(h_x,d_x,nsize*sizeof(float),cudaMemcpyDeviceToHost);

  // print results
  for (n=0; n<nsize; n++) printf(" n,  x  =  %d  %f \n",n,h_x[n]);

  // check results
  float sumcheck = 0.; 
  float sumcheckcorrect = 0.;
  for (int i = 0; i < nblocks * nthreads; ++i) {
     sumcheck += h_x[i]; 
  }
  for (int j=0; j<nthreads; ++j) {
     sumcheckcorrect += j;
  }
  sumcheckcorrect *= 2;
  if (fabs(sumcheck-sumcheckcorrect)<1e-6) {
     printf("PASSED!\n");
  }
  else
  {
     printf("FAILED!\n");
  }

  // free memory 
  cudaFree(d_x);
  free(h_x);

  return 0;
}

 
