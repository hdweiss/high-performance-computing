__global__ void mm_kernel1(Matrix C, const Matrix A, const Matrix B)
// Naive kernel
{
	 int y = threadIdx.y + blockDim.y * blockIdx.y;
	 int x = threadIdx.x + blockDim.x * blockIdx.x;

	 float sum = 0.0f;

	 int k = A.width;
	 int n = B.width;

	 if(y < C.height && x < C.width) {

		 for(int i = 0; i < k; i++)
			 sum += A.elements[y*k + i] * B.elements[i*n+ x];

		 C.elements[y * n + x] = sum;
	 }
}

#define BLOCK_SIZE 16

__global__ void mm_kernel2(Matrix C, const Matrix A, const Matrix B)
// Shared memory kernel
{
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Each thread block computes one sub­matrix Csub of C
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol, BLOCK_SIZE);

    float Cvalue = 0;

    int row = threadIdx.y;
    int col = threadIdx.x;

    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
        Matrix Asub = GetSubMatrix(A, blockRow, m, BLOCK_SIZE);
        Matrix Bsub = GetSubMatrix(B, m, blockCol, BLOCK_SIZE);

        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);

        __syncthreads();

        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];

        __syncthreads();
    }
    SetElement(Csub, row, col, Cvalue);
}

__global__ void mm_kernel3(Matrix C, const Matrix A, const Matrix B)
// Optimized 4a
{
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Each thread block computes one sub­matrix Csub of C
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol, BLOCK_SIZE);

	float Cvalue[BLOCK_SIZE] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

    int row = threadIdx.y;
    int col = threadIdx.x;

    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
        Matrix Asub = GetSubMatrix(A, blockRow, m, BLOCK_SIZE);
        Matrix Bsub = GetSubMatrix(B, m, blockCol, BLOCK_SIZE);

        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        As[row][col] = GetElement(Asub, row, col);

		float Bs[BLOCK_SIZE] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};


        __syncthreads();
		for(int i = 0; i < BLOCK_SIZE; ++i){
			// TODO populate Bs to new values.
			
			for (int e = 0; e < BLOCK_SIZE; ++e)
				Cvalue += As[row+i][e] * Bs[e]; // TODO: Fix the indexing of the Arrays
		}

        __syncthreads();
    }
    SetElement(Csub, row, col, Cvalue);
}

__global__ void mm_kernel4(Matrix C, const Matrix A, const Matrix B)
// Optimized 4b
{

}

__global__ void mm_kernel5(Matrix C, const Matrix A, const Matrix B)
// Optimize as much as possible
{

}

