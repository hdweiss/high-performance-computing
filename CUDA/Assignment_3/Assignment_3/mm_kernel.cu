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

	float Cvalue[BLOCK_SIZE];

    int col = threadIdx.x;
	
	#pragma unroll BLOCK_SIZE
	for(int i = 0; i < BLOCK_SIZE; ++i){
		Cvalue[i] = 0.0f;
	}


    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
		__syncthreads();
        Matrix Asub = GetSubMatrix(A, blockRow, m, BLOCK_SIZE);
        Matrix Bsub = GetSubMatrix(B, m, blockCol, BLOCK_SIZE);

        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
		
		#pragma unroll BLOCK_SIZE
		for(int i = 0; i < BLOCK_SIZE; ++i){
			As[i][col] = GetElement(Asub, i, col);
		}

        __syncthreads();

		#pragma unroll BLOCK_SIZE
		for(int e = 0; e < BLOCK_SIZE; ++e){	
			float Bs = GetElement(Bsub, e, col);
			#pragma unroll BLOCK_SIZE
			for (int i = 0; i < BLOCK_SIZE; ++i)
				Cvalue[i] += As[i][e] * Bs; 
		}
    }
	for(int i = 0; i < BLOCK_SIZE; ++i){
		SetElement(Csub, i, col, Cvalue[i]);
	}
}

#define BLOCK_SIZE2 64
__global__ void mm_kernel4(Matrix C, const Matrix A, const Matrix B)
// Optimized 4b
{
	int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Each thread block computes one sub­matrix Csub of C
    Matrix Csub = GetSubMatrixXX(C, blockRow, blockCol, BLOCK_SIZE2, BLOCK_SIZE);

	float Cvalue[BLOCK_SIZE];

    int col = threadIdx.x;

	//#pragma unroll BLOCK_SIZE
	for(int i = 0; i < BLOCK_SIZE; ++i){
		Cvalue[i] = 0.0f;
	}


    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
		__syncthreads();
        Matrix Asub = GetSubMatrix(A, blockRow, m, BLOCK_SIZE);
        Matrix Bsub = GetSubMatrixXX(B, m, blockCol, BLOCK_SIZE2, BLOCK_SIZE);

        __shared__ float As[BLOCK_SIZE*BLOCK_SIZE];

		for(int i = 0; i < 4; ++i)
			As[(i*BLOCK_SIZE2)+col] = GetElement(Asub, (col/BLOCK_SIZE)+i*4, col&(BLOCK_SIZE-1));

        __syncthreads();

		for(int e = 0; e < BLOCK_SIZE; ++e){	
			float Bs = GetElement(Bsub, e, col);
			for (int i = 0; i < BLOCK_SIZE; ++i)
				Cvalue[i] += As[(i*BLOCK_SIZE)+e] * Bs; 
		}
    }
	for(int i = 0; i < BLOCK_SIZE; ++i){
		SetElement(Csub, i, col, Cvalue[i]);
	}
}

__global__ void mm_kernel5(Matrix C, const Matrix A, const Matrix B)
// Optimize as much as possible
{
	int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Each thread block computes one sub­matrix Csub of C
    Matrix Csub = GetSubMatrixXX(C, blockRow, blockCol, BLOCK_SIZE2, BLOCK_SIZE);

	float Cvalue[BLOCK_SIZE];

    int col = threadIdx.x;

	#pragma unroll BLOCK_SIZE
	for(int i = 0; i < BLOCK_SIZE; ++i){
		Cvalue[i] = 0.0f;
	}


	int mm = (A.width / BLOCK_SIZE);
    for (int m = 0; m < mm; ++m) {
		__syncthreads();
        Matrix Asub = GetSubMatrix(A, blockRow, m, BLOCK_SIZE);
        Matrix Bsub = GetSubMatrixXX(B, m, blockCol, BLOCK_SIZE2, BLOCK_SIZE);

        __shared__ float As[BLOCK_SIZE*BLOCK_SIZE];

		#pragma unroll 4
		for(int i = 0; i < 4; ++i){		//				Shift by 4 = dividing by the block size
										//							Anding with the block size minus 1 = modulo block size
			As[(i*BLOCK_SIZE2)+col] = GetElement(Asub, (col>>4)+i*4, col&(BLOCK_SIZE-1));
		}

        __syncthreads();

		#pragma unroll BLOCK_SIZE
		for(int e = 0; e < BLOCK_SIZE; ++e){	
			float Bs = GetElement(Bsub, e, col);
			#pragma unroll BLOCK_SIZE
			for (int i = 0; i < BLOCK_SIZE; ++i)
				Cvalue[i] += As[(i*BLOCK_SIZE)+e] * Bs; 
		}
    }
	#pragma unroll BLOCK_SIZE
	for(int i = 0; i < BLOCK_SIZE; ++i){
		SetElement(Csub, i, col, Cvalue[i]);
	}
}

