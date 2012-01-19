
#include <cutil.h>

typedef struct {
    int width;
    int height;
    int stride;
    float* elements;
} Matrix;

inline void create_matrix (Matrix *m, int row, int col) {
	m->width = col;
	m->height = row;
	m->stride = col;
	m->elements = (float*)malloc(row*col*sizeof(float));
}

inline Matrix clone_matrix(Matrix* device_matrix) {
	Matrix matrix;
	matrix.height = device_matrix->height;
	matrix.width = device_matrix->width;
	matrix.stride = device_matrix->stride;

	int size = matrix.width * matrix.height * sizeof(float);
	matrix.elements = (float*)malloc(size);

	return matrix;
}

inline void copy_matrix_from_device(Matrix* host_matrix, Matrix* device_matrix) {
	int size = device_matrix->width * device_matrix->height * sizeof(float);

	CUDA_SAFE_CALL( cudaMemcpy( host_matrix->elements, device_matrix->elements, 
		size, cudaMemcpyDeviceToHost) );
}

inline Matrix alloc_matrix_on_device(Matrix* matrix) {
	int size = matrix->width * matrix->height * sizeof(float);

	Matrix matrix_d;
	matrix_d.height = matrix->height;
	matrix_d.width = matrix->width;
	matrix_d.stride = matrix->stride;

	CUDA_SAFE_CALL( cudaMalloc( &matrix_d.elements, size) );
	return matrix_d;
}

inline void copy_matrix_to_device(Matrix* h_matrix, Matrix* d_matrix) {
	int size = h_matrix->width * h_matrix->height * sizeof(float);
	CUDA_SAFE_CALL( cudaMemcpy( d_matrix->elements, h_matrix->elements, size, cudaMemcpyHostToDevice) );
	return;
}



inline long check_matrix(Matrix* correct, Matrix* matrix) {
	long errors = 0;
	int debuggy = 1;
	
	for (int i = 0; i < matrix->width*matrix->height; i++) {
		//if(matrix->elements[i] != 1.0) {
		if(abs(matrix->elements[i] - correct->elements[i]) > 0.1) {
			errors++;

			//if(debuggy)
			//	printf("Error at: %i\n", i);
			if(debuggy && errors == 1){
				printf("  1. Error at: %i   Elements: %f vs. %f\n", i, matrix->elements[i], correct->elements[i]);
			}
		}
	}

	return errors;
}

inline void print_matrix_result(Matrix* matrix, const char* name, 
								unsigned timer_gpu, unsigned timer_cpu, Matrix* correct) {

	printf("  GPU time %s   : %.4f (ms) , speedup %.2fx\n", name,
		cutGetTimerValue(timer_gpu),cutGetTimerValue(timer_cpu)/cutGetTimerValue(timer_gpu));

	int errors = check_matrix(correct, matrix);
	if (errors == 0)
		printf("  Passed\n\n");
	else {
		printf("  FAILED with %i errors\n\n", errors);
	}
}

// Get a matrix element
__device__ float GetElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}

// Set a matrix element
__device__ inline void SetElement(Matrix A, int row, int col,
                           float value)
{
    A.elements[row * A.stride + col] = value;
}

// Get the BLOCK_SIZExBLOCK_SIZE sub-­matrix Asub of A that is
// located col sub-­matrices to the right and row sub-­matrices down
// from the upper-­left corner of A
__device__ Matrix GetSubMatrix(Matrix A, int row, int col, int BLOCK_SIZE)
{
    Matrix Asub;
    Asub.width = BLOCK_SIZE;
    Asub.height = BLOCK_SIZE;
    Asub.stride = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row
                                    + BLOCK_SIZE * col];
    return Asub;
}

__device__ Matrix GetSubMatrixXX(Matrix A, int row, int col, int BLOCK_WIDTH, int BLOCK_HEIGHT)
{
    Matrix Asub;
    Asub.width = BLOCK_WIDTH;
    Asub.height = BLOCK_HEIGHT;
    Asub.stride = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_HEIGHT * row
                                    + BLOCK_WIDTH * col];
    return Asub;
}
