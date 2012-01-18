#include "sgemm.h"

void mm_gold( float* a, float* b, float* c, const unsigned int N) 
{
	float alpha = 1;
	float beta = 0;

	int lda = N; // 10. Parameter
	int ldb = N; // 8. Parameter

	int ldc = N; // 13. Parameter
	
	sgemm('N', 'N',
		N, N, N,
		alpha,
		b, ldb,
		a, lda,
		beta,
		c, ldc);

	return;
}


