#include "sgemm.h"

void mm_gold(const int m, const int n, const int k, float* a, float* b, float* c) 
{
	float alpha = 1;
	float beta = 0;

	int lda = k; // 10. Parameter
	int ldb = n; // 8. Parameter

	int ldc = n; // 13. Parameter
	
	sgemm('N', 'N',
		n, m, k,
		alpha,
		b, ldb,
		a, lda,
		beta,
		c, ldc);

	return;
}


