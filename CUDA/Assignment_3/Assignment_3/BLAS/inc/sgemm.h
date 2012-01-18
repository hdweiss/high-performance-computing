#ifdef __cplusplus
extern "C" { 
#endif  

#include "f2c.h" 

int sgemm_(char *transa, char *transb, integer *m, integer *n, integer *k, real *alpha, real *a, integer *lda, real *b, integer *ldb, real *beta, real *c__, integer *ldc);

int sgemm(char transa, char transb, integer m, integer n, integer k, real alpha, real *a, integer lda, real *b, integer ldb, real beta, real *c__, integer ldc)
{
	return sgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c__, &ldc);
}

#ifdef __cplusplus
}
#endif