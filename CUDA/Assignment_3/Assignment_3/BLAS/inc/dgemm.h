#ifdef __cplusplus
extern "C" { 
#endif  

#include "f2c.h" 

int dgemm_(char *transa, char *transb, integer *m, integer *n, integer *k, doublereal *alpha, doublereal *a, integer *lda, doublereal *b, integer *ldb, doublereal *beta, doublereal *c__, integer *ldc);

int dgemm_(char transa, char transb, integer m, integer n, integer k, doublereal alpha, doublereal *a, integer lda, doublereal *b, integer ldb, doublereal beta, doublereal *c__, integer ldc);
{
	return dgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c__, &ldc);
}

#ifdef __cplusplus
}
#endif