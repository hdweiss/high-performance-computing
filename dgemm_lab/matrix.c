
#include "matrix.h"

extern int debug;

double** create_matrix(int row, int column) {
    double** matrix = (double**) malloc(row * sizeof(double*));
    double* space = (double*) calloc(row * column, sizeof(double));

    for(int p = 0; p < row; p++) {
        matrix[p] = space + p * column;
    }

    return matrix;
}

void destroy_matrix(double** matrix) {
    free(matrix[0]);
    free(matrix);
}

void init_matrix(int row, int column, int scale_factor, double** matrix) {
    for(int i = 0; i < row; i++) {
        for(int j = 0; j < column; j++) {
            matrix[i][j] = scale_factor * (i+1) + (j+1);
        }
    }
}

void init_zero(int p, int q, int s, double** matrix) {
	for(int i = 0; i < s; i++) {
        for(int j = 0; j < s; j++) {
            matrix[i+p*s][j+q*s] = 0.0;
        }
    }
}

void print_matrix(double** matrix, int row, int column, const char* message) {
    if(debug == 0)
        return;

    if(matrix == NULL) {
        printf("Print got null matrix %s", message);
        return;
    }
        
    printf("%s Matrix:\n", message);
    for(int i = 0; i < row; i++) {
        for(int j = 0; j < column; j++) {
            printf("%.1f\t", matrix[i][j]);
        }
        printf("\n");
    }
}


int simple_mm(int m, int n, int k, double** a, double** b, double** c) {
	double start_time;
	double loop_time = 0;
	int loopcount = 0;
	while((loop_time < 3.0) || (loopcount <= 2)){
		c = create_matrix(m, n);
		start_time = xtime();
		for(int i = 0; i < m; i++) {
		    for(int j = 0; j < n; j++) {
		        for(int q = 0; q < k; q++) {
		            c[i][j] = c[i][j] + a[i][q] * b[q][j];
		        }
		    }
		}
		loop_time += xtime() - start_time;
		loopcount++;
        destroy_matrix(c);
	}
    return loopcount;
}

double sub_mm(int p, int q, int r, int s, int m, int n, int k, double** a, double** b, double** c) {
	for(int i = 0; (i < s) && ((i+p*s) < m); i++) {	
        for(int j = 0; (j < s) && ((j+q*s) < n); j++) {
            for(int l = 0; (l < s) && ((l+r*s) < k); l++) {
                c[i+p*s][j+q*s] = c[i+p*s][j+q*s] + a[i+p*s][l+r*s] * b[l+r*s][j+q*s];
            }
        }
    }

}

int block_mm(int m, int n, int k, double** a, double** b, double** c, int s) { 
	double start_time;
	double loop_time = 0;
	double loopcount = 0;
	int p, q, r;
	double sn_d = (double)n/(double)s;
	double sm_d = (double)m/(double)s;
	double sk_d = (double)k/(double)s;
	int sn = (sn_d - (int)sn_d) > 0? ((int)sn_d + 1) : (int)sn_d;
	int sm = (sm_d - (int)sm_d) > 0? ((int)sm_d + 1) : (int)sm_d;
	int sk = (sk_d - (int)sk_d) > 0? ((int)sk_d + 1) : (int)sk_d;
	
	while(loop_time < 3.0 || loopcount <= 2){
		c = create_matrix(m,n);
		start_time = xtime();
		for (p = 0; p < sm; p++ ) {
			for (q = 0; q < sn; q++) {
				//clear submatrix
				// c(p,q) = 0
				//init_zero(p, q, s, c);
				for (r = 0; r < sk; r++) {
					//block multiplication
					// c(p,q)
					sub_mm(p, q, r, s, m, n, k, a, b, c);
					
				}
			}
		}
		loop_time += xtime() - start_time;
		loopcount++;
        destroy_matrix(c);
	}
	return loopcount;
}

int dgemm_mm(int m, int n, int k, double** a, double** b, double** c) {
	double start_time;
	double loop_time = 0;
	double loopcount = 0;
    double alpha = 1;
    double beta = 0;

    int lda = k; // 10. Parameter
    int ldb = n; // 8. Parameter

    int ldc = n; // 13. Parameter
	while(loop_time < 3.0 || loopcount <= 2){
		c = create_matrix(m,n);
	    start_time = xtime();
    	dgemm_64('N', 'N',
    	      n, m, k,
	          alpha,
	          b[0], ldb,
	          a[0], lda,
	          beta,
	          c[0], ldc);
		loop_time += xtime() - start_time;
		loopcount++;
        destroy_matrix(c);
	}
    return loopcount;
}
