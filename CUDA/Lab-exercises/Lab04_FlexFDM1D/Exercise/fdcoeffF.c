int min2(int a, int b)
{
	if (a<b)
		return a;
	else
		return b;
}

void fdcoeffF(int k, float xbar, float x[], float c[], int n)
{
	//
	// Compute coefficients for finite difference approximation for the
	// derivative of order k at xbar based on grid values at points in x.
	//
	// This function returns a row vector c of dimension 1 by n, where n=length(x),
	// containing coefficients to approximate u^{(k)}(xbar), 
	// the k'th derivative of u evaluated at xbar,  based on n values
	// of u at x(1), x(2), ... x(n).  
	//
	// If U is a column vector containing u(x) at these n points, then 
	// c*U will give the approximation to u^{(k)}(xbar).
	//
	// Note for k=0 this can be used to evaluate the interpolating polynomial 
	// itself.
	//
	// Requires length(x) > k.  
	// Usually the elements x(i) are monotonically increasing
	// and x(1) <= xbar <= x(n), but neither condition is required.
	// The x values need not be equally spaced but must be distinct.  
	//
	// This program should give the same results as fdcoeffV.m, but for large
	// values of n is much more stable numerically.
	//
	// Based on the program "weights" in 
	//   B. Fornberg, "Calculation of weights in finite difference formulas",
	//   SIAM Review 40 (1998), pp. 685-691.
	//
	// Note: Forberg's algorithm can be used to simultaneously compute the
	// coefficients for derivatives of order 0, 1, ..., m where m <= n-1.
	// This gives a coefficient matrix C(1:n,1:m) whose k'th column gives
	// the coefficients for the k'th derivative.
	//
	// In this version we set m=k and only compute the coefficients for
	// derivatives of order up to order k, and then return only the k'th column
	// of the resulting C matrix (converted to a row vector).  
	// This routine is then compatible with fdcoeffV.   
	// It can be easily modified to return the whole array if desired.
	//
	// From  http://www.amath.washington.edu/~rjl/fdmbook/  (2007)
	//
	// Translated from Matlab to C by Allan P. Engsig-Karup, 9 May 2010.
	//
	int s;
	float c1, c2, c3, c4, c5;
	int i, i1, mn, j1, s1, j;
	float* C = (float*) malloc(n*(k+1)*sizeof(float)); //[n][k+1];
		
	// initialize C array
	for (i=0; i<n; ++i)
		for (j=0; j<k+1; ++j)
			C[i*(k+1)+j]=0.0;
		
	c1 = 1.0;
	c4 = x[0] - xbar;
	C[0] = 1.0;
	for (i=0; i<n-1; ++i)
	{
		i1 = i+1;
		mn =  ((i)>(k-1)?(k-1):(i));
		c2 = 1.0;
		c5 = c4;
		c4 = x[i1] - xbar;
		for (j=-1; j<=i-1; ++j)
		{
			j1 = j+1;
			c3 = x[i1] - x[j1];
			c2 = c2*c3;
			if (j==i-1)
			{
				for (s=mn; s>=0; --s)
				{
					s1 = s+1;
					C[i1*(k+1)+s1] = c1*((s+1)*C[(i1-1)*(k+1)+s1-1] - c5*C[(i1-1)*(k+1)+s1])/c2;
				}
				C[i1*(k+1)] = -c1*c5*C[(i1-1)*(k+1)]/c2;
			}
			for (s=mn; s>=0; --s)
			{
				s1 = s+1;
				C[j1*(k+1)+s1] = (c4*C[j1*(k+1)+s1] - (s+1)*C[j1*(k+1)+s1-1])/c3;
            }
			C[j1*(k+1)] = c4*C[j1*(k+1)]/c3;
		}
		c1 = c2;
	}
	for (i=0; i<n; ++i)
		c[i] = C[i*(k+1)+k];            // last column of c gives desired row vector
	
	if(C) free(C);
		
}
