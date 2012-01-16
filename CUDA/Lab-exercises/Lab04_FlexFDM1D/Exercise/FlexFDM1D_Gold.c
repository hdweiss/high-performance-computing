//
// Host code for computing flexible-order finite differences.
//
// Written by Allan Engsig-Karup, October 20, 2010.
//

void FlexFDM1D_Gold(float* U, float* Ux, int N, int alpha, float* stencils, int rank) 
{
	// index in output vector Ux to be computed
	
	for (int row=0; row<N; ++row)
	{ 
		float value=0.0;
		// Compute dot-product between FDM stencil weights and input vector U

		// diff is used for automatically taking one-sided difference near boundaries
		int diff = 0; 
		if (row<alpha)
			diff = alpha - row;
		else if (row>N-1-alpha)  
			diff = N-1-alpha-row;

		// use diff to determine which pre-computed stencil we need
		int tmp = (alpha-diff)*rank+alpha;
		// use diff to off-center input values near boundaries
		int tmp2 = row + diff;
		
		// Compute finite difference approximation
		for (int i = -alpha; i<alpha+1; ++i)
		{
			value += U[tmp2+i]*stencils[tmp+i];
		}
		
		// Output result
		Ux[row] = value; 
	}
}

