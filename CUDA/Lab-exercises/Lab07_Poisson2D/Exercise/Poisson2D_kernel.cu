
// Performs Jacobi smoother iterations on all inner points.
void jacobi_d(float* u, float const* f, const unsigned int Nx, const unsigned int Ny, float w = 1.0f)
{
	// YOUR TASKS:
	// Create and call kernel
}

// Performs Red-black Gauss-Seidel smoother iterations on all inner points.
void rb_gauss_seidel_d(float* u, float const* f, const unsigned int Nx, const unsigned int Ny)
{
	// YOUR TASKS:
	// Create and call kernel
}

// Smoothing based on smoother
inline void smooth_d(int iter, SMOOTHER s, float* u, float const* f, unsigned int Nx, unsigned int Ny)
{
	for(int i=0; i<iter; ++i)
	{
		switch (s)
		{
		case JAC:
			jacobi_d(u,f,Nx,Ny);
			break;
		case GS:
			// Not implemented on device, call red-black GS
			rb_gauss_seidel_d(u,f,Nx,Ny);
			break;
		case RBGS:
			rb_gauss_seidel_d(u,f,Nx,Ny);
			break;
		}
	}
}
