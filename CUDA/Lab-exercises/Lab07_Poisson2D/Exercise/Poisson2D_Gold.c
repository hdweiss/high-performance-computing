// Calculates the 2-norm of v, ||v||.
float norm2(float const* v, const unsigned int N)
{
	float nrm = 0.0f;
	for (unsigned int i = 0; i < N; ++i)
		nrm += v[i]*v[i];

	return sqrt(nrm);
}

// Calculates the defect (residual) d = b-Ax
void defect(float* d, float const* u, float const* f, const unsigned int Nx, const unsigned int Ny)
{
	memset(d,0,sizeof(float)*Nx*Ny);
	for (unsigned int i = 1; i < Ny-1; ++i)
	{
		for (unsigned int j = 1; j < Nx-1; ++j)
		{
			d[i*Nx+j] = f[i*Nx+j] - (4*u[i*Nx+j] - u[(i-1)*Nx+j] - u[(i+1)*Nx+j] - u[i*Nx+(j-1)] - u[i*Nx+(j+1)]);
		}
	}
}

// Performs weighted Jacobi smoother iterations on all inner points.
void jacobi_h(float* u, float const* f, const unsigned int Nx, const unsigned int Ny, float w = 1.0f)
{
	float* u_in = (float*)malloc(Nx*Ny*sizeof(float));
	memcpy(u_in,u,sizeof(float)*Nx*Ny);
	
	for (unsigned int i = 1; i < Ny-1; ++i)
	{
		for (unsigned int j = 1; j < Nx-1; ++j)
		{
			u[i*Nx+j] = (1.f-w)*u_in[i*Nx+j] + w*0.25f*(u_in[(i-1)*Nx+j] + u_in[(i+1)*Nx+j] + u_in[i*Nx+(j-1)] + u_in[i*Nx+(j+1)] + f[i*Nx+j]);
		}
	}
}

// Performs Gauss-Seidel smoother iterations on all inner points.
void gauss_seidel_h(float* u, float const* f, const unsigned int Nx, const unsigned int Ny)
{
	for (unsigned int i = 1; i < Ny-1; ++i)
	{
		for (unsigned int j = 1; j < Nx-1; ++j)
		{
			u[i*Nx+j] = 0.25f*(u[(i-1)*Nx+j] + u[(i+1)*Nx+j] + u[i*Nx+(j-1)] + u[i*Nx+(j+1)] + f[i*Nx+j]);
		}
	}
}

// Smoothing based on smoother
inline void smooth_h(int iter, SMOOTHER s, float* u, float const* f, unsigned int Nx, unsigned int Ny)
{
	for(int i=0; i<iter; ++i)
	{
		switch (s)
		{
		case JAC:
			jacobi_h(u,f,Nx,Ny);
			break;
		case GS:
			gauss_seidel_h(u,f,Nx,Ny);
			break;
		case RBGS:
			// Not implemented on host, call GS
			gauss_seidel_h(u,f,Nx,Ny);
			break;
		}
	}
}
