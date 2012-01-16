// Simple iterative solution of Poisson problem in 2D.
//
// Implemented by 
//  - Stefan Lemvig Glimberg - slgl@imm.dtu.dk
//  - Allan P. Engsig-Karup  - apek@imm.dtu.dk
//
//
// Solves the 2D Poisson problem in a square grid with Dirichlet 
// boundary condition equal to zero and with a physical dimension 
// of 1x1.
// Uses finite central difference.
//
// du/dxx + du/dyy = -2*pi^2*sin(x*pi)*sin(y*pi)
//
// Grid is a unit square. Solution is defined such
// that it takes values of zero at boundaries and this
// is exploited in code. 
//
//      0         x         0
//    0 +-------------------+ 0
//      |   |   |   |   |   |
//      |---+---+---+---|---|
//      |   |   |   |   |   |
//      |---+---+---+---|---|
//    y |   |   | 1 |   |   | y
//      |---+---+---+---|---|
//      |   |   |   |   |   |
//      |---+---+---+---|---|
//      |   |   |   |   |   |
//    0 +---+---+---+---|---+ 0
//      0         x         0
//

// Defines
// Uncomment the line below to include the GPU kernel file instead of the CPU gold file
 //#define USE_DEVICE

// Uncomment the line below to switch to double precision
//#define float double

#define _USE_MATH_DEFINES

// Included C libraries
#include <stdio.h>
#include <math.h>

// Included CUDA libraries
#include <cutil.h> 

// Smoother types
enum SMOOTHER
{
	JAC,   // Jacobi
	GS,    // Gauss-Seidel
	RBGS   // Red-black Gauss-Seidel - Not implemented
};

// Included own files
#include "Poisson2D_kernel.cu"
#include "Poisson2D_Gold.c"

// Exact (analytic) solution to the Poisson problem
inline float fun(const float x, const float y, const int n)
{
	return sinf(M_PI*x*n)*sinf(M_PI*y*n);
}

int main(int argc, char* argv[]) {
	// Screen output
	printf("Iterative solution of Poisson’s problem in two space dimensions\n  ./Poisson2D <p:default=4> <maxiter:default=1000> <SMOOTHER:default=0>\n\n");

	int p  = (argc>1) ? max(1,atoi(argv[1])) : 4;                                     // Determines size of the mesh
	int maxiter = (argc>2) ? atoi(argv[2]) : 1000;                                    // Maximum number of outer iterations
	SMOOTHER smoother = (argc>3) ? (enum SMOOTHER) min(3,max(0,atoi(argv[3]))) : JAC; // Smoother type (JAC,GS,RBGS)
	float relres;       // Tolerance obtained after iterations
	
	int Nx = (1<<p)+1;	    			   // No. point is x-direction, always a 'nice' number
	int Ny = Nx;                           // Square domain
	int N = Nx*Ny;                         // Total number of elements
	float h = 1.f/((float)Nx-1);		   // Step size
	
	// Allocate host memory and reset
	float* u_h    = (float*) malloc(N*sizeof(float));
	float* u_true = (float*) malloc(N*sizeof(float));
	float* d_h    = (float*) malloc(N*sizeof(float));
	float* f_h    = (float*) malloc(N*sizeof(float));
	memset(f_h,0,sizeof(float)*N);
	memset(u_h,0,sizeof(float)*N);

	// Create the true solution for reference
	for(int i=0; i<Ny; ++i)
	{
		for(int j=0; j<Nx; ++j)
		{
			u_true[i*Nx+j] = fun((float)j*h,(float)i*h,1);
			u_h[i*Nx+j]  = 0.0f; 
		}
	}

	// Create the right hand side of the Poisson equation using FD on the true solution
	for(int i=1; i<Ny-1; ++i)
	{
		for(int j=1; j<Nx-1; ++j)
		{
			// Make sure that u_true is exact solution to problem by including truncation errors in the rhs function
			f_h[i*Nx+j] = 4.f*u_true[i*Nx+j] - u_true[(i-1)*Nx+j] - u_true[(i+1)*Nx+j] - u_true[i*Nx+(j-1)] - u_true[i*Nx+(j+1)];
		}
	}

#ifdef USE_DEVICE
	// YOUR TASKS:
	// Create and copy device pointers and copy u_h and f_h to the device
	

#endif

	char* ss[] = {"JAC","GS","RBGS"};
	char smoothname[30];
	sprintf(smoothname, "%s Method", ss[smoother]);

	// start timer
	double start = (double) clock();

    // Solve iteratively using maxiter iterations
#ifdef USE_DEVICE
	printf("Execution %s using GPU...\n",smoothname);
	// YOUR TASKS: Call smoother with your device pointers
	
#else
	printf("Execution %s using CPU...\n",smoothname);
	smooth_h(maxiter,smoother,u_h,f_h,Nx,Ny); // Host
#endif

	// end timer
	double end = (double) clock();
	
#ifdef USE_DEVICE
    // YOUR TASK:
    // - Copy the result back to host.
#endif

    defect(d_h, u_h, f_h, Nx, Ny); // Compute defect
	float norm2b = norm2(f_h, N);
    relres = norm2(d_h, N)/norm2b;
    
	printf("Time used: %f s\n", (end-start)/(double) CLOCKS_PER_SEC);

	// Print result
	printf("%d iterations completed to achieve a relative defect tolerance of %e\n",maxiter,relres);

        // Check if method was convergent
	if (relres<1.0f)
		printf("PASSED!\n");
	else
		printf("FAILED!\n");

	// Cleanup
	free(u_h);
	free(u_true);
	free(f_h);

#ifdef USE_DEVICE
    // YOUR TASKS: cleanup device memory

#endif

};
