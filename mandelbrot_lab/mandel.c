void
mandel(int disp_width, int disp_height, int *array, int max_iter) {

    double 	scale_real, scale_imag;
    double 	x, y, u, v, u2, v2;
    int 	i, j, iter;

    scale_real = 3.5 / (double)disp_width;
    scale_imag = 3.5 / (double)disp_height;
	
#pragma omp parallel for \
	shared(max_iter,scale_real,scale_imag,array,disp_height,disp_width) \
	private(i,j,y,x,u,v,u2,v2,iter) \
	shedule(dynamic,6)
    for(i = 0; i < disp_width; i++) {

		x = ((double)i * scale_real) - 2.25; 

		for(j = 0; j < disp_height; j++) {
			y = ((double)j * scale_imag) - 1.75; 

			u    = 0.0;
			v    = 0.0;
			u2   = 0.0;
			v2   = 0.0;
			iter = 0;

			while ( u2 + v2 < 4.0 &&  iter < max_iter ) {
			v = 2 * v * u + y;
			u = u2 - v2 + x;
			u2 = u*u;
			v2 = v*v;
			iter = iter + 1;
			}

			// if we exceed max_iter, reset to zero
			iter = iter == max_iter ? 0 : iter;

			array[i*disp_height + j] = iter;
		}
    }
}
