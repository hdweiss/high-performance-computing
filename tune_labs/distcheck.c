#include "distcheck.h"
#include <unistd.h>

#ifdef ALL_IN_ONE

double 
distcheck(particle_t *p, int n) {
    // dummy code - please change
    sleep(1);
    return(-1.0);
}

#else

double 
distcheck(double *v, int n) {
    //dummy code - please change
    sleep(1);
    return(-1.0);
}

#endif
