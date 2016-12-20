#include "linearAlgebra.h"

float c_dot(float *v, float *u, int n){
	int i = 0;
	MKL_INT len = n;
	float ret = cblas_sdot (len, v, 1, u, 1);
	return ret;
}