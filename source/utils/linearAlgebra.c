#include "linearAlgebra.h"
#include <mkl.h>
#include <mkl_cblas.h>
#include <mkl_blas.h>
#include <mkl_lapack.h>
#include <mkl_lapacke.h>

float dot(float *v, float *u, int n)
{
	int i = 0;
	MKL_INT len = n;
	float ret = cblas_sdot (len, v, 1, u, 1);
	return ret;
}