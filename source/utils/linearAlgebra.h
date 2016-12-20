#ifndef _linearAlgebra_h
#define _linearAlgebra_h
#include <mkl.h>
#include <mkl_cblas.h>
#include <mkl_blas.h>
#include <mkl_lapack.h>
#include <mkl_lapacke.h>

float c_dot(float *v, float *u, int n);

#endif