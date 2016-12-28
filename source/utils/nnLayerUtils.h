#ifndef _nnLayerUtils_h
#define _nnLayerUtils_h
#ifdef MKL
#include <mkl.h>
#include <mkl_cblas.h>
#include <mkl_blas.h>
#include <mkl_lapack.h>
#include <mkl_lapacke.h>
#endif

#ifndef MKL
#define MKL_INT const int
#endif

typedef struct nnLayer {
	MKL_INT n;
	MKL_INT m;
	float *A;
	float *b;
} nnLayer;

#endif