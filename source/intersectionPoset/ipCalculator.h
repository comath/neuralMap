#ifndef _ipCalculator_h
#define _ipCalculator_h
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "../utils/parallelTree.h"
#include "../utils/nnLayerUtils.h"

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


typedef struct ipCache {
	Tree *bases;
	float *hpOffsetVecs;
	float *hpNormals;
	uint inDim;
	uint outDim;
} ipCache;

ipCache * allocateCache(nnLayer *layer0);
int * getIntersectionSignature(ipCache *cache, float *v);
void freeCache(ipCache *cache);



#endif