#ifndef _ipCalculator_h
#define _ipCalculator_h
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "parallelTree.h"
#include "nnLayerUtils.h"

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
	nnLayer *layer0;
	Tree *bases;
	float *hpOffsetVecs;
	float *hpNormals;
	uint inDim;
	uint outDim;
	float threshold;
} ipCache;

ipCache * allocateCache(nnLayer *layer0, float threshold);
void freeCache(ipCache *cache);

void getInterSig(float *p, uint *ipSignature, ipCache * cache);



#endif