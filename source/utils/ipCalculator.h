#ifndef _ipCalculator_h
#define _ipCalculator_h
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "parallelTree.h"
#include "nnLayerUtils.h"

#include <mkl.h>
#include <mkl_cblas.h>
#include <mkl_blas.h>
#include <mkl_lapack.h>
#include <mkl_lapacke.h>



typedef struct ipCache {
	nnLayer *layer0;
	Tree *bases;
	float *hpOffsetVecs;
	float *hpNormals;
	float threshold;
} ipCache;

ipCache * allocateCache(nnLayer *layer0, float threshold);
void freeCache(ipCache *cache);

void getInterSig(ipCache * cache, float *p, uint *ipSignature);
void getInterSigBatch(ipCache *cache, float *data, uint *ipSignature, uint numData, uint numProc);



#endif