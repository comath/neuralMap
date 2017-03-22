#ifndef _ipCalculator_h
#define _ipCalculator_h
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "parallelTree.h"
#include "nnLayerUtils.h"
#include <stdint.h>

#include <mkl.h>
#include <mkl_cblas.h>
#include <mkl_blas.h>
#include <mkl_lapack.h>
#include <mkl_lapacke.h>



typedef struct ipCache {
	nnLayer *layer;
	Tree *bases;
	float *hpOffsetVecs;
	float *hpNormals;
	float threshold;
	int depthRestriction;
} ipCache;

ipCache * allocateCache(nnLayer *layer0, float threshold, int depthRestriction);
void freeCache(ipCache *cache);

void getInterSig(ipCache * cache, float *p, kint *ipSignature);
void getInterSigBatch(ipCache *cache, float *data, kint *ipSignature, uint numData, uint numProc);



#endif