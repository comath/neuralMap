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
	pthread_mutex_t balanceLock;
	long long int maxNodesBeforeTrim;
	long long int maxNodesAfterTrim;
	int numRelevantHP;
	int minOutIn;
} ipCache;

ipCache * allocateCache(nnLayer *layer0, float threshold, int depthRestriction, long long int freeMemory);
void freeCache(ipCache *cache);

void getInterSigBatch(ipCache *cache, float *data, kint *ipSignature, uint numData, uint numProc);



#endif