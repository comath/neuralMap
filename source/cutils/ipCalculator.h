/*
Soon to be repurposed to be used in ipTrace
*/

#ifndef _ipCalculator_h
#define _ipCalculator_h
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdint.h>

#ifdef USE_MKL
#include <mkl.h>
#include <mkl_cblas.h>
#include <mkl_blas.h>
#include <mkl_lapack.h>
#include <mkl_lapacke.h>
#endif

#ifdef USE_OPENBLAS
#include <lapacke.h>
#include <cblas.h>
#endif

#include "parallelTree.h"
#include "nnLayerUtils.h"

typedef struct intersection {
	int numHps;
	float *subA;
	float *subB;
} intersection;

typedef struct distanceWithIndex {
	int index;
	float dist;
} distanceWithIndex;

typedef struct ipMemory {
	float *s;  // Diagonal entries
	float *u; 
	float *vt; 
	float *c;  
	float *localCopy;
	float *superb; 
	float *px;
	float *py;
	intersection *I;
	distanceWithIndex *distances;
	uint minMN;
	
} ipMemory;

typedef struct ipCache {
	nnLayer *layer;
	Tree *bases;
	float *hpOffsetVecs;
	float *hpNormals;
	float threshold;
	pthread_mutex_t balanceLock;
} ipCache;

// Types should be: kernel projection or perp kernel projection

ipCache * allocateCache(nnLayer *layer0, float threshold, long long freeMemory);
void freeCache(ipCache *cache);

void getInterSigBatch(ipCache *cache, float *data, kint *ipSignature, uint numData, uint numProc);
void traceDistsSigBatch(ipCache *cache, float *data, kint *ipSigTraces, float * dists, uint numData, uint numProc);



#endif