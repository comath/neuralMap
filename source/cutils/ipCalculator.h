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



ipCache * allocateCache(nnLayer *layer0, float threshold, long long freeMemory);
void freeCache(ipCache *cache);

void getInterSig(ipCache * cache, float *p, kint *ipSignature, ipMemory *mb);
void getInterSigBatch(ipCache *cache, float *data, kint *ipSignature, uint numData, uint numProc);



#endif