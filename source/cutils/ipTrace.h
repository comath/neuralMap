#ifndef _fullTrace_h
#define _fullTrace_h
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "nnLayerUtils.h"
#include "key.h"
#include <stdint.h>
#include <float.h>
#include <string.h>

#include <mkl.h>
#include <mkl_cblas.h>
#include <mkl_blas.h>
#include <mkl_lapack.h>
#include <mkl_lapacke.h>


typedef struct distanceWithIndex {
	int index;
	float dist;
} distanceWithIndex;

typedef struct traceMemory {
	float * permA;
	float * pointBuff;
	float * tau;
	distanceWithIndex * distances;
	float * interDists;
} traceMemory;

typedef struct traceCache
{
	uint keyLen;
	nnLayer * layer;
	float * solution;

	float *hpOffsetVals;
	float *hpNormals;
} traceCache;

traceCache * allocateTraceCache(nnLayer * layer);
void freeTraceCache(traceCache * tc);

traceMemory * allocateTraceMB(int m, int n);
void freeTraceMB(traceMemory * tc);

void fullTrace(traceCache * tc, traceMemory * tm, float * point, float * dists, kint * intersections);
void batchFullTrace(traceCache * tc, float * point, float * dists, kint * intersections, int numData, int numProc);

void ipCalc(traceCache * tc, traceMemory * tm, float * point, kint *ipSig, float threshold);
void batchIpCalc(traceCache * tc, float * data, float * dists, kint * ipSigs, float threshold, int numData, int numProc);

#endif