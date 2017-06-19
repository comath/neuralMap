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

// Internal struct, contains the index of a hyperplane and the distance to that hyperplane
typedef struct distanceWithIndex {
	int index;
	float dist;
} distanceWithIndex;

/*
Memory buffer for the trace calculation. 

This is alloacted per thread and should not be shared.
*/
typedef struct traceMemory {
	float * permA;
	float * pointBuff;
	float * tau;
	distanceWithIndex * distances;
	float * interDists;
} traceMemory;

/*
Contains the information shared between the threads. 
The hpNormals/offsets contains the precomputed information to speed up 
computation of the distance between the points and the hyperplanes.

The solution contains a point that lies on the maximal rank intersection. 
As we're restricted to the situation were inDim > outDim this is valid.

todo: (though this is quite large) Change the "solution" from a single solution to a tree containing all solutions to relax the restrition
*/
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

/*
Computes the intersection poset trace for a point

Input: point, tc, tm
Output: traceDistances, traceIntersections
*/
void fullTrace(traceCache * tc, traceMemory * tm, float * point, float * traceDists, kint * traceIntersections);
void batchFullTrace(traceCache * tc, float * point, float * dists, kint * intersections, int numData, int numProc);

/*
Computes the associated intersection for a point

Input: point, tc, tm, threshold
Output: ipSig
*/
void ipCalc(traceCache * tc, traceMemory * tm, float * point, float threshold, kint *ipSig);
void batchIpCalc(traceCache * tc, float * data, float threshold, kint * ipSigs, int numData, int numProc);

/*
Computes both of the above, though we have the ordering that can be formed into intersection sigs, not the sigs themselves

Inputs: tc, tm, point, threshold
Outputs, ipSig, traceDists, traceRaws (the ordering needed to form the traceIntersections)
*/
void bothIPCalcTrace(traceCache * tc, traceMemory * tm, float * point, float threshold, kint *ipSig, float * traceDists, int *traceRaw);


#endif