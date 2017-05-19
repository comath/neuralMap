#ifndef _mapper_h
#define _mapper_h
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "mapperTree.h"
#include "ipTrace.h"
#include "nnLayerUtils.h"
#include <stdint.h>

#include <mkl.h>
#include <mkl_cblas.h>
#include <mkl_blas.h>
#include <mkl_lapack.h>
#include <mkl_lapacke.h>

typedef struct pointInfo {
	int * traceRaw;
	float * traceDists;
	long int pointIndex;
}

typedef struct location {
	kint *ipSig;
	kint *regSig;
	vector *points;
} location;

typedef struct _nnMap {
	nnLayer *layer;
	MapTree *locationTree;
	traceCache *traceCache;
} _nnMap;

_nnMap * allocateMap(nnLayer *layer0, float threshold, float errorThreshhold);
void freeMap(_nnMap *map);

void addDatumToMap(_nnMap * map, float *datum);
void addDataToMapBatch(_nnMap * map, float *data, uint numData, uint numProc);

unsigned int numLoc(_nnMap * map);
location * getLocationArray(_nnMap * map);

#endif