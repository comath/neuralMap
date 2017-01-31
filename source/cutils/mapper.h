#ifndef _mapper_h
#define _mapper_h
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "parallelTree.h"
#include "ipCalculator.h"
#include "nnLayerUtils.h"

#include <mkl.h>
#include <mkl_cblas.h>
#include <mkl_blas.h>
#include <mkl_lapack.h>
#include <mkl_lapacke.h>

typedef struct location {
	uint *ipSig;
	uint *regSig;
	uint numPoints;
	uint numErrorPoints;
	float *avgPoint;
	float *avgErrorPoint;
} location;

typedef struct nnMap {
	nnLayer *layer0;
	nnLayer *layer1;
	Tree *locationTree;
	ipCache *cache;
} nnMap;

nnMap * allocateMap(nnLayer *layer0, float threshold);
void freeMap(nnMap *map);

void addDatum(nnMap * map, float *datum);
void addData(nnMap *map, float *data, uint numData, uint numProc);

location getMaxErrorLoc(nnMap * map);
location * getLocationArray(nnMap * map);
location getLocationByKey(nnMap *map, uint * ipSig, uint * regSig);

#endif