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

typedef struct refinedLocation {
	location * subLocations;
	uint numSubLocations;
	uint numPoints;
	uint numErrorPoints;
} refinedLocation;


typedef struct _nnMap {
	nnLayer *layer0;
	nnLayer *layer1;
	Tree *locationTree;
	ipCache *cache;

	float errorThreshhold;
} _nnMap;

_nnMap * allocateMap(nnLayer *layer0, nnLayer *layer1, float threshold, float errorThreshhold);
void freeMap(_nnMap *map);

void addDatumToMap(_nnMap * map, float *datum, float errorMargin);
void addDataToMapBatch(_nnMap * map, float *data, float * errorMargins, uint numData, uint numProc);

unsigned int numLoc(_nnMap * map);
location * getLocationArray(_nnMap * map);

#endif