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



typedef struct _nnMap {
	nnLayer *layer;
	mapTree *locationTree;
	traceCache *tc;
} _nnMap;



_nnMap * allocateMap(nnLayer *layer0);
void freeMap(_nnMap *map);

void addPointToMap(_nnMap * map, float *point, int pointIndex, int errorClass, float threshold);
void addDataToMapBatch(_nnMap * map, float *data, int *indexes, int *errorClasses, float threshold, uint numData, uint numProc);

location getPointsAt(_nnMap *map, kint *keyPair);

int regOrder(const void * a, const void * b);

unsigned int numLoc(_nnMap * map);
mapTreeNode ** getLocations(_nnMap *map, char orderBy);



#endif