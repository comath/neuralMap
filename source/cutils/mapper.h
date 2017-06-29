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

// Given a layer it initalizes the traceCache, and preps the mapTree
_nnMap * allocateMap(nnLayer * layer);
void freeMap(_nnMap *map);

/*
Adds points to the map. You have to provide the points, the index of those points, 
a 0 or 1 for if you want to classify it as an error, and a threshold for the intersection calculation
We save the trace & the point index

todo: Make saving the trace optional, it's not needed adaptive backprop
*/
void addPointToMap(_nnMap * map, float *point, int pointIndex, int errorClass, float threshold);
void addPointsToMapBatch(_nnMap * map, float *data, int *indexes, int *errorClasses, float threshold, uint numData, uint numProc);


// Given a bit packed representation of (intersection set, region set) returns the saved details of that location
location getPointsAt(_nnMap *map, kint *keyPair);

// Returns the number of locations
unsigned int numLoc(_nnMap * map);

/*
Gives an array of pointers to the internal nodes, this can be used to unpack the map for use in adaptive backprop

todo: unpack this into a database efficiently, and provide a way to rebuild the map from this database 
*/
mapTreeNode ** getLocations(_nnMap *map, char orderBy);

// Tool exposed for the adaptive tools. Can be used with qsort to reorder to a regKey primary order.
int regOrder(const void * a, const void * b);





#endif