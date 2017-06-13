#ifndef _refineMap_h
#define _refineMap_h
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "mapperTree.h"
#include "mapper.h"
#include "nnLayerUtils.h"
#include "key.h"
#include "vector.h"
#include <stdint.h>
#include <limits.h>
#include <string.h>
#include <mkl.h>
#include <mkl_cblas.h>
#include <mkl_blas.h>
#include <mkl_lapack.h>
#include <mkl_lapacke.h>


typedef struct maxPopGroupData {
	mapTreeNode ** locations;
	kint * hpCrossed;
	int count;
	int selectionIndex;
} maxPopGroupData;
void freeMaxPopGroupData(maxPopGroupData * group);


maxPopGroupData * refineMapAndGetMax(mapTreeNode ** locArr, int maxLocIndex, nnLayer * selectionLayer);

void createNewHPVec(maxPopGroupData * maxErrorGroup, float * avgError, float *solution, float *newHPVec, float *offset, float *A, float *b, uint inDim, uint outDim);

vector * getRegSigs(mapTreeNode ** locArr, int numNodes);
void unpackRegSigs(vector * regSigs, uint dim, float * unpackedSigs);


#endif