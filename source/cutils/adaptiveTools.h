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


/*
Stores the needed information to make a new hyperplane
*/
typedef struct maxErrorCorner {
	mapTreeNode ** locations;
	kint * hpCrossed;
	int locCount;
	int weightedCount;
	int selectionIndex;
} maxErrorCorner;
void freemaxErrorCorner(maxErrorCorner * group);

/*
For a given selection matrix, this finds the corner that in a neural network that has the most error amoung all selection vectors. 
This does not form a database of all corners.
locArr has to be provided in ipSig primary order. (Lexicographically with ipKey first.)

Input: locArr, maxLocIndex (last index of locArr), selectionLayer
Returns: Info needed to make a new hyperplane.
*/
maxErrorCorner * refineMapAndGetMax(mapTreeNode ** locArr, int maxLocIndex, nnLayer * selectionLayer);
/*
Returns tha average error in the max-error-corner. This can be replaced by your own function if need be (if the data doesn't fit in ram or something). 
Data should be an array of floats that contain all datapoints.
Average error should be a floar buffer of length inDim for the hyperplane layer.

Input: maxErrorGroup, data
Output: averageError
*/
void getAverageError(maxErrorCorner * maxErrorGroup, float *data, float * avgError);
/*
Creates a hyperplane vector that points from the average error location to the corner. 
The newVec should be a float array of length inDim for the hpLayer, the newOff should be a 1 length float array

Inputs: maxErrorGroup, aveError, solution, hpLayer.
Outputs: newVec, newOff

*/
void createNewHPVec(maxErrorCorner * maxErrorGroup, float * avgError, float *solution, nnLayer *hpLayer, float *newVec, float *newOff);

/*
Collects the region signatures from the location array, also reorders to a region centric ordering. Returns a vector that contains all regions.
*/
vector * getRegSigs(mapTreeNode ** locArr, int numNodes);
/*
Creates a artificial dataset with which we can train the new selection vector.
Inputs: maxErrorGroup, selectionLayer, regSigs
Outputs: unpackedSigs, labels
*/
void createData(maxErrorCorner *maxErrorGroup, nnLayer *selectionLayer, vector *regSigs, float *unpackedSigs, float * labels);

//void unpackRegSigs(vector * regSigs, uint dim, float * unpackedSigs);

// Helper functions for cython.
float *getSolutionPointer(_nnMap *map);
int getSelectionIndex(maxErrorCorner * maxGroup);


#endif