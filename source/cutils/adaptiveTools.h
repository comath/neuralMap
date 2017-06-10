#ifndef _refineMap_h
#define _refineMap_h
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "mapperTree.h"
#include "nnLayerUtils.h"
#include "key.h"



typedef struct maxPopGroupData {
	mapTreeNode ** locations;
	kint * hpCrossed;
	int count;
	int selectionIndex;
} maxPopGroupData;

maxPopGroupData * refineMapAndGetMax(mapTreeNode ** locArr, int maxLocIndex, nnLayer * selectionLayer);
vector * getRegSigs(mapTreeNode ** locArr, int numNodes);
void unpackRegSigs(vector * regSigs, uint dim, float * unpackedSigs);

#endif