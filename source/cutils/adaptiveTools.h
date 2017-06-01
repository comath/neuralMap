#ifndef _refineMap_h
#define _refineMap_h
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "vector.h"
#include "mapperTree.h"

typedef struct maxPopGroupData {
	vector locations;
	kint * hpCrossed;
	int count;
	int selectionIndex;
} maxPopGroupData;

maxPopGroupData * refineMap(mapTreeNode ** locArr, int maxLocIndex, float *selectionMat, float selectionBias);
void getMaxPopulation(vector * refinedMap, maxPopGroupData * output);


#endif