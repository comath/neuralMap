#include "mapper.h"
#include <float.h>

#include <mkl.h>
#include <mkl_cblas.h>
#include <mkl_blas.h>
#include <mkl_lapack.h>
#include <mkl_lapacke.h>

typedef struct mapperInput {
	//This should be a constant
	const ipCache * info;
	//This shouldn't be
	float * point;
	char error;

} ipCacheInput;

typedef struct mapperData {
	uint numPoints;
	uint numErrorPoints;
	float *avgPoint;
	float *avgErrorPoint;
} ipCacheData;

void * mapperDataCreator(void * input)
{

}

void mapperDataModifier(void * input, void * data)
{

}

void mapperDataDestroy(void * data)
{
		
}

nnMap * allocateMap(nnLayer *layer0, nnLayer *layer1, float threshold)
{
	nnMap *map = malloc(sizeof(nnMap));
	uint keyLength = calcKeyLen(layer0->inDim);
	map->locationTree = createTree(16, keyLength, mapperDataCreator,mapperDataModifier,mapperDataDestroy);
	map->layer0 = copyLayer(layer0);
	map->layer1 = copyLayer(layer1);
	map->cache = allocateCache(map->layer0, threshhold);
}
void freeMap(nnMap *map)
{
	if(map){
		freeTree(map->locationTree);
		freeCache(map->cache);
		freeLayer(map->layer0);
		freeLayer(map->layer1);
		free(map);
	} else {
		printf("Map does not exist.\n");
	}
}

void addDatum(nnMap * map, float *datum);
void addData(nnMap *map, float *data, uint numData, uint numProc);

location getMaxErrorLoc(nnMap * map);
location * getLocationArray(nnMap * map);
location getLocationByKey(nnMap *map, uint * ipSig, uint * regSig);

