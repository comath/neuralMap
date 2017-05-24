#include "mapper.h"
#include <float.h>
#include <stdint.h>

#include <mkl.h>
#include <mkl_cblas.h>
#include <mkl_blas.h>
#include <mkl_lapack.h>
#include <mkl_lapacke.h>

_nnMap * allocateMap(nnLayer *layer0, float threshold, float errorThreshold)
{
	_nnMap *map = malloc(sizeof(_nnMap));
	uint keyLength = calcKeyLen(layer0->outDim);
	// The data is stored lexographically by (ipSig,regSig) as one long key. Thus keyLen has to double
	map->locationTree = createTree(16, 2*keyLength, mapperDataCreator,mapperDataModifier,mapperDataDestroy);
	map->layer0 = layer0;
	map->cache = allocateCache(map->layer0, threshold);
	map->errorThreshhold = errorThreshhold;
	return map;
}
void freeMap(_nnMap *map)
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

void addPointToMap(_nnMap * map, traceCache *tc, float *point, int pointIndex, float threshold)
{
	uint inDim = map->layer0->inDim;
	uint outDim = map->layer0->outDim;
	uint keyLength = calcKeyLen(outDim);
	traceMemory tm = allocateTraceMB(map->layer->outDim, map->layer->inDim);
	pointInfo 

	// The data is stored lexographically by (ipSig,regSig) as one long key.
	// This can be easily achieved with some pointer arithmetic
	pointInfo *pi = allocPointInfo(outDim);
	mapMemory *mm = allocMapMemory(outDim);
	addPointToMapInternal(map,mm,tc,tm,pi,point,pointIndex,threshold);
	freeMapMemory(mm);
}

void addPointToMapInternal(_nnMap * map, mapMemory *mm, traceCache *tc, traceMemory *tm, pointInfo *pi
							float *point, int pointIndex, float threshold)
{
	uint outDim = map->layer->outDim;

	// The data is stored lexographically by (ipSig,regSig) as one long key.
	// This can be easily achieved with some pointer arithmetic
	
	// Get the IP Signature
	bothIPCalcTrace(tc,tm,point,threshold, mm->keyPair, pi->traceDists, int pi->traceRaw);
	pi->pointIndex = pointIndex;

	// Get the Region Signature, save it offset by keyLength
	evalLayer(map->layer0, point, mm->outOfLayer);
	convertFloatToKey(outOfLayer0, mm->keyPair + keyLength,outDim);

	addMapData(map->locationTree, mm->keyPair, pi);
}

struct mapperAddThreadArgs {
	uint tid;
	uint numThreads;

	uint numData;
	_nnMap * map;
	traceCache *tc;
	traceMemory *tm;
	float *data;
	float *errorMargins;
};

void * addMapperBatch_thread(void *thread_args)
{
	struct mapperAddThreadArgs *myargs;
	myargs = (struct mapperAddThreadArgs *) thread_args;

	uint tid = myargs->tid;	
	uint numThreads = myargs->numThreads;

	uint numData = myargs->numData;
	_nnMap * map = myargs->map;

	
	float * errorMargins = myargs->errorMargins;
	float * data = myargs->data;
		
	uint dim = map->layer0->inDim;
	
	uint i = 0;
	for(i=tid;i<numData;i=i+numThreads){		
		addDatumToMap(map, myargs->tc,myargs->tm, data + i*dim, errorMargins[i]);
	}
	pthread_exit(NULL);
}

void addDataToMapBatch(_nnMap * map, traceCache *tc, float *data, float * errorMargins, uint numData, uint numProc)
{
	int maxThreads = numProc;
	int rc =0;
	int i =0;
	

	struct mapperAddThreadArgs *thread_args = malloc(maxThreads*sizeof(struct mapperAddThreadArgs));

	pthread_t threads[maxThreads];
	pthread_attr_t attr;
	void *status;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	for(i=0;i<maxThreads;i++){
		thread_args[i].numThreads = maxThreads;
		thread_args[i].tid = i;
		thread_args[i].tc = tc;
		thread_args[i].tm = allocateTraceMB(map->layer->outDim, map->layer->inDim);
		thread_args[i].numData = numData;
		thread_args[i].map = map;
		thread_args[i].data = data;
		thread_args[i].errorMargins = errorMargins;
		
		rc = pthread_create(&threads[i], NULL, addMapperBatch_thread, (void *)&thread_args[i]);
		if (rc){
			printf("Error, unable to create thread\n");
			exit(-1);
		}
	}

	for( i=0; i < maxThreads; i++ ){
		rc = pthread_join(threads[i], &status);
		if (rc){
			printf("Error, unable to join: %d \n", rc);
			exit(-1);
     	}
	}
	for(i=0;i<maxThreads;i++){
		freeTraceMB(thread_args[i].tm);
	}
	free(thread_args);
}



unsigned int numLoc(_nnMap * map)
{
	return map->locationTree->numNodes;
}



/*
location * getLocationArray(_nnMap * map)
{	
	uint numLoc = map->locationTree->numNodes;
	location * locArr = malloc(numLoc * sizeof(location));
	#ifdef DEBUG
		printf("Root node is %p, numNodes is %u\n",map->locationTree->root, numLoc);
	#endif
	traverseLocationSubtree(map, locArr, map->locationTree->root);
	return locArr;
}
*/