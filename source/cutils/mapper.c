#include "mapper.h"

_nnMap * allocateMap(nnLayer * layer)
{
	_nnMap *map = malloc(sizeof(_nnMap));
	// The data is stored lexographically by (ipSig,regSig) as one long key. Thus keyLen has to double
	map->locationTree = createMapTree(layer->outDim);
	map->layer = layer;
	map->tc = allocateTraceCache(map->layer);
	return map;
}
void freeMap(_nnMap *map)
{
	if(map){
		freeMapTree(map->locationTree);
		freeTraceCache(map->tc);
		free(map);
	} else {
		printf("Map does not exist.\n");
	}
}

typedef struct mapMemory
{
	float * outOfLayer;
	kint * keyPair;
	int keyLength;
} mapMemory;

mapMemory * allocMapMemory(int m)
{
	mapMemory * mm = malloc(sizeof(mapMemory));
	if(mm){
		mm->outOfLayer = malloc(m*sizeof(float));
		mm->keyLength = calcKeyLen(m);
		mm->keyPair = malloc(mm->keyLength * 2 * sizeof(kint));
		if(mm->outOfLayer && mm->keyPair){
			return mm;
		}
	}
	exit(-1);
}

void freeMapMemory(mapMemory *mm)
{
	if(mm){
		if(mm->outOfLayer){
			free(mm->outOfLayer);
		} 
		if(mm->keyPair){
			free(mm->keyPair);
		}
		free(mm);
	}
}

void addPointToMapInternal(_nnMap * map, mapMemory *mm, traceMemory *tm, pointInfo *pi,
							float *point, int pointIndex, int errorClass, float threshold)
{
	// The data is stored lexographically by (ipSig,regSig) as one long key.
	// This can be easily achieved with some pointer arithmetic
	
	// Get the IP Signature
	bothIPCalcTrace(map->tc,tm,point,threshold, mm->keyPair, pi->traceDists, pi->traceRaw);
	pi->index = pointIndex;
	pi->errorClass = errorClass;
	// Get the Region Signature, save it offset by keyLength
	evalLayer(map->layer, point, mm->outOfLayer);
	convertFromFloatToKey(mm->outOfLayer, mm->keyPair + mm->keyLength,map->layer->outDim);
	addMapData(map->locationTree, mm->keyPair, pi);
}

void addPointToMap(_nnMap * map, float *point, int pointIndex, int errorClass, float threshold)
{
	uint outDim = map->layer->outDim;
	traceMemory * tm = allocateTraceMB(outDim, map->layer->inDim);
	pointInfo *pi = allocPointInfo(outDim);
	mapMemory *mm = allocMapMemory(outDim);
	addPointToMapInternal(map,mm,tm,pi,point,pointIndex,errorClass,threshold);
	freeMapMemory(mm);
	freeTraceMB(tm);
	freePointInfo(pi);
}

struct mapperAddThreadArgs {
	uint tid;
	uint numThreads;

	uint numData;
	_nnMap * map;
	float *data;
	int *indexes;
	int *errorClasses;
	float threshold;
};

void * addMapperBatch_thread(void *thread_args)
{
	struct mapperAddThreadArgs *myargs;
	myargs = (struct mapperAddThreadArgs *) thread_args;

	uint tid = myargs->tid;	
	uint numThreads = myargs->numThreads;

	uint numData = myargs->numData;
	_nnMap * map = myargs->map;

	int m = map->layer->outDim;
	int n = map->layer->inDim;
		
	traceMemory *tm = allocateTraceMB(m,n);
	// The data is stored lexographically by (ipSig,regSig) as one long key.
	// This can be easily achieved with some pointer arithmetic
	pointInfo *pi = allocPointInfo(m);
	mapMemory *mm = allocMapMemory(m);
	
	uint i = 0;

	for(i=tid;i<numData;i=i+numThreads){
		addPointToMapInternal(map,mm,tm,pi,myargs->data + i*n,myargs->indexes[i],myargs->errorClasses[i],myargs->threshold);
	}
	freePointInfo(pi);
	freeMapMemory(mm);
	freeTraceMB(tm);
	pthread_exit(NULL);
}

void addPointsToMapBatch(_nnMap * map, float *data, int *indexes, int *errorClasses, float threshold, uint numData, uint numProc)
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
		thread_args[i].numData = numData;
		thread_args[i].map = map;
		thread_args[i].data = data;
		thread_args[i].threshold = threshold;
		thread_args[i].indexes = indexes;
		thread_args[i].errorClasses = errorClasses;

		
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
	free(thread_args);
}



unsigned int numLoc(_nnMap * map)
{
	return map->locationTree->numNodes;
}

int regOrder(const void * a, const void * b)
{
	struct mapTreeNode *myA = *(mapTreeNode * const *)a;
	struct mapTreeNode *myB = *(mapTreeNode * const *)b;
	int regCmp = compareKey(myA->regKey,myB->regKey,myB->createdKL);
	if(regCmp == 0){
		return (int)compareKey(myA->ipKey,myB->ipKey,myB->createdKL);
	}
	return regCmp;
}

mapTreeNode ** getLocations(_nnMap *map, char orderBy)
{	
	mapTreeNode ** locArr = getAllNodes(map->locationTree);
	
	if(orderBy == 'r'){
		qsort(locArr, map->locationTree->numNodes, sizeof(mapTreeNode *), regOrder);
	}
	return locArr;
}

// Given a bit packed representation of (intersection set, region set) returns the saved details of that location
// Currently very dumb, but it will combine all intersections within a region into a single region data in the future.
location * getPointsAt(_nnMap *map, kint *keyPair, char bothBool)
{
	return getMapData(map->locationTree, keyPair);
}

// Given a point returns the points near a given point and the keyPair. 
location * getPointsNear(_nnMap *map, float *point, kint *keyPair, int threshold)
{
	uint outDim = map->layer->outDim;
	traceMemory * tm = allocateTraceMB(outDim, map->layer->inDim);
	float * outBuff = malloc(outDim*sizeof(float));
	ipCalc(map->tc,tm, point, threshold, keyPair);
	evalLayer(map->layer, point, outBuff);
	convertFromFloatToKey(outBuff, keyPair + calcKeyLen(outDim),outDim);
	free(outBuff);
	freeTraceMB(tm);
	return getMapData(map->locationTree, keyPair);
}


// A HACK
void batchCheckRegionsNear(_nnMap *map, float *points, int numPoints, int * results)
{
	uint outDim = map->layer->outDim;
	uint inDim = map->layer->inDim;
	uint keyLen = calcKeyLen(outDim);
	kint * key = malloc(keyLen*sizeof(kint));

	float * outBuff = malloc(outDim*sizeof(float));
	
	int numLoc = map->locationTree->numNodes;
	memset(results,0,numPoints*sizeof(int));

	mapTreeNode ** loc = getLocations(map, 'r');
	for(int i = 0; i < numPoints;i++){
		evalLayer(map->layer, points + i*inDim, outBuff);
		convertFromFloatToKey(outBuff, key,outDim);
		for(int j = 0; j < numLoc; j++){
			if(compareKey(key,loc[j]->regKey,keyLen) == 0){
				results[i] = 1;
			}
		}
	}

	free(outBuff);
	free(key);
}

void batchCheckPointsNear(_nnMap *map, float *points, int numPoints, float threshold, int * results)
{
	uint outDim = map->layer->outDim;
	uint inDim = map->layer->inDim;

	uint keyLen = calcKeyLen(outDim);
	kint * keyPair = malloc(keyLen*sizeof(kint));

	traceMemory * tm = allocateTraceMB(outDim, map->layer->inDim);
	float * outBuff = malloc(outDim*sizeof(float));

	int numLoc = map->locationTree->numNodes;
	memset(results,0,numPoints*sizeof(int));

	for(int i = 0; i < numPoints;i++){
		ipCalc(map->tc,tm, points + i*inDim, threshold, keyPair);
		evalLayer(map->layer, points + i*inDim, outBuff);
		convertFromFloatToKey(outBuff, keyPair + calcKeyLen(outDim),outDim);
		if(getMapData(map->locationTree, keyPair) != NULL){
			results[i] = 1;
		}
	}

	free(outBuff);
	free(keyPair);
	freeTraceMB(tm);
}