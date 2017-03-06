#include "mapper.h"
#include <float.h>

#include <mkl.h>
#include <mkl_cblas.h>
#include <mkl_blas.h>
#include <mkl_lapack.h>
#include <mkl_lapacke.h>

typedef struct mapperInput {
	float * point;
	uint dim;
	float errorMargin;
	float errorThreshhold;
} mapperInput;

typedef struct mapperData {
	float numPoints;
	float numErrorPoints;
	float *avgPoint;
	float *avgErrorPoint;
} mapperData;

void * mapperDataCreator(void * input)
{
	struct mapperInput *myInput;
	myInput = (struct mapperInput *) input;
	uint dim = myInput->dim;

	mapperData * arrayData = malloc(sizeof(mapperData));
	arrayData->avgPoint = malloc(sizeof(float)* dim * 2);
	arrayData->avgErrorPoint = arrayData->avgPoint + dim;
	arrayData->numPoints = 0;
	arrayData->numErrorPoints = 0;
	
	return arrayData;
}

void mapperDataModifier(void * input, void * data)
{
	struct mapperInput *myInput;
	myInput = (struct mapperInput *) input;
	uint dim = myInput->dim;
	struct mapperData *myData;
	myData = (struct mapperData *) data;

	if(myData->numPoints == 0) {
		cblas_scopy (dim, myInput->point, 1, myData->avgPoint, 1);
		myData->numPoints = 1.0f;
	} else {
		cblas_saxpy (dim, 1/myData->numPoints, myInput->point, 1, myData->avgPoint, 1);
		cblas_sscal (dim, myData->numPoints/(myData->numPoints+1), myData->avgPoint, 1);
		myData->numPoints++;
	}
	if(myInput->errorMargin > myInput->errorThreshhold){
		if(myData->numErrorPoints == 0.0){
			cblas_scopy (dim,myInput->point, 1, myData->avgErrorPoint, 1);
			myData->numErrorPoints = 1.0f;
		} else {
			cblas_saxpy (dim, 1/myData->numErrorPoints, myInput->point, 1, myData->avgErrorPoint, 1);
			cblas_sscal (dim, myData->numErrorPoints/(myData->numErrorPoints+1), myData->avgErrorPoint, 1);
			myData->numErrorPoints++;
		}
	}
}

void mapperDataDestroy(void * data)
{	
	struct mapperData *myData;
	myData = (struct mapperData *) data;
	if(myData){
		free(myData->avgPoint);
		free(myData);
	} else {
		printf("Invalid Free\n");
	}
}

_nnMap * allocateMap(nnLayer *layer0, float threshhold, float errorThreshhold)
{
	_nnMap *map = malloc(sizeof(_nnMap));
	uint keyLength = calcKeyLen(layer0->outDim);
	// The data is stored lexographically by (ipSig,regSig) as one long key. Thus keyLen has to double
	map->locationTree = createTree(16, 2*keyLength, mapperDataCreator,mapperDataModifier,mapperDataDestroy);
	map->layer0 = layer0;
	map->cache = allocateCache(map->layer0, threshhold);
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

void addDatumToMap(_nnMap * map, float *datum, float errorMargin)
{
	uint inDim = map->layer0->inDim;
	uint outDim = map->layer0->outDim;
	uint keyLength = calcKeyLen(outDim);
	float * outOfLayer0 = malloc(outDim* sizeof(float));

	// The data is stored lexographically by (ipSig,regSig) as one long key.
	// This can be easily achieved with some pointer arithmetic
	uint * sig  = malloc(2*keyLength*sizeof(uint));
	
	// Get the IP Signature
	getInterSig(map->cache, datum, sig);
	// Get the Region Signature, save it offset by keyLength
	evalLayer(map->layer0, datum, outOfLayer0);
	convertFloatToKey(outOfLayer0,sig + keyLength,outDim);

	mapperInput inputStruct;
	inputStruct.point = datum;
	inputStruct.errorMargin = errorMargin;
	inputStruct.dim = inDim;
	inputStruct.errorThreshhold = map->errorThreshhold;
	
	addData(map->locationTree, sig, &inputStruct);
}

struct mapperAddThreadArgs {
	uint tid;
	uint numThreads;

	uint numData;
	_nnMap * map;
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
		addDatumToMap(map, data + i*dim, errorMargins[i]);
	}
	pthread_exit(NULL);
}

void addDataToMapBatch(_nnMap * map, float *data, float * errorMargins, uint numData, uint numProc)
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

	free(thread_args);
}

unsigned int numLoc(_nnMap * map)
{
	return map->locationTree->numNodes;
}

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

void traverseLocationSubtree(_nnMap * map, location * locArr, TreeNode *node)
{
	#ifdef DEBUG
		printf("Working on node %p node\n",node);
	#endif
	int i = 0;
	int nodeDepth = map->locationTree->depth;
	node = node - (1 << nodeDepth) + 1;
	struct mapperData *myData = NULL;
	int n = (1 << (nodeDepth+1)) - 1;



	for(i=0;i<n;i++){
		
		if(i%2==0 && node[i].smallNode){
			traverseLocationSubtree(map, locArr, node[i].smallNode);
		}		
		if(node[i].dataPointer && node[i].created){
			locArr->ipSig = node[i].key;
			locArr->regSig = node[i].key + map->locationTree->keyLength;
			myData = (struct mapperData *) node[i].dataPointer;
			
			locArr->numPoints = myData->numPoints;
			locArr->numErrorPoints = myData->numErrorPoints;
			locArr->avgPoint = myData->avgPoint;
			locArr->avgErrorPoint = myData->avgErrorPoint;
			locArr++;
		}
		if(i%2==0 && node[i].bigNode){
			traverseLocationSubtree(map, locArr, node[i].bigNode);
		}
	}	
}

