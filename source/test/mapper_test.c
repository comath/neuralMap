/*
This just tests that there's no race conditions, or memory leaks. 
Running Valgrind or gdb on the python wrap leads to a bit more headaches.
*/
#include <stdio.h>
#include <stdlib.h>
#include "../cutils/nnLayerUtils.h"
#include "../cutils/mapper.h"
#include "../cutils/adaptiveTools.h"
#include "../cutils/selectionTrainer.h"



void printFloatArrNoNewLine(float * arr, int numElements){
	int i = 0;
	printf("[");
	for(i=0;i<numElements-1;i++){
		printf("%f,",arr[i]);
	}
	printf("%f", arr[numElements-1]);
	printf("]");
}

//Creates a layer with HPs parrallel to the cordinate axes
nnLayer *createDumbLayer(uint dim, uint numHP)
{
	nnLayer * layer = calloc(1,sizeof(nnLayer));
	layer->A = calloc((dim+1)*numHP,sizeof(float));
	layer->b = layer->A + dim*numHP;
	uint i,j;
	for(i=0;i<numHP;i++){
		for(j=0;j<dim;j++){
			if(i==j){
				layer->A[i*dim + j] = 1;
			} else {
				layer->A[i*dim + j] = 0;
			}
		}
		layer->b[i] = 0;
	}
	layer->inDim = dim;
	layer->outDim = numHP;
	return layer;
}

//Creates a layer with HPs parrallel to the cordinate axes
nnLayer *createDumbSelectionLayer(uint numHP, uint selection)
{
	nnLayer * layer = calloc(1,sizeof(nnLayer));
	layer->A = calloc((numHP+1)*numHP,sizeof(float));
	layer->b = layer->A + numHP*numHP;
	uint i,j;
	for(i=0;i<selection;i++){
		for(j=0;j<numHP;j++){
			if(i==j){
				layer->A[i*numHP + j] = 1;
			} else {
				layer->A[i*numHP + j] = 1;
			}
		}
		layer->b[i] = 0.5-(float)numHP;
	}
	layer->inDim = numHP;
	layer->outDim = selection;
	return layer;
}

void randomizePoint(float *p, uint dim)
{
	uint i = 0;
	for(i=0;i<dim;i++){
		p[i] = (double)((rand() % 20000) - 10000)/10000;
	}
}

float * randomData(uint dim, uint numData)
{
	float * data = malloc(dim*numData*sizeof(float));
	uint i;
	for(i=0;i<numData;i++){
		randomizePoint(data +dim*i,dim);
	}
	return data;
}


int main(int argc, char* argv[])
{
	uint dim = 9;
	uint numHP = 3;
	uint finalDim = 1;

	uint numData = 400;
	uint keySize = calcKeyLen(numHP);
	uint maxThreads = sysconf(_SC_NPROCESSORS_ONLN);

	srand(time(NULL));
	printf("If no faliures are printed then we are fine.\n");
	nnLayer *layer0 = createDumbLayer(dim,numHP);
	nnLayer *layer1 = createDumbSelectionLayer(numHP, finalDim);
	_nnMap *map = allocateMap(layer0);

	uint *ipSignature = malloc(keySize*numData*sizeof(uint));
	float *data = randomData(dim,numData);
	int * indexes = malloc(numData*sizeof(int));
	int * errorClasses = malloc(numData*sizeof(int));

	for(uint i =0;i<numData;i++){
		indexes[i] = i;
		errorClasses[i] = i%2;
	} 

	printf("Calculating the signature of %d Points\n",numData);
	addPointsToMapBatch(map,data,indexes,errorClasses,2,numData,4);

	mapTreeNode ** locations = getLocations(map, 'i');
	int maxLocIndex = numLoc(map);

	maxErrorCorner * max = refineMapAndGetMax(locations, maxLocIndex, layer1);
	float * avgError = malloc(dim*sizeof(float));
	getAverageError(max, data, avgError, dim);

	float * solution = getSolutionPointer(map);
	float * newHPVec = malloc((dim+1)*sizeof(float));
	createNewHPVec(max, avgError, solution, layer0, newHPVec, newHPVec+dim);


	
/*
	int dataLength = 2*vector_total(vecRegKeys);
	float *unpackedSigs = malloc(2*dataLength*(numHP+1)*sizeof(float));
	int *labels = malloc(2*dataLength*sizeof(int));
	createData(max, layer1, vecRegKeys,unpackedSigs,labels);
*/
	float * newweight = malloc((numHP+1)*finalDim*sizeof(float));
	float * newbias = malloc((finalDim)*sizeof(float));

	trainNewSelector(layer1, locations, maxLocIndex, max,newweight,newbias);



	freeMaxErrorCorner(max);
	free(locations);
	free(ipSignature);
	free(indexes);
	freeMap(map);
	free(data);
	free(errorClasses);


	return 0;
}