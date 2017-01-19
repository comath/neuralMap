/*
This just tests that there's no race conditions, or memory leaks.
*/
#include <stdio.h>
#include <stdlib.h>
#include "../utils/nnLayerUtils.h"
#include "../utils/ipCalculator.h"

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

void randomizePoint(float *p, uint dim)
{
	int i = 0;
	for(i=0;i<dim;i++){
		p[i] = (double)((rand() % 20000) - 10000)/10000;
	}
}

float * randomData(uint dim, uint numData)
{
	float * data = malloc(dim*numData*sizeof(float));
	int i;
	for(i=0;i<numData;i++){
		randomizePoint(data +dim*i,dim);
	}
	return data;
}




int main(int argc, char* argv[])
{
	uint dim = 3;
	uint numHP = 3;
	uint numData = 100;
	uint keySize = calcKeyLen(numHP);
	uint maxThreads = sysconf(_SC_NPROCESSORS_ONLN);

	srand(time(NULL));
	int i = 0;
	printf("If no faliures are printed then we are fine.\n");
	nnLayer *layer = createDumbLayer(3,3);
	ipCache *cache = allocateCache(layer,2);
	

	uint *ipSignature = malloc(keySize*numData*sizeof(uint));
	float *data = randomData(dim,numData);

	printf("Calculating the signature of Points\n");

	getInterSigBatch(cache, data, ipSignature, numData, maxThreads);

	for(i = 0; i < numData; i++){
		printFloatArrNoNewLine(data+i*dim,dim);
		printf(" has ip sig: ");
		printKeyArr(ipSignature+i*keySize,keySize);
	}

	free(data);
	free(ipSignature);
	freeLayer(layer);
	freeCache(cache);
	return 0;
}