/*
This just tests that there's no race conditions, or memory leaks. 
Running Valgrind or gdb on the python wrap leads to a bit more headaches.
*/
#include <stdio.h>
#include <stdlib.h>
#include "../cutils/nnLayerUtils.h"
#include "../cutils/ipCalculator.h"

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
				layer->A[i*dim + j] = 2;
			} else {
				layer->A[i*dim + j] = 0;
			}
		}
		layer->b[i] = 1;
	}
	layer->inDim = dim;
	layer->outDim = numHP;
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
	uint dim = 4;
	uint numHP = 5;
	uint numData = 5;
	uint keySize = calcKeyLen(numHP);
	uint maxThreads = sysconf(_SC_NPROCESSORS_ONLN);

	srand(time(NULL));
	uint i = 0;
	printf("If no faliures are printed then we are fine.\n");
	nnLayer *layer = createDumbLayer(dim,numHP);
	ipCache *cache = allocateCache(layer,2, 1000000000);
	

	kint *ipSignature = malloc(keySize*numData*sizeof(kint));
	float *data = randomData(dim,numData);

	printf("Calculating the signature of Points\n");

	getInterSigBatch(cache, data, ipSignature, numData, 1);

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