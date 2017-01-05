#include <stdio.h>
#include <stdlib.h>
#include "../utils/ipCalculator.h"


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
		layer->b[i] = 2;
	}
	layer->inDim = dim;
	layer->outDim = numHP;
	return layer;
}

void randomizePoint(float *p, uint dim)
{
	int i = 0;
	for(i=0;i<dim;i++){
		p[i] = (double)((rand() % 200) - 100)/50;
	}
}



int main(int argc, char* argv[])
{
	uint dim = 3;
	uint numHP = 3;
	uint keySize = 1;
	srand(time(NULL));
	int i = 0;
	printf("If no faliures are printed then we are fine.\n");
	nnLayer *layer = createDumbLayer(3,3);
	if(layer->inDim == dim){
		printf("Layer Successfully created.\n");
	}
	ipCache *cache = allocateCache(layer,2);
	if(cache->inDim == dim){
		printf("Cache Successfully created.\n");
	}
	uint *ipSignature = malloc(keySize*sizeof(uint));
	float *p = malloc(dim*sizeof(uint));
	randomizePoint(p,dim);

	getInterSig(p, ipSignature, cache);
	return 0;
}