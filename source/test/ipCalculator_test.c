/*
This just tests that there's no race conditions, or memory leaks.
*/
#include <stdio.h>
#include <stdlib.h>
#include "../utils/nnLayerUtils.h"
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
		p[i] = (double)((rand() % 200) - 100)/100;
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

struct dataAddThreadArgs {
	int tid;
	int numThreads;

	uint numData;
	float * data;
	uint *ipSignature;

	ipCache *cache;
};

void * addBatch_thread(void *thread_args)
{
	struct dataAddThreadArgs *myargs;
	myargs = (struct dataAddThreadArgs *) thread_args;

	int tid = myargs->tid;	
	int numThreads = myargs->numThreads;

	uint numData = myargs->numData;
	float *data = myargs->data;
	uint *ipSignature = myargs->ipSignature;

	ipCache *cache = myargs->cache;

	uint dim = cache->layer0->inDim;
	uint keySize = calcKeyLen(cache->layer0->outDim);

	int i = 0;
	for(i=tid;i<numData;i=i+numThreads){
		getInterSig(data+i*dim, ipSignature+i*keySize, cache);
	}
	pthread_exit(NULL);
}

void addBatch(ipCache *cache, float *data, uint *ipSignature, uint numData)
{
	int maxThreads = sysconf(_SC_NPROCESSORS_ONLN);
	int rc =0;
	int i =0;

	//Add one data to the first node so that we can avoid the race condition.
	

	struct dataAddThreadArgs *thread_args = malloc(maxThreads*sizeof(struct dataAddThreadArgs));

	pthread_t threads[maxThreads];
	pthread_attr_t attr;
	void *status;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	
	for(i=0;i<maxThreads;i++){
		thread_args[i].cache = cache;
		thread_args[i].numData = numData;
		thread_args[i].ipSignature = ipSignature;
		thread_args[i].data = data;
		thread_args[i].numThreads = maxThreads;
		thread_args[i].tid = i;
		rc = pthread_create(&threads[i], NULL, addBatch_thread, (void *)&thread_args[i]);
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

void printKeyArr(uint *key, uint length){
	int i = 0;
	printf("[");
	for(i=0;i<length;i++){
		printf("%u,",key[i]);
	}
	printf("%u", key[length]);
	printf("]\n");
}

int main(int argc, char* argv[])
{
	uint dim = 300;
	uint numHP = 200;
	uint numData = 2000;
	uint keySize = calcKeyLen(numHP);

	srand(time(NULL));
	int i = 0;
	printf("If no faliures are printed then we are fine.\n");
	nnLayer *layer = createDumbLayer(3,3);
	ipCache *cache = allocateCache(layer,2);
	

	uint *ipSignature = malloc(keySize*numData*sizeof(uint));
	float *data = randomData(dim,numData);

	printf("Calculating the signature of Points\n");

	addBatch(cache, data, ipSignature, numData);

	free(data);
	free(ipSignature);
	freeLayer(layer);
	freeCache(cache);
	return 0;
}