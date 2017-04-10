#include "nnLayerUtils.h"
#include <stdio.h>
#include <float.h>
#include <stdlib.h>
#include <stdint.h>


#include <mkl.h>
#include <mkl_cblas.h>
#include <mkl_blas.h>
#include <mkl_lapack.h>
#include <mkl_lapacke.h>



nnLayer * createLayer(float * A, float *b, uint inDim, uint outDim)
{
	//Allocating A and b together
	nnLayer * layer = malloc(sizeof(nnLayer));
	layer->A = A;
	layer->b = b;
	layer->inDim = inDim;
	layer->outDim = outDim;
	return layer;
}

void freeLayer(nnLayer *layer)
{	
	//As we allocated A and b together there needs to only be 1 free operation, not 2
	if(layer){
		if(layer->A){
			free(layer->A);
		}
		free(layer);
	}
}

void evalLayer(nnLayer *layer, float * input, float * output)
{
	uint inDim = layer->inDim;
	uint outDim = layer->outDim;
	#ifdef DEBUG
		if(inDim < 10 && outDim < 20){
			printMatrix(layer->A,layer->inDim,layer->outDim);
			printFloatArr(layer->b,layer->outDim);
		}
	#endif
	cblas_scopy (outDim, layer->b, 1, output, 1);
	cblas_sgemv (CblasRowMajor, CblasTrans, inDim,outDim,1, layer->A, inDim, input, 1, 1, output, 1);
}

void printFloatArr(float *arr, uint length){
	uint i = 0;
	printf("{");
	if(length>0){
		for(i=0;i<length-1;i++){
			if(arr[i] == FLT_MAX){
				printf("---,");
			} else {
				printf("%f,",arr[i]);
			}
			
		}	
		if(arr[length-1] == FLT_MAX){
			printf("---,");
		} else {
			printf("%f",arr[length-1]);
		}
	}
	printf("}\n");
}

void printMatrix(float *arr, uint inDim, uint outDim){
	uint i = 0;
	for(i=0;i<outDim;i++){
		printFloatArr(arr + inDim*i,inDim);
	}
}
	

void getRegSig(nnLayer *layer, float *p, kint * regSig)
{
	// Prepare all the internal values and place commonly referenced ones on the stack
	uint outDim = layer->outDim;
	uint keyLength = calcKeyLen(outDim);
	clearKey(regSig, keyLength);
	float *output = malloc(outDim*sizeof(float));
	evalLayer(layer, p,output);
	convertFloatToKey(output,regSig,outDim);
}

struct getRegThreadArgs {
	uint tid;
	uint numThreads;

	uint numData;
	float * data;
	kint *regSig;

	nnLayer *layer;
};

void * getRegBatch_thread(void *thread_args)
{
	struct getRegThreadArgs *myargs;
	myargs = (struct getRegThreadArgs *) thread_args;

	uint tid = myargs->tid;	
	uint numThreads = myargs->numThreads;

	uint numData = myargs->numData;
	float *data = myargs->data;
	kint *regSig = myargs->regSig;

	nnLayer *layer = myargs->layer;

	uint dim = layer->inDim;
	uint keySize = calcKeyLen(layer->outDim);

	uint i = 0;
	for(i=tid;i<numData;i=i+numThreads){		
		getRegSig(layer,data+i*dim, regSig+i*keySize);
	}
	pthread_exit(NULL);
}

void getRegSigBatch(nnLayer *layer, float *data, kint *regSig, uint numData, uint numProc)
{
	int maxThreads = numProc;
	int rc =0;
	int i =0;
	//printf("Number of processors: %d\n",maxThreads);
	//Add one data to the first node so that we can avoid the race condition.
	

	struct getRegThreadArgs *thread_args = malloc(maxThreads*sizeof(struct getRegThreadArgs));

	pthread_t threads[maxThreads];
	pthread_attr_t attr;
	void *status;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	
	for(i=0;i<maxThreads;i++){
		thread_args[i].layer = layer;
		thread_args[i].numData = numData;
		thread_args[i].regSig = regSig;
		thread_args[i].data = data;
		thread_args[i].numThreads = maxThreads;
		thread_args[i].tid = i;
		rc = pthread_create(&threads[i], NULL, getRegBatch_thread, (void *)&thread_args[i]);
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