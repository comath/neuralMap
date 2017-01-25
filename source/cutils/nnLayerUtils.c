#include "nnLayerUtils.h"
#include <stdio.h>
#include <stdlib.h>
#ifdef MKL
#include <mkl.h>
#include <mkl_cblas.h>
#include <mkl_blas.h>
#include <mkl_lapack.h>
#include <mkl_lapacke.h>
#endif

#ifndef MKL
#define MKL_INT const int
#endif

nnLayer * createLayer(float * A, float *b, uint inDim, uint outDim)
{
	nnLayer * layer = malloc(sizeof(nnLayer));
	layer->A = malloc((inDim*outDim + outDim)*sizeof(float));
	layer->b = layer->A + inDim*outDim;
	cblas_scopy (inDim*outDim, A, 1, layer->A, 1);
	cblas_scopy (outDim, b, 1, layer->b, 1);
	layer->inDim = inDim;
	layer->outDim = outDim;
	return layer;
}

void freeLayer(nnLayer *layer)
{
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
	cblas_scopy (outDim, layer->b, 1, output, 1);
	cblas_sgemv (CblasRowMajor, CblasTrans, outDim, inDim,1, layer->A, inDim, input, 1, 1, output, 1);
}

void printFloatArr(float *arr, uint length){
	uint i = 0;
	printf("[");
	for(i=0;i<length-1;i++){
		printf("%f,",arr[i]);
	}	
	printf("%f", arr[length-1]);
	printf("]\n");
}

void printMatrix(float *arr, uint inDim, uint outDim){
	uint i = 0;
	for(i=0;i<outDim;i++){
		printFloatArr(arr + inDim*i,inDim);
	}
}
	

