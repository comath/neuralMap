#ifndef _nnLayerUtils_h
#define _nnLayerUtils_h
#include <stdint.h>
#include "key.h"
#ifdef MKL
#include <mkl.h>
#include <mkl_cblas.h>
#include <mkl_blas.h>
#include <mkl_lapack.h>
#include <mkl_lapacke.h>
#endif

/*
Struct for passing layers of a perceptron.
*/
typedef struct nnLayer {
	uint outDim;
	uint inDim;
	float *A;
	float *b;
} nnLayer;

// Tools for dealing with a layer
nnLayer *allocateLayer(uint inDim, uint outDim);
nnLayer *createCopyLayer(float *A, float *b, uint outDim, uint inDim);
nnLayer *createLayer(float *A, float *b, uint outDim, uint inDim);
void freeLayer(nnLayer * layer);

// Computes Ax+b and puts the result in the output
void evalLayer(nnLayer * layer, float *x, float *output);
void printFloatArr(float * arr, uint length);
void printMatrix(float * arr, uint inDim, uint outDim);
void printLayer(nnLayer * layer);


/*
Computes Ax+b then puts it through the heavyside function and bitpacks it, put the bitpacking in regSig. 
I.E. Computes the associated region set for a point
*/
void getRegSig(nnLayer *layer, float *p, kint * regSig);
void getRegSigBatch(nnLayer *layer, float *data, kint *regSig, uint numData, uint numProc);

#endif