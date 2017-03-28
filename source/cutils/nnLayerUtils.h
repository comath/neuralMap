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


typedef struct nnLayer {
	long long int outDim;
	long long int inDim;
	float *A;
	float *b;
} nnLayer;

nnLayer * createLayer(float *A, float *b, int outDim, int inDim);
void freeLayer(nnLayer * layer);
void evalLayer(nnLayer * layer, float *x, float *output);
void printFloatArr(float * arr, uint length);
void printMatrix(float * arr, uint inDim, uint outDim);
void getRegSig(nnLayer *layer, float *p, kint * regSig);
void getRegSigBatch(nnLayer *layer, float *data, kint *regSig, uint numData, uint numProc);

#endif