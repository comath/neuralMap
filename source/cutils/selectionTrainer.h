#ifndef _selectionTrainer_h
#define _selectionTrainer_h
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "nnLayerUtils.h"
#include "key.h"
#include "vector.h"
#include "adaptiveTools.h"
#include <stdint.h>
#include <limits.h>
#include <string.h>
#include <mkl.h>
#include <mkl_cblas.h>
#include <mkl_blas.h>
#include <mkl_lapack.h>
#include <mkl_lapacke.h>

#define INITSTEPSIZE 0.05
#define NUMEPOCHS 10000

void trainNewSelector(nnLayer *selectionLayer, mapTreeNode ** locArr, int locArrLen, maxErrorCorner *maxGroup, float *newSelectionWeights, float *newSelectionBias);

#endif