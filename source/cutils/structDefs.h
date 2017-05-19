
#ifndef _structDefs_h
#define _structDefs_h
#include <unistd.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>
#include <mkl_cblas.h>
#include <mkl_blas.h>
#include <mkl_lapack.h>
#include <mkl_lapacke.h>

#include <float.h>
#include "key.h"



typedef struct nnLayer {
	MKL_INT outDim;
	MKL_INT inDim;
	float *A;
	float *b;
} nnLayer;

typedef struct ipCacheData {
	float *solution;
	float *projection;
} ipCacheData;

typedef struct TreeNode {
	uint createdKL;

	pthread_spinlock_t keyspinlock;
	kint *key;

	pthread_mutex_t dataspinlock;
	int dataModifiedCount;
	ipCacheData * dataPointer;

	pthread_mutex_t smallspinlock;
	struct TreeNode *smallNode;
	pthread_mutex_t bigspinlock;
	struct TreeNode *bigNode;
} TreeNode;

typedef struct Tree {
	// Tree properties
	unsigned int keyLength;
	pthread_spinlock_t nodecountspinlock;
	unsigned int numNodes;
	unsigned int depth;
	int maxIPRank;
	TreeNode ** root;
} Tree;

typedef struct ipCache {
	nnLayer *layer;
	Tree *bases;
	float *hpOffsetVecs;
	float *hpNormals;
	float threshold;
	int depthRestriction;
	pthread_mutex_t balanceLock;
	int maxNodesBeforeTrim;
	int maxNodesAfterTrim;
} ipCache;



#endif