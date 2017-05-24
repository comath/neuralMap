#ifndef _paralleltree_h
#define _paralleltree_h
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <error.h>
#include <unistd.h>
#include <stdint.h>

#include "key.h"
#include "vector.h"


#ifndef NODEDEPTH
#define NODEDEPTH 8
#define SUBTREESIZE ((1 << (NODEDEPTH+1)) - 1)
#define SUBCENTER ((1 << NODEDEPTH) - 1)
#endif

typedef struct pointInfo {
	int * traceRaw;
	float * traceDists;
	long int pointIndex;
}

typedef struct location {
	kint *ipSig;
	kint *regSig;
	vector *points;
} location;

typedef struct mapTreeNode {
	int createdKL;
	kint *ipKey;
	kint *regKey;

	pthread_mutex_t datamutex;
	vector *points;
} TreeNode;

typedef struct mapSubTree {
	pthread_mutex_t traverseMutexLock;

	mapTreeNode nodes[SUBTREESIZE];
	int nodeCount;

	struct SubTree *nextSubTrees[SUBTREESIZE+1];
} SubTree;


typedef struct mapTree {
	// Data handling function pointers

	long int maxTreeMemory;

	// Tree properties
	unsigned int keyLength;
	pthread_spinlock_t nodeCountSpinLock;
	int numNodes;
	long int currentMemoryUseage;
	
	mapSubTree * root;
} Tree;

mapTree * createMapTree(uint keyLength, long int maxTreeMemory);
void freeMapTree(mapTree *tree);

pointInfo * allocPointInfo(int m);
void freePointInfor(pointInfo *pi);

vector * addMapData(mapTree *tree, kint * keyPair, pointInfo *pi);
vector * getMapData(mapTree *tree, kint * keyPair);

#endif