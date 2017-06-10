#ifndef _paralleltree_h
#define _paralleltree_h
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <error.h>
#include <unistd.h>
#include <stdint.h>

#include "key.h"
#include "location.h"


#ifndef NODEDEPTH
#define NODEDEPTH 8
#define SUBTREESIZE ((1 << (NODEDEPTH+1)) - 1)
#define SUBCENTER ((1 << NODEDEPTH) - 1)
#endif


typedef struct mapTreeNode {
	int createdKL;
	kint *ipKey;
	kint *regKey;

	pthread_mutex_t datamutex;
	location loc;
} mapTreeNode;

typedef struct mapSubTree {
	pthread_mutex_t traverseMutexLock;

	mapTreeNode nodes[SUBTREESIZE];
	int nodeCount;

	struct mapSubTree *nextSubTrees[SUBTREESIZE+1];
} mapSubTree;


typedef struct mapTree {
	// Data handling function pointers

	int outDim;

	// Tree properties
	unsigned int keyLength;
	pthread_spinlock_t nodeCountSpinLock;
	int numNodes;
	long int currentMemoryUseage;
	
	mapSubTree * root;
} mapTree;

mapTree * createMapTree(int outDim);
void freeMapTree(mapTree *tree);

location * addMapData(mapTree *tree, kint * keyPair, pointInfo *pi);
location * getMapData(mapTree *tree, kint * keyPair);

mapTreeNode ** getAllNodes(mapTree * tree);

void nodeGetIPKey(mapTreeNode * node, int * ipKeyuncompressed, uint outDim);
void nodeGetRegKey(mapTreeNode * node, int * regKeyuncompressed, uint outDim);
void nodeGetPointIndexes(mapTreeNode * node, int *indexHolder);
int nodeGetTotal(mapTreeNode * node, int errorClass);

#endif