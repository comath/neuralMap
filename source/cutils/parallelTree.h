/*
Used by ipCalculator. Currently not called as ipCalculator is not currently used.
*/

#ifndef _paralleltree_h
#define _paralleltree_h
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <error.h>
#include <unistd.h>
#include <stdint.h>

#include "key.h"


#ifndef NODEDEPTH
#define NODEDEPTH 8
#define SUBTREESIZE ((1 << (NODEDEPTH+1)) - 1)
#define SUBCENTER ((1 << NODEDEPTH) - 1)
#endif


typedef struct TreeNode {
	int createdKL;
	kint *key;

	pthread_mutex_t datamutex;
	int dataModifiedCount;
	void * dataPointer;
	long int memoryUsage;
	int accessCount;
} TreeNode;

typedef struct SubTree {
	pthread_mutex_t traverseMutexLock;

	TreeNode nodes[SUBTREESIZE];
	int nodeCount;

	struct SubTree *nextSubTrees[SUBTREESIZE+1];
} SubTree;


typedef struct Tree {
	// Data handling function pointers
	void * (*dataCreator)(void * input);
	void (*dataModifier)(void * input, void * data);
	void (*dataDestroy)(void * data);
	int maxDatumMemory;

	long int maxTreeMemory;

	// Tree properties
	unsigned int keyLength;
	pthread_spinlock_t nodeCountSpinLock;
	int numNodes;
	long int currentMemoryUseage;
	
	SubTree ** root;
	int numTrees;
} Tree;

Tree * createTree(uint keyLength, uint numTrees, 
				int maxDatumMemory, long int maxTreeMemory, 
				void * (*dataCreator)(void * input),
				void (*dataModifier)(void * input, void * data),
				void (*dataDestroy)(void * data));
void * addData(Tree *tree, kint *key, int treeIndex, void * datum, int memoryUsage);
void * getData(Tree *tree, kint *key, int treeIndex);
void balanceAndTrimTree(Tree *tree, long int memMax);
void freeTree(Tree *tree);


#endif