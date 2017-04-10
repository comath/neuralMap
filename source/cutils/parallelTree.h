#ifndef _paralleltree_h
#define _paralleltree_h
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <error.h>
#include <unistd.h>
#include <stdint.h>

#include "key.h"

typedef struct TreeNode {
	int createdKL;

	pthread_spinlock_t keyspinlock;
	kint *key;

	pthread_mutex_t datamutex;
	int dataModifiedCount;
	void * dataPointer;
	long int memoryUsage;

	pthread_spinlock_t smallspinlock;
	struct TreeNode *smallNode;
	pthread_spinlock_t bigspinlock;
	struct TreeNode *bigNode;
} TreeNode;



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
	unsigned int depth;
	TreeNode ** root;
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