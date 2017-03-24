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
	uint createdKL;

	pthread_spinlock_t keyspinlock;
	kint *key;

	pthread_mutex_t dataspinlock;
	int dataModifiedCount;
	void * dataPointer;

	pthread_mutex_t smallspinlock;
	struct TreeNode *smallNode;
	pthread_mutex_t bigspinlock;
	struct TreeNode *bigNode;
} TreeNode;



typedef struct Tree {
	// Data handling function pointers
	void * (*dataCreator)(void * input);
	void (*dataModifier)(void * input, void * data);
	void (*dataDestroy)(void * data);

	// Tree properties
	unsigned int keyLength;
	pthread_spinlock_t nodecountspinlock;
	unsigned int numNodes;
	unsigned int depth;
	TreeNode * root;
} Tree;

Tree * createTree(int depth, uint keyLength, void * (*dataCreator)(void * input),
				void (*dataModifier)(void * input, void * data),void (*dataDestroy)(void * data));
void * addData(Tree *tree, kint *key, void * datum);
void * getData(Tree *tree, kint *key);
void freeTree(Tree *tree);
void balanceAndTrimTree(Tree *tree, int finalNodeCount);
int * getAccessCounts(Tree *tree);

#endif