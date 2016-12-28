#ifndef _paralleltree_h
#define _paralleltree_h
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <error.h>
#include <unistd.h>

#include "key.h"





typedef struct TreeNode {
	char created;
	uint *key;

	pthread_spinlock_t dataspinlock;
	int dataModifiedCount;
	void * dataPointer;

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

	// Tree properties
	unsigned int keyLength;
	unsigned int numNodes;
	unsigned int depth;
	TreeNode * root;
} Tree;

Tree * createTree(int depth, void * (*dataCreator)(void * input),
				void (*dataModifier)(void * input, void * data),void (*dataDestroy)(void * data));
void * addData(Tree *tree, Key *key, void * datum);
void * getData(Tree *tree,Key *key);
void freeTree(Tree *tree);

void convertToKey(int * raw, Key * key,int length);
void convertFromKey(Key * key, int * output, int length);
char compareKey(Key *x, Key *y);

#endif