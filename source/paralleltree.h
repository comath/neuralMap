#ifndef _paralleltree_h
#define _paralleltree_h
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <error.h>
#include <unistd.h>

typedef struct Key {
	unsigned char *key;
	int length;
} Key;

typedef struct TreeNode {
	char created;
	int value;

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
	unsigned int numNodes;
	unsigned int depth;
	TreeNode * root;
} Tree;

Tree * createTree(int depth, void * (*dataCreator)(void * input),
				void (*dataModifier)(void * input, void * data),void (*dataDestroy)(void * data));
void addData(Tree *tree, int key, void * datum);
//void printTree(Tree *tree);
void freeTree(Tree *tree);

#endif