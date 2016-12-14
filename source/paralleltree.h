#ifndef _paralleltree_h
#define _paralleltree_h
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <error.h>
#include <unistd.h>




typedef struct TreeNode {
	char created;
	int value;

	pthread_spinlock_t dataspinlock;
	int data;

	pthread_spinlock_t smallspinlock;
	struct TreeNode *smallNode;
	pthread_spinlock_t bigspinlock;
	struct TreeNode *bigNode;
} TreeNode;

typedef struct Tree {
	unsigned int numNodes;
	unsigned int depth;
	TreeNode * root;
} Tree;

Tree * createTree(int depth);
void addVector(Tree *tree, int vector);
void addBatch(Tree * tree, int *vectors, int numVectors);
void printTree(Tree *tree);
void freeTree(Tree *tree);

#endif