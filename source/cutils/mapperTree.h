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

/*
We allocate blocks of 2^NODEDEPTH-1 nodes at a time. 

We also store the nodes in order, not in a heap
*/
#ifndef NODEDEPTH
#define NODEDEPTH 8
#define SUBTREESIZE ((1 << (NODEDEPTH+1)) - 1)
#define SUBCENTER ((1 << NODEDEPTH) - 1)
#endif

/*
The individual nodes. The ipKey, regKey is saved in a single array of length 2*keyLength. 
2 pointers are provided for code clarity.
*/
typedef struct mapTreeNode {
	int createdKL; // 0 if not initalized, keyLength if it has been filled
	kint *ipKey; // Key to intersection set, bit packed. See key.h
	kint *regKey; // Key to region set, bit packed

	pthread_mutex_t datamutex;
	location loc;
} mapTreeNode;

/*
A subtree, allocated to an array of width SUBTREESIZE. 
All nodes at most 8 edge distance are initalized to 0 at allocation. 
Once we reach a leaf we allocate a new tree.
We store them in order (not in a heap), with the root being in the center.
*/
typedef struct mapSubTree {
	pthread_mutex_t traverseMutexLock;

	mapTreeNode nodes[SUBTREESIZE];
	int nodeCount;

	struct mapSubTree *nextSubTrees[SUBTREESIZE+1];
} mapSubTree;

/*
Main tree struct. Contains the basic info needed for future operations.
*/
typedef struct mapTree {
	// Tree properties
	unsigned int keyLength;
	pthread_spinlock_t nodeCountSpinLock;
	int numNodes;
	int outDim;
	
	// Tree
	mapSubTree * root;
} mapTree;

// Allocate/free functions. Need to provide the number of hyperplanes we're using
mapTree * createMapTree(int outDim);
void freeMapTree(mapTree *tree);

/*
Adds data to the tree. 

The information needs to be packed into a pointInfo struct. See location.h
*/
location * addMapData(mapTree *tree, kint * keyPair, pointInfo *pi);
location * getMapData(mapTree *tree, kint * keyPair);

// Makes a array of all the nodes in the mapTree
mapTreeNode ** getAllNodes(mapTree * tree);

// used by the cython wrapper to retrieve the information in the tree nodes
void nodeGetIPKey(mapTreeNode * node, int * ipKeyuncompressed, uint outDim);
void nodeGetRegKey(mapTreeNode * node, int * regKeyuncompressed, uint outDim);
void nodeGetPointIndexes(mapTreeNode * node, int errorClass, int *indexHolder);
int nodeGetTotal(mapTreeNode * node, int errorClass);

#endif