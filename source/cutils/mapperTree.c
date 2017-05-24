#include "parallelTree.h"
#include <stdint.h>
#include <string.h>


mapSubTree * allocateNodes(uint keyLength)
{
	
	mapSubTree * tree = malloc(sizeof(mapSubTree));
	// To store both ip keys an reg keys.
	kint * keys = malloc(2 * SUBTREESIZE * keyLength * sizeof(kint));
	int rc = 0;
	rc = pthread_mutex_init(&(tree->traverseMutexLock), 0);
	if (rc != 0) {
        printf("spinlock Initialization failed at %p", (void *) tree);
    }
	for (int i = 0; i < SUBTREESIZE; i++)
	{
		tree->nodes[i].accessCount = 0;
		tree->nodes[i].createdKL = 0;
		tree->nodes[i].ipKey = keys + 2*i*keyLength;
		tree->nodes[i].regKey = keys + (2*i+1)*keyLength;

		tree->nodes[i].points = NULL;

		rc = pthread_mutex_init(&(tree->nodes[i].datamutex), 0);
		if (rc != 0) {
	        printf("mutex Initialization failed at %p", (void *) tree + i);
	    }
	}
	memset(tree->nextSubTrees,0, (SUBTREESIZE+1)*sizeof(mapSubTree *));
	return tree;
}

void freeMapNodes(mapTree *tree, SubTree *st)
{
	int i = 0;
	// Free the keyspace.
	free(st->nodes[0].key);
	int n = (1 << (NODEDEPTH+1)) - 1;

	for(i=0;i<n;i=i+2){
		if(st->nextSubTrees[i]){
			freeNodes(tree,st->nextSubTrees[i]);
		}
		if(st->nextSubTrees[i+1]){
			freeNodes(tree,st->nextSubTrees[i+1]);
		}
	}
	for(i=0;i<n;i++){
		if(st->nodes[i].dataPointer){
			tree->dataDestroy(st->nodes[i].dataPointer);
		}
		pthread_mutex_destroy(&(st->nodes[i].datamutex));
	}

	free(st);
}

mapTree * createMapTree(uint keyLength, long int maxTreeMemory){
	mapTree * tree = malloc(sizeof(mapTree));
	tree->numTrees = numTrees;
	tree->root = malloc(numTrees*sizeof(mapSubTree *));
	for(uint i=0;i<numTrees;i++){
		tree->root[i] = allocateNodes(keyLength);
	}
	int rc = pthread_spin_init(&(tree->nodeCountSpinLock), 0);
	if (rc != 0) {
        printf("spinlock Initialization failed at %p", (void *) tree);
    }
	tree->numNodes = 0;
	tree->keyLength = keyLength;

	tree->currentMemoryUseage = 0;
	tree->maxDatumMemory = maxDatumMemory;
	tree->maxTreeMemory = maxTreeMemory;
	return tree;	
}

void freeMapTree(mapTree *tree)
{
	if(tree){
		freeMapNodes(tree,tree->root);
		
		free(tree->root);
		pthread_spin_destroy(&(tree->nodeCountSpinLock));
		free(tree);
	}
}

/* 
We are comparing the key pairs, ipKey and the regKey. 
They are stored in memory in that order so we can compare them as a single key.
*/

vector * addMapData(mapTree *tree, kint * keyPair, pointInfo *pi){
	SubTree * st =	 tree->root[treeIndex];
	int keyLen = tree->keyLength;
	int i = SUBCENTER;
	int d = NODEDEPTH;
	
	pthread_mutex_lock(&(st->traverseMutexLock));

	if (st->nodes[i].createdKL == 0){
		st->nodes[i].createdKL = keyLen;
		copyKey(keyPair, st->nodes[i].ipKey, 2*keyLen);
		vector_init(points);
		tree->numNodes++;
	}

	
	int keyCompare;
	while((keyCompare = compareKey(keyPair,st->nodes[i].ipKey,2*keyLen))){
		if(d == 0){
			if(keyCompare == 1){
				i = i + 1;
			} else {
				i = i;
			}
			if(st->nextSubTrees[i] == NULL){
				st->nextSubTrees[i] = allocateNodes(keyLen);
				pthread_mutex_unlock(&(st->traverseMutexLock));
				st = st->nextSubTrees[i];
				pthread_mutex_lock(&(st->traverseMutexLock));

				i = SUBCENTER;
				d = NODEDEPTH;
				(st->nodes[i]).createdKL = keyLen;
				copyKey(keyPair, st->nodes[i].ipKey, 2*keyLen);
				vector_init(points);
				pthread_spin_lock(&(tree->nodeCountSpinLock));
					tree->numNodes++;
				pthread_spin_unlock(&(tree->nodeCountSpinLock));
			} else {
				pthread_mutex_unlock(&(st->traverseMutexLock));
				st = st->nextSubTrees[i];
				pthread_mutex_lock(&(st->traverseMutexLock));

				i = SUBCENTER;
				d = NODEDEPTH;
			}
			
		} else {
			d--;
			if(keyCompare == 1){
				i = i + (1 << d);
			} else {
				i = i - (1 << d);
			}
			if((st->nodes[i]).createdKL == 0){
				(st->nodes[i]).createdKL = keyLen;
				copyKey(keyPair, st->nodes[i].ipKey, 2*keyLen);
				(st->nodes[i]).dataPointer = tree->dataCreator(datum);
				pthread_spin_lock(&(tree->nodeCountSpinLock));
					tree->numNodes++;
				pthread_spin_unlock(&(tree->nodeCountSpinLock));
			} 
		}
	}
	pthread_mutex_unlock(&(st->traverseMutexLock));
	pthread_mutex_lock(&(st->nodes[i].datamutex));
		vector_add(points, pi);
	pthread_mutex_unlock(&(st->nodes[i].datamutex));
	return (void *) st->nodes[i].points;
}

vector * getMapData(mapTree *tree, kint * keyPair){
	SubTree * st =	 tree->root[treeIndex];
	int keyLen = tree->keyLength;
	int i = SUBCENTER;
	int d = NODEDEPTH;

	if (st->nodes[i].createdKL == 0){
		return NULL;
	}

	int keyCompare;
	while((keyCompare = compareKey(keyPair,st->nodes[i].ipKey,2*keyLen))){
		if(d == 0){
			if(keyCompare == 1){
				i = i + 1;
			} else {
				i = i;
			}
			if(st->nextSubTrees[i] == NULL){
				return NULL;
			} else {
				st = st->nextSubTrees[i];

				i = SUBCENTER;
				d = NODEDEPTH;
			}
			
		} else {
			d--;
			if(keyCompare == 1){
				i = i + (1 << d);
			} else {
				i = i - (1 << d);
			}
			if((st->nodes[i]).createdKL == 0){
				return NULL;
			} 
		}
	}
	return (void *) st->nodes[i].points;
}

