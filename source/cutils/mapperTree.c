#include "mapperTree.h"
#include <stdint.h>
#include <string.h>


mapSubTree * allocateMapNodes(uint keyLength)
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
		tree->nodes[i].createdKL = 0;
		tree->nodes[i].ipKey = keys + 2*i*keyLength;
		tree->nodes[i].regKey = keys + (2*i+1)*keyLength;


		rc = pthread_mutex_init(&(tree->nodes[i].datamutex), 0);
		if (rc != 0) {
	        printf("mutex Initialization failed at %p", (void *) tree + i);
	    }
	}
	memset(tree->nextSubTrees,0, (SUBTREESIZE+1)*sizeof(mapSubTree *));
	return tree;
}

void freeMapNodes(mapTree *tree, mapSubTree *st)
{
	int i = 0;
	// Free the keyspace.
	free(st->nodes[0].ipKey);
	int n = (1 << (NODEDEPTH+1)) - 1;

	for(i=0;i<n;i=i+2){
		if(st->nextSubTrees[i]){
			freeMapNodes(tree,st->nextSubTrees[i]);
		}
		if(st->nextSubTrees[i+1]){
			freeMapNodes(tree,st->nextSubTrees[i+1]);
		}
	}
	for(i=0;i<n;i++){
		if(st->nodes[i].createdKL)
			location_free(&(st->nodes[i].loc));
		pthread_mutex_destroy(&(st->nodes[i].datamutex));
	}

	free(st);
}

mapTree * createMapTree(int outDim){
	mapTree * tree = malloc(sizeof(mapTree));
	tree->outDim = outDim;
	uint keyLength = calcKeyLen(outDim);

	tree->root = allocateMapNodes(keyLength);
	int rc = pthread_spin_init(&(tree->nodeCountSpinLock), 0);
	if (rc != 0) {
        printf("spinlock Initialization failed at %p", (void *) tree);
    }
	tree->numNodes = 0;
	tree->keyLength = keyLength;

	tree->currentMemoryUseage = 0;
	return tree;	
}

void freeMapTree(mapTree *tree)
{
	if(tree){
		freeMapNodes(tree,tree->root);
		
		pthread_spin_destroy(&(tree->nodeCountSpinLock));
		free(tree);
	}
}

/* 
We are comparing the key pairs, ipKey and the regKey. 
They are stored in memory in that order so we can compare them as a single key.
*/

location * addMapData(mapTree *tree, kint * keyPair, pointInfo *pi){
	mapSubTree * st = tree->root;
	int keyLen = tree->keyLength;
	int i = SUBCENTER;
	int d = NODEDEPTH;
	
	pthread_mutex_lock(&(st->traverseMutexLock));

	if (st->nodes[i].createdKL == 0){
		st->nodes[i].createdKL = keyLen;
		copyKey(keyPair, st->nodes[i].ipKey, 2*keyLen);
		location_init(&(st->nodes[i].loc),tree->outDim);
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
				st->nextSubTrees[i] = allocateMapNodes(keyLen);
				pthread_mutex_unlock(&(st->traverseMutexLock));
				st = st->nextSubTrees[i];
				pthread_mutex_lock(&(st->traverseMutexLock));

				i = SUBCENTER;
				d = NODEDEPTH;
				(st->nodes[i]).createdKL = keyLen;
				copyKey(keyPair, st->nodes[i].ipKey, 2*keyLen);
				location_init(&(st->nodes[i].loc),tree->outDim);
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
				location_init(&(st->nodes[i].loc), tree->outDim);
				pthread_spin_lock(&(tree->nodeCountSpinLock));
					tree->numNodes++;
				pthread_spin_unlock(&(tree->nodeCountSpinLock));
			} 
		}
	}
	pthread_mutex_unlock(&(st->traverseMutexLock));
	pthread_mutex_lock(&(st->nodes[i].datamutex));
		location_add(&(st->nodes[i].loc), pi);
	pthread_mutex_unlock(&(st->nodes[i].datamutex));
	return &(st->nodes[i].loc);
}

location * getMapData(mapTree *tree, kint * keyPair){
	mapSubTree * st =	 tree->root;
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
	return &(st->nodes[i].loc);
}

void traverseSubtree(mapTreeNode *(*(*traversePointer)), mapSubTree *st)
{
	int i = 0;
	//printf("-----traverseSubtree-----\n");
	for(i=0;i<SUBTREESIZE;i++){
		
		if(i%2==0 && st->nextSubTrees[i]){
			//printf("---callingsmall--\n");
			traverseSubtree(traversePointer, st->nextSubTrees[i]);
		}
		if( st->nodes[i].createdKL){
			/*
			printf("Node access: %p, ",st->nodes+i);
			printf("dataPointer: %p, ", st->nodes[i].dataPointer);
			printf("with access count %d, ", st->nodes[i].accessCount);
			printf("memory weight: %d. ", st->nodes[i].memoryUsage);
			printf("key[0]: %lu\n ", st->nodes[i].key[0]);
			*/
				*(*traversePointer) = st->nodes + i;
				(*traversePointer)++;
		}
		if(i%2==0 && st->nextSubTrees[i+1]){
			//printf("---callingBig--\n");
			traverseSubtree(traversePointer, st->nextSubTrees[i+1]);
		}
	}
	//printf("---//traverseSubtree//--\n");
}

mapTreeNode ** getAllNodes(mapTree * tree)
{
	mapTreeNode *(*nodePointerArr) = malloc(tree->numNodes*sizeof(mapTreeNode *));
	printf("%p %lu %d %lu\n", nodePointerArr,tree->numNodes*sizeof(mapTreeNode *),tree->numNodes, sizeof(mapTreeNode *));
	mapTreeNode *(*traversePointer) = nodePointerArr;
	traverseSubtree(&traversePointer,tree->root);
	
	if(traversePointer - nodePointerArr != tree->numNodes){
		printf("Node Count Off! %ld, should be %d\n", traversePointer - nodePointerArr,tree->numNodes);
		exit(-1);
	}
	return nodePointerArr;
}