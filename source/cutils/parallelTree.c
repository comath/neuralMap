#include "parallelTree.h"
#include <stdint.h>

#ifndef NODEDEPTH
#define NODEDEPTH 6 
#endif

void fillTreeNodesPointers(TreeNode *node, int nodeDepth)
{
	if(nodeDepth > -1) {
		node->smallNode = node - (1 << nodeDepth);
		node->bigNode = node + (1 << nodeDepth);
		fillTreeNodesPointers(node->smallNode,nodeDepth-1);
		fillTreeNodesPointers(node->bigNode,nodeDepth-1);
	} else {
		node->smallNode = NULL;
		node->bigNode = NULL;
	}
}

TreeNode * allocateNodes(uint keyLength)
{
	const int maxTreeSize = (1 << (NODEDEPTH+1)) - 1;
	struct TreeNode * tree = malloc(maxTreeSize * sizeof(TreeNode));
	kint * keys = malloc(maxTreeSize * keyLength * sizeof(kint));
	int rc = 0;
	for (int i = 0; i < maxTreeSize; i++)
	{
		tree[i].dataModifiedCount = 0;
		tree[i].createdKL = 0;
		tree[i].key = keys + i*keyLength;
		tree[i].dataPointer = NULL;

		rc = pthread_spin_init(&(tree[i].keyspinlock), 0);
		if (rc != 0) {
	        printf("spinlock Initialization failed at %p", (void *) tree + i);
	    }
		rc = pthread_spin_init(&(tree[i].smallspinlock), 0);
		if (rc != 0) {
	        printf("spinlock Initialization failed at %p", (void *) tree + i);
	    }
		rc = pthread_spin_init(&(tree[i].bigspinlock), 0);
		if (rc != 0) {
	        printf("spinlock Initialization failed at %p", (void *) tree + i);
	    }
		rc = pthread_mutex_init(&(tree[i].datamutex), 0);
		if (rc != 0) {
	        printf("mutex Initialization failed at %p", (void *) tree + i);
	    }
	}
	tree = tree + (1 << NODEDEPTH) - 1;
	fillTreeNodesPointers(tree,NODEDEPTH-1);
	return tree;
}

Tree * createTree(uint keyLength, uint numTrees, int maxDatumMemory, long int maxTreeMemory, void * (*dataCreator)(void * input),
				void (*dataModifier)(void * input, void * data),void (*dataDestroy)(void * data))
{
	Tree * tree = malloc(sizeof(Tree));
	tree->numTrees = numTrees;
	tree->root = malloc(numTrees*sizeof(TreeNode *));
	for(uint i=0;i<numTrees;i++){
		tree->root[i] = allocateNodes(keyLength);
	}
	int rc = pthread_spin_init(&(tree->nodeCountSpinLock), 0);
	if (rc != 0) {
        printf("spinlock Initialization failed at %p", (void *) tree);
    }
	tree->numNodes = 0;
	tree->keyLength = keyLength;
	tree->dataCreator = dataCreator;
	tree->dataModifier = dataModifier;
	tree->dataDestroy = dataDestroy;

	tree->currentMemoryUseage = 0;
	tree->maxDatumMemory = maxDatumMemory;
	tree->maxTreeMemory = maxTreeMemory;
	return tree;	
}


void * addData(Tree *tree, kint * key, int treeIndex, void * datum, int memoryUsage){
	TreeNode * node = tree->root[treeIndex];
	int keyLen = tree->keyLength;
	pthread_spin_lock(&(node->keyspinlock));
	if (node->createdKL == 0){
		node->createdKL = 1;
		copyKey(key, node->key, keyLen);
		node->dataPointer = tree->dataCreator(datum);
		node->memoryUsage = memoryUsage;
		tree->numNodes++;
	}
	int keyCompare = compareKey(key,node->key,keyLen);
	pthread_spin_unlock(&(node->keyspinlock));
	while(keyCompare){
		if (keyCompare == 1) {
			pthread_spin_lock(&(node->bigspinlock));
			if(node->bigNode == NULL){
				node->bigNode = allocateNodes(keyLen);
				(node->bigNode)->createdKL = keyLen;
				copyKey(key, (node->bigNode)->key, keyLen);
				(node->bigNode)->dataPointer = tree->dataCreator(datum);
				(node->bigNode)->memoryUsage = memoryUsage;
				pthread_spin_lock(&(tree->nodeCountSpinLock));
					tree->numNodes++;
					tree->currentMemoryUseage += memoryUsage;
				pthread_spin_unlock(&(tree->nodeCountSpinLock));
			} 
			if ((node->bigNode)->createdKL == 0){
				(node->bigNode)->createdKL = 1;
				copyKey(key, (node->bigNode)->key, keyLen);
				(node->bigNode)->dataPointer = tree->dataCreator(datum);
				(node->bigNode)->memoryUsage = memoryUsage;
				pthread_spin_lock(&(tree->nodeCountSpinLock));
					tree->numNodes++;
					tree->currentMemoryUseage += memoryUsage;
				pthread_spin_unlock(&(tree->nodeCountSpinLock));
			}
			pthread_spin_unlock(&(node->bigspinlock));
			node = node->bigNode;
		} else if (keyCompare == -1) {
			pthread_spin_lock(&(node->smallspinlock));
			if(node->smallNode == NULL){
				node->smallNode = allocateNodes(keyLen);
				(node->smallNode)->createdKL = 1;
				copyKey(key, (node->smallNode)->key, keyLen);
				(node->smallNode)->dataPointer = tree->dataCreator(datum);
				(node->smallNode)->memoryUsage = memoryUsage;
				pthread_spin_lock(&(tree->nodeCountSpinLock));
					tree->numNodes++;
					tree->currentMemoryUseage += memoryUsage;
				pthread_spin_unlock(&(tree->nodeCountSpinLock));
			} 
			if ((node->smallNode)->createdKL == 0){
				(node->smallNode)->createdKL = 1;
				copyKey(key, (node->smallNode)->key, keyLen);
				(node->smallNode)->dataPointer = tree->dataCreator(datum);
				(node->smallNode)->memoryUsage = memoryUsage;
				pthread_spin_lock(&(tree->nodeCountSpinLock));
					tree->numNodes++;
					tree->currentMemoryUseage += memoryUsage;
				pthread_spin_unlock(&(tree->nodeCountSpinLock));
			}
			pthread_spin_unlock(&(node->smallspinlock));
			node = node->smallNode;
		}
		pthread_spin_lock(&(node->keyspinlock));
		keyCompare = compareKey(key,node->key,keyLen);
		pthread_spin_unlock(&(node->keyspinlock));
	}

	pthread_mutex_lock(&(node->datamutex));
		if(tree->dataModifier){
			tree->dataModifier(datum,node->dataPointer);
		}
		node->dataModifiedCount++;
	pthread_mutex_unlock(&(node->datamutex));
	return (void *) node->dataPointer;
}

void * getData(Tree *tree, kint *key, int treeIndex)
{
	int keyLen = tree->keyLength;
	TreeNode * node = tree->root[treeIndex];
	if (node->createdKL == 0){
		return NULL;
	}
	char comparison;
	while((comparison = compareKey(key,node->key,keyLen))){

		if (comparison == 1) {
			pthread_spin_lock(&(node->bigspinlock));
			if(node->bigNode == NULL || node->bigNode->createdKL == 0){
				return NULL;
			}
			pthread_spin_unlock(&(node->bigspinlock));
			node = node->bigNode;
		} else if (comparison == -1) {
			pthread_spin_lock(&(node->smallspinlock));
			if(node->smallNode == NULL || node->smallNode->createdKL == 0){
				return NULL;
			}
			pthread_spin_unlock(&(node->smallspinlock));
			node = node->smallNode;
		}
		
	}
	return (void *) node->dataPointer;
}


void freeNodes(Tree *tree, TreeNode *node)
{
	int i = 0;
	node = node - (1 << NODEDEPTH) + 1;
	// Free the keyspace.
	free(node->key);
	int n = (1 << (NODEDEPTH+1)) - 1;

	for(i=0;i<n;i=i+2){
		if(node[i].bigNode){
			freeNodes(tree,node[i].bigNode);
		}
		if(node[i].smallNode){
			freeNodes(tree,node[i].smallNode);
		}
	}
	for(i=0;i<n;i++){
		if(node[i].dataPointer){
			tree->dataDestroy(node[i].dataPointer);
		}
		pthread_spin_destroy(&(node[i].bigspinlock));
		pthread_spin_destroy(&(node[i].smallspinlock));
		pthread_mutex_destroy(&(node[i].datamutex));
	}

	free(node);
}

void freeTree(Tree *tree)
{
	if(tree){
		for(int i=0;i<tree->numTrees;i++){
			freeNodes(tree,tree->root[i]);
		}
		free(tree->root);
		pthread_spin_destroy(&(tree->nodeCountSpinLock));
		free(tree);
	}
}

void moveNode(Tree *tree, TreeNode * oldNode, int treeIndex){
	TreeNode * newNode = tree->root[treeIndex];
	
	if (newNode->createdKL == 0){
		newNode->createdKL = tree->keyLength;
		copyKey(oldNode->key, newNode->key, tree->keyLength);
		newNode->dataPointer = oldNode->dataPointer;
		oldNode->dataPointer = NULL;
		tree->numNodes++;
	}
	int keyCompare = compareKey(oldNode->key,newNode->key,tree->keyLength);

	while(keyCompare){
		if (keyCompare == 1) {
			if(newNode->bigNode == NULL){
				newNode->bigNode = allocateNodes(tree->keyLength);
				(newNode->bigNode)->createdKL = tree->keyLength;
				copyKey(oldNode->key, (newNode->bigNode)->key, tree->keyLength);
				(newNode->bigNode)->dataPointer = oldNode->dataPointer;
				oldNode->dataPointer = NULL;
				tree->numNodes++;
				tree->currentMemoryUseage += oldNode->memoryUsage;
			} 
			if ((newNode->bigNode)->createdKL == 0){
				(newNode->bigNode)->createdKL = tree->keyLength;
				copyKey(oldNode->key, (newNode->bigNode)->key, tree->keyLength);
				(newNode->bigNode)->dataPointer = oldNode->dataPointer;
				oldNode->dataPointer = NULL;
				tree->numNodes++;
				tree->currentMemoryUseage += oldNode->memoryUsage;
			}
			newNode = newNode->bigNode;
		} else if (keyCompare == -1) {
			if(newNode->smallNode == NULL){
				newNode->smallNode = allocateNodes(tree->keyLength);
				(newNode->smallNode)->createdKL = tree->keyLength;
				copyKey(oldNode->key, (newNode->smallNode)->key, tree->keyLength);
				(newNode->smallNode)->dataPointer = oldNode->dataPointer;
				oldNode->dataPointer = NULL;
				tree->numNodes++;
				tree->currentMemoryUseage += oldNode->memoryUsage;
			} 
			if ((newNode->smallNode)->createdKL == 0){
				(newNode->smallNode)->createdKL = tree->keyLength;
				copyKey(oldNode->key, (newNode->smallNode)->key, tree->keyLength);
				(newNode->smallNode)->dataPointer = oldNode->dataPointer;
				oldNode->dataPointer = NULL;
				tree->numNodes++;
				tree->currentMemoryUseage += oldNode->memoryUsage;
			}
			newNode = newNode->smallNode;
		}
		keyCompare = compareKey(oldNode->key,newNode->key,tree->keyLength);
	}
	newNode->dataModifiedCount = oldNode->dataModifiedCount;
	newNode->memoryUsage = oldNode->memoryUsage;
}

void traverseSubtree(TreeNode *(*(*traversePointer)), TreeNode *node, int minAccess, int maxMemory)
{
	int i = 0;
	node = node - (1 << NODEDEPTH) + 1;
	int n = (1 << (NODEDEPTH+1)) - 1;
	printf("-----traverseSubtree-----\n");
	for(i=0;i<n;i++){
		
		if(i%2==0 && node[i].smallNode){
			printf("---callingsmall--\n");
			traverseSubtree(traversePointer, node[i].smallNode,minAccess,maxMemory);
		}
		if(node[i].dataPointer && node[i].createdKL && 
			(minAccess<0 || node[i].dataModifiedCount >= minAccess)){
			printf("Node access: %p, ",node+i);
			printf("dataPointer: %p, ", node[i].dataPointer);
			printf("with access count %d, ", (node+i)->dataModifiedCount);
			printf("memory weight: %d. ", node[i].memoryUsage);
			printf("key[0]: %lu\n ", node[i].key[0]);
			if(minAccess<0 || node[i].dataModifiedCount*node[i].dataModifiedCount > minAccess || maxMemory > node[i].memoryUsage){
				*(*traversePointer) = node+i;
				(*traversePointer)++;
			} 
		}
		if(i%2==0 && node[i].bigNode){
			printf("---callingBig--\n");
			traverseSubtree(traversePointer, node[i].bigNode,minAccess,maxMemory);
		}
	}
	printf("---//traverseSubtree//--\n");
}

int intergrateMemoryUseageTilMax(TreeNode ** nodes, int nodeCount, long int memMax)
{
	long int curMem = 0;
	int i = 0;
	while(curMem<memMax && i < nodeCount){
		curMem += nodes[i]->memoryUsage;
		i++;
	}
	return i;
}

TreeNode ** getAllNodes(Tree * tree)
{
	TreeNode *(*nodePointerArr) = malloc(tree->numNodes*sizeof(TreeNode *));
	printf("%p %lu %d %lu\n", nodePointerArr,tree->numNodes*sizeof(TreeNode *),tree->numNodes, sizeof(TreeNode *));
	TreeNode *(*traversePointer) = nodePointerArr;
	for(int i = 0;i<tree->numTrees;i++){
		printf("===============================On Rank %d=====================================\n", i);
		traverseSubtree(&traversePointer,tree->root[i],-1,0);
	}
	if(traversePointer - nodePointerArr != tree->numNodes){
		printf("Node Count Off! %ld, should be %d\n", traversePointer - nodePointerArr,tree->numNodes);
		exit(-1);
	}
	return nodePointerArr;
}

int accessMemoryOrderingCmp (const void * a, const void * b)
{
	struct TreeNode *myA = *(TreeNode * const *)a;
	struct TreeNode *myB = *(TreeNode * const *)b;
	int comp = ((myB)->dataModifiedCount - (myA)->dataModifiedCount);
	if(comp == 0){
		return (myA)->memoryUsage-(myB)->memoryUsage;
	} 
	return comp;
}

int keyOrderingCmp(const void * a, const void * b)
{
	struct TreeNode *myA = *(TreeNode * const *)a;
	struct TreeNode *myB = *(TreeNode * const *)b;
	return (int)(compareKey(myA->key,myB->key,myB->createdKL));
}

void recursiveMoveNode(TreeNode ** nodeArr, Tree *tree, int recursionDepth, int treeIndex)
{
	int k = (1 << (recursionDepth))-1;
	moveNode(tree, nodeArr[k],treeIndex);
	if(recursionDepth > 0){
		recursiveMoveNode(nodeArr, tree, recursionDepth-1, treeIndex);
		recursiveMoveNode(nodeArr + k, tree, recursionDepth-1, treeIndex);
	}
}

void balanceAndTrimTree(Tree *tree, long int memMax)
{
	printf("Balancing tree and trimming to %ld memoryUsage\n", memMax);

	int finalNodeCount = 0;
	int oldNodeCount = tree->numNodes;
	
	if(memMax < oldNodeCount){
		finalNodeCount = memMax;
	} else {
		finalNodeCount = oldNodeCount;
	}
	
	TreeNode ** nodeArr = getAllNodes(tree);
	TreeNode ** traversePointer;
	tree->numNodes = 0;
	tree->currentMemoryUseage = 0;
	qsort(nodeArr, oldNodeCount, sizeof(TreeNode *), accessMemoryOrderingCmp);
	int lastNodeIndex = intergrateMemoryUseageTilMax(nodeArr, oldNodeCount, memMax);
	int minAccess = nodeArr[lastNodeIndex-1]->dataModifiedCount;
	int maxMemory = nodeArr[lastNodeIndex-1]->memoryUsage;
	
	printf("-----------Obtained the minAccess, maxMemory: %d,%d-----------\n",minAccess, maxMemory);

	int i=0,j=0,k=0;
	int batchConstant = (1 << (NODEDEPTH+1))-1;

	for(i = 0;i<tree->numTrees;i++){
		TreeNode * oldRoot = tree->root[i];
		TreeNode * newRoot = allocateNodes(tree->keyLength);
		tree->root[i] = newRoot;
		traversePointer = nodeArr;
		traverseSubtree(&traversePointer,oldRoot,minAccess,maxMemory);
		int currentNodeCount = (int)(traversePointer-nodeArr);
		printf("currentNodeCount %d\n", currentNodeCount);
		finalNodeCount -= currentNodeCount;

		qsort(nodeArr, currentNodeCount, sizeof(TreeNode *), accessMemoryOrderingCmp);
		j=0;
		while(j<currentNodeCount-batchConstant )
		{
			qsort(nodeArr+j, batchConstant, sizeof(TreeNode *), keyOrderingCmp);
			recursiveMoveNode(nodeArr+j, tree, NODEDEPTH,i);
			j+=batchConstant;
		}
		if(j<currentNodeCount)
		{
			for(k=j;k<currentNodeCount;k++){
				moveNode(tree, nodeArr[k],i);
			}
		}
		freeNodes(tree,oldRoot);

	}
	
	free(nodeArr);
}

int * getAccessCounts(Tree *tree)
{
	int i = 0;
	int nodeCount = tree->numNodes;
	TreeNode ** nodeArr = getAllNodes(tree);
	int * accessCounts = malloc(nodeCount*sizeof(int));
	for(i=0;i<nodeCount;i++){
		accessCounts[i] = nodeArr[i]->dataModifiedCount;
	}
	free(nodeArr);
	return accessCounts;
}

void printNodeArray(TreeNode ** nodeArr, int count)
{
	for(int i = 0;i<count;i++){
		printf("Node access: %p, ",nodeArr[i]);
		printf("dataPointer: %p, ", nodeArr[i]->dataPointer);
		printf("with access count %d, ", nodeArr[i]->dataModifiedCount);
		printf("memory weight: %d. ", nodeArr[i]->memoryUsage);
		printf("key[0]: %lu\n ", nodeArr[i]->key[0]);
	}
}