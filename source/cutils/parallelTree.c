#include "parallelTree.h"



SubTree * allocateNodes(uint keyLength)
{
	
	SubTree * tree = malloc(sizeof(SubTree));
	kint * keys = malloc(SUBTREESIZE * keyLength * sizeof(kint));
	int rc = 0;
	rc = pthread_mutex_init(&(tree->traverseMutexLock), 0);
	if (rc != 0) {
        printf("spinlock Initialization failed at %p", (void *) tree);
    }
	for (int i = 0; i < SUBTREESIZE; i++)
	{
		tree->nodes[i].accessCount = 0;
		tree->nodes[i].createdKL = 0;
		tree->nodes[i].key = keys + i*keyLength;
		tree->nodes[i].dataPointer = NULL;

		rc = pthread_mutex_init(&(tree->nodes[i].datamutex), 0);
		if (rc != 0) {
	        printf("mutex Initialization failed at %p", (void *) tree + i);
	    }
	}
	memset(tree->nextSubTrees,0, (SUBTREESIZE+1)*sizeof(SubTree *));
	return tree;
}

void freeNodes(Tree *tree, SubTree *st)
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

Tree * createTree(uint keyLength, uint numTrees, int maxDatumMemory, long int maxTreeMemory, void * (*dataCreator)(void * input),
				void (*dataModifier)(void * input, void * data),void (*dataDestroy)(void * data))
{
	Tree * tree = malloc(sizeof(Tree));
	tree->numTrees = numTrees;
	tree->root = malloc(numTrees*sizeof(SubTree *));
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


void * addData(Tree *tree, kint * key, int treeIndex, void * datum, int memoryUsage){
	SubTree * st =	 tree->root[treeIndex];
	int keyLen = tree->keyLength;
	int i = SUBCENTER;
	int d = NODEDEPTH;
	
	pthread_mutex_lock(&(st->traverseMutexLock));

	if (st->nodes[i].createdKL == 0){
		st->nodes[i].createdKL = keyLen;
		copyKey(key, st->nodes[i].key, keyLen);
		st->nodes[i].dataPointer = tree->dataCreator(datum);
		st->nodes[i].memoryUsage = memoryUsage;
		tree->numNodes++;
	}

	int keyCompare;
	while((keyCompare = compareKey(key,st->nodes[i].key,keyLen))){
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
				copyKey(key, (st->nodes[i]).key, keyLen);
				(st->nodes[i]).dataPointer = tree->dataCreator(datum);
				(st->nodes[i]).memoryUsage = memoryUsage;
				pthread_spin_lock(&(tree->nodeCountSpinLock));
					tree->numNodes++;
					tree->currentMemoryUseage += memoryUsage;
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
				copyKey(key, (st->nodes[i]).key, keyLen);
				(st->nodes[i]).dataPointer = tree->dataCreator(datum);
				(st->nodes[i]).memoryUsage = memoryUsage;
				pthread_spin_lock(&(tree->nodeCountSpinLock));
					tree->numNodes++;
					tree->currentMemoryUseage += memoryUsage;
				pthread_spin_unlock(&(tree->nodeCountSpinLock));
			} 
		}
	}
	pthread_mutex_unlock(&(st->traverseMutexLock));
	pthread_mutex_lock(&(st->nodes[i].datamutex));
		if(tree->dataModifier){
			tree->dataModifier(datum,st->nodes[i].dataPointer);
		}
		st->nodes[i].accessCount++;
	pthread_mutex_unlock(&(st->nodes[i].datamutex));
	return (void *) st->nodes[i].dataPointer;
}


void moveNode(Tree *tree, TreeNode * oldNode, int treeIndex){
	SubTree * st = tree->root[treeIndex];
	int keyLen = tree->keyLength;
	int i = SUBCENTER;
	int d = NODEDEPTH;
	if (st->nodes[i].createdKL == 0){
		st->nodes[i].createdKL = keyLen;
		copyKey(oldNode->key, st->nodes[i].key, keyLen);
		st->nodes[i].dataPointer = oldNode->dataPointer;
		st->nodes[i].memoryUsage = oldNode->memoryUsage;
		st->nodes[i].accessCount = oldNode->accessCount;
		tree->numNodes++;
	}
	int keyCompare;
	while((keyCompare = compareKey(oldNode->key,st->nodes[i].key,keyLen))){
		if(d == 0){
			if(keyCompare == 1){
				i = i + 1;
			} else {
				i = i;
			}
			if(st->nextSubTrees[i] == NULL){
				st->nextSubTrees[i] = allocateNodes(keyLen);
				st = st->nextSubTrees[i];

				i = SUBCENTER;
				d = NODEDEPTH;

				(st->nodes[i]).createdKL = keyLen;
				copyKey(oldNode->key, (st->nodes[i]).key, keyLen);
				(st->nodes[i]).dataPointer = oldNode->dataPointer;
				(st->nodes[i]).memoryUsage = oldNode->memoryUsage;
				(st->nodes[i]).accessCount = oldNode->accessCount;
				tree->numNodes++;
				tree->currentMemoryUseage += oldNode->memoryUsage;
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
				(st->nodes[i]).createdKL = keyLen;
				copyKey(oldNode->key, (st->nodes[i]).key, keyLen);
				(st->nodes[i]).dataPointer = oldNode->dataPointer;
				(st->nodes[i]).memoryUsage = oldNode->memoryUsage;
				(st->nodes[i]).accessCount = oldNode->accessCount;
				tree->numNodes++;
				tree->currentMemoryUseage += oldNode->memoryUsage;
			}
		}
	}
	oldNode->dataPointer = NULL;
	
}

void traverseSubtree(TreeNode *(*(*traversePointer)), SubTree *st, int minAccess, int maxMemory)
{
	int i = 0;
	//printf("-----traverseSubtree-----\n");
	for(i=0;i<SUBTREESIZE;i++){
		
		if(i%2==0 && st->nextSubTrees[i]){
			//printf("---callingsmall--\n");
			traverseSubtree(traversePointer, st->nextSubTrees[i],minAccess,maxMemory);
		}
		if(st->nodes[i].dataPointer && st->nodes[i].createdKL && 
			(minAccess<0 || st->nodes[i].accessCount >= minAccess)){
			/*
			printf("Node access: %p, ",st->nodes+i);
			printf("dataPointer: %p, ", st->nodes[i].dataPointer);
			printf("with access count %d, ", st->nodes[i].accessCount);
			printf("memory weight: %d. ", st->nodes[i].memoryUsage);
			printf("key[0]: %lu\n ", st->nodes[i].key[0]);
			*/
			if(minAccess<0 || st->nodes[i].accessCount*st->nodes[i].accessCount > minAccess || maxMemory > st->nodes[i].memoryUsage){
				*(*traversePointer) = st->nodes + i;
				(*traversePointer)++;
			} 
		}
		if(i%2==0 && st->nextSubTrees[i+1]){
			//printf("---callingBig--\n");
			traverseSubtree(traversePointer, st->nextSubTrees[i+1],minAccess,maxMemory);
		}
	}
	//printf("---//traverseSubtree//--\n");
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
		//printf("===============================On Rank %d=====================================\n", i);
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
	int comp = ((myB)->accessCount - (myA)->accessCount);
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
		recursiveMoveNode(nodeArr + k+1, tree, recursionDepth-1, treeIndex);
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
	int minAccess = nodeArr[lastNodeIndex-1]->accessCount;
	int maxMemory = nodeArr[lastNodeIndex-1]->memoryUsage;
	
	printf("-----------Obtained the minAccess, maxMemory: %d,%d-----------\n",minAccess, maxMemory);

	int i=0,j=0,k=0;
	int batchConstant = (1 << (NODEDEPTH+1))-1;

	for(i = 0;i<tree->numTrees;i++){
		SubTree * oldRoot = tree->root[i];
		SubTree * newRoot = allocateNodes(tree->keyLength);
		tree->root[i] = newRoot;
		traversePointer = nodeArr;
		traverseSubtree(&traversePointer,oldRoot,minAccess,maxMemory);
		int currentNodeCount = (int)(traversePointer-nodeArr);
		//printf("currentNodeCount %d\n", currentNodeCount);
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
		accessCounts[i] = nodeArr[i]->accessCount;
	}
	free(nodeArr);
	return accessCounts;
}

void printNodeArray(TreeNode ** nodeArr, int count)
{
	for(int i = 0;i<count;i++){
		printf("Node access: %p, ",nodeArr[i]);
		printf("dataPointer: %p, ", nodeArr[i]->dataPointer);
		printf("with access count %d, ", nodeArr[i]->accessCount);
		printf("memory weight: %ld. ", nodeArr[i]->memoryUsage);
		printf("key[0]: %lu\n ", nodeArr[i]->key[0]);
	}
}