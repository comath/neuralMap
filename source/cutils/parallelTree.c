#include "parallelTree.h"
#include <stdint.h>

#define NODEDEPTH 6 

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
		rc = pthread_spin_init(&(tree[i].dataspinlock), 0);
		if (rc != 0) {
	        printf("spinlock Initialization failed at %p", (void *) tree + i);
	    }
	}
	tree = tree + (1 << NODEDEPTH) - 1;
	fillTreeNodesPointers(tree,NODEDEPTH-1);
	return tree;
}

Tree * createTree(uint keyLength, uint numTrees, void * (*dataCreator)(void * input),
				void (*dataModifier)(void * input, void * data),void (*dataDestroy)(void * data))
{
	Tree * tree = malloc(sizeof(Tree));
	tree->numTrees = numTrees;
	tree->root = malloc(numTrees*sizeof(TreeNode *));
	for(uint i=0;i<numTrees;i++){
		tree->root[i] = allocateNodes(keyLength);
	}
	tree->numNodes = 0;
	tree->keyLength = keyLength;
	tree->dataCreator = dataCreator;
	tree->dataModifier = dataModifier;
	tree->dataDestroy = dataDestroy;
	return tree;	
}


void * addData(Tree *tree, kint * key, int treeIndex, void * datum){
	TreeNode * node = tree->root[treeIndex];
	int keyLen = tree->keyLength;
	pthread_spin_lock(&(node->keyspinlock));
	if (node->createdKL == 0){
		node->createdKL = 1;
		copyKey(key, node->key, keyLen);
		node->dataPointer = tree->dataCreator(datum);
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
				tree->numNodes++;
			} 
			if ((node->bigNode)->createdKL == 0){
				(node->bigNode)->createdKL = 1;
				copyKey(key, (node->bigNode)->key, keyLen);
				(node->bigNode)->dataPointer = tree->dataCreator(datum);
				tree->numNodes++;
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
				tree->numNodes++;
			} 
			if ((node->smallNode)->createdKL == 0){
				(node->smallNode)->createdKL = 1;
				copyKey(key, (node->smallNode)->key, keyLen);
				(node->smallNode)->dataPointer = tree->dataCreator(datum);
				tree->numNodes++;
			}
			pthread_spin_unlock(&(node->smallspinlock));
			node = node->smallNode;
		}
		pthread_spin_lock(&(node->keyspinlock));
		keyCompare = compareKey(key,node->key,keyLen);
		pthread_spin_unlock(&(node->keyspinlock));
	}

	pthread_spin_lock(&(node->dataspinlock));
		if(tree->dataModifier){
			tree->dataModifier(datum,node->dataPointer);
		}
		node->dataModifiedCount++;
	pthread_spin_unlock(&(node->dataspinlock));
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
		pthread_spin_destroy(&(node[i].dataspinlock));
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
		free(tree);
	}
}

void moveNode(Tree *tree, TreeNode * oldNode, int treeIndex){
	TreeNode * node = tree->root[treeIndex];
	
	if (node->createdKL == 0){
		node->createdKL = tree->keyLength;
		copyKey(oldNode->key, node->key, tree->keyLength);
		node->dataPointer = oldNode->dataPointer;
		tree->numNodes++;
	}
	int keyCompare = compareKey(oldNode->key,node->key,tree->keyLength);

	while(keyCompare){
		if (keyCompare == 1) {
			if(node->bigNode == NULL){
				node->bigNode = allocateNodes(tree->keyLength);
				(node->bigNode)->createdKL = tree->keyLength;
				copyKey(oldNode->key, (node->bigNode)->key, tree->keyLength);
				(node->bigNode)->dataPointer = oldNode->dataPointer;
				tree->numNodes++;
			} 
			if ((node->bigNode)->createdKL == 0){
				(node->bigNode)->createdKL = tree->keyLength;
				copyKey(oldNode->key, (node->bigNode)->key, tree->keyLength);
				(node->bigNode)->dataPointer = oldNode->dataPointer;
				tree->numNodes++;
			}
			node = node->bigNode;
		} else if (keyCompare == -1) {
			if(node->smallNode == NULL){
				node->smallNode = allocateNodes(tree->keyLength);
				(node->smallNode)->createdKL = tree->keyLength;
				copyKey(oldNode->key, (node->smallNode)->key, tree->keyLength);
				(node->smallNode)->dataPointer = oldNode->dataPointer;
				tree->numNodes++;
			} 
			if ((node->smallNode)->createdKL == 0){
				(node->smallNode)->createdKL = tree->keyLength;
				copyKey(oldNode->key, (node->smallNode)->key, tree->keyLength);
				(node->smallNode)->dataPointer = oldNode->dataPointer;
				tree->numNodes++;
			}
			node = node->smallNode;
		}
		keyCompare = compareKey(oldNode->key,node->key,tree->keyLength);
	}
	node->dataModifiedCount = oldNode->dataModifiedCount;
	node->dataPointer = NULL;
}

void traverseSubtree(TreeNode *(*(*traversePointer)), TreeNode *node, int minAccess)
{
	int i = 0;
	node = node - (1 << NODEDEPTH) + 1;
	int n = (1 << (NODEDEPTH+1)) - 1;
	printf("-----traverseSubtree-----\n");
	for(i=0;i<n;i++){
		
		if(i%2==0 && node[i].smallNode){
			printf("---callingsmall--\n");
			traverseSubtree(traversePointer, node[i].smallNode,minAccess);
		}
		if(node[i].dataPointer && node[i].createdKL && (minAccess<0 || node[i].dataModifiedCount > minAccess)){
			printf("Node access: %p, with access count %d. Pointer: %p\n",node+i, (node+i)->dataModifiedCount, *traversePointer);
			*(*traversePointer) = node+i;
			(*traversePointer) ++;
		}
		if(i%2==0 && node[i].bigNode){
			printf("---callingBig--\n");
			traverseSubtree(traversePointer, node[i].bigNode,minAccess);
		}
	}
	printf("---//traverseSubtree//--\n");
}


TreeNode ** getAllNodes(Tree * tree)
{
	TreeNode *(*nodePointerArr) = malloc(tree->numNodes*sizeof(TreeNode *));
	printf("%p %lu %d %lu\n", nodePointerArr,tree->numNodes*sizeof(TreeNode *),tree->numNodes, sizeof(TreeNode *));
	TreeNode *(*traversePointer) = nodePointerArr;
	for(int i = 0;i<tree->numTrees;i++){
		traverseSubtree(&traversePointer,tree->root[i],-1);
	}
	if(traversePointer - nodePointerArr != tree->numNodes){
		printf("Node Count Off! %ld, should be %d\n", traversePointer - nodePointerArr,tree->numNodes);
		exit(-1);
	}
	return nodePointerArr;
}

int accessOrderingCmp (const void * a, const void * b)
{
	struct TreeNode *myA = *(TreeNode * const *)a;
	struct TreeNode *myB = *(TreeNode * const *)b;
	return ( (myB)->dataModifiedCount - (myA)->dataModifiedCount );
}

int keyOrderingCmp(const void * a, const void * b)
{
	struct TreeNode *myA = *(TreeNode * const *)a;
	struct TreeNode *myB = *(TreeNode * const *)b;
	return (int)(compareKey(myA->key,myB->key,myB->createdKL));
}

void recursiveAddNode(TreeNode ** nodeArr, Tree *tree, int recursionDepth, int treeIndex)
{
	int k = (1 << (recursionDepth))-1;
	moveNode(tree, nodeArr[k],treeIndex);
	if(recursionDepth > 0){
		recursiveAddNode(nodeArr, tree, recursionDepth-1, treeIndex);
		recursiveAddNode(nodeArr + k, tree, recursionDepth-1, treeIndex);
	}
}

void balanceAndTrimTree(Tree *tree, int desiredNodeCount)
{
	printf("Balancing tree and trimming to %d\n", desiredNodeCount);

	int finalNodeCount = 0;
	int oldNodeCount = tree->numNodes;
	
	if(desiredNodeCount < oldNodeCount){
		finalNodeCount = desiredNodeCount;
	} else {
		finalNodeCount = oldNodeCount;
	}
	
	TreeNode ** nodeArr = getAllNodes(tree);
	TreeNode ** traversePointer;
	tree->numNodes = 0;
	qsort(nodeArr, oldNodeCount, sizeof(TreeNode *), accessOrderingCmp);
	int minAccess = nodeArr[finalNodeCount-1]->dataModifiedCount;
	printf("-----------Obtained the minAccess: %d-----------\n",minAccess);

	int i=0;
	unsigned int j=0, k=0;
	int batchConstant = (1 << (NODEDEPTH+1))-1;

	for(i = 0;i<tree->numTrees;i++){
		TreeNode * oldRoot = tree->root[i];
		TreeNode * newRoot = allocateNodes(tree->keyLength);
		tree->root[i] = newRoot;
		traversePointer = nodeArr;
		traverseSubtree(&traversePointer,oldRoot,minAccess);
		long unsigned int currentNodeCount = traversePointer-nodeArr;
		printf("currentNodeCount %lu\n", currentNodeCount);
		finalNodeCount -= currentNodeCount;

		qsort(nodeArr, currentNodeCount, sizeof(TreeNode *), accessOrderingCmp);
		j=0;
		while(j<currentNodeCount )
		{
			qsort(nodeArr+j, batchConstant, sizeof(TreeNode *), keyOrderingCmp);
			recursiveAddNode(nodeArr, tree, NODEDEPTH,i);
			j+=batchConstant;
		}
		j-=batchConstant;
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