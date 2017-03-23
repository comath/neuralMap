#include "parallelTree.h"
#include <stdint.h>

void fillTreeNodes(TreeNode *node, int nodeDepth)
{
	if(nodeDepth > -1) {
		node->smallNode = node - (1 << nodeDepth);
		node->bigNode = node + (1 << nodeDepth);
		fillTreeNodes(node->smallNode,nodeDepth-1);
		fillTreeNodes(node->bigNode,nodeDepth-1);
	} else {
		node->smallNode = NULL;
		node->bigNode = NULL;
	}
}

TreeNode * allocateNodes(int treeDepth, uint keyLength)
{
	const int maxTreeSize = (1 << (treeDepth+1)) - 1;
	struct TreeNode * tree = malloc(maxTreeSize * sizeof(TreeNode));
	kint * keys = malloc(maxTreeSize * keyLength * sizeof(kint));
	int rc = 0;
	for (int i = 0; i < maxTreeSize; i++)
	{
		tree[i].dataModifiedCount = 0;
		tree[i].createdKL = keyLength;
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
	tree = tree + (1 << treeDepth) - 1;
	fillTreeNodes(tree,treeDepth-1);
	return tree;
}

Tree * createTree(int treeDepth, uint keyLength, void * (*dataCreator)(void * input),
				void (*dataModifier)(void * input, void * data),void (*dataDestroy)(void * data))
{
	int rc = 0;
	Tree * tree = malloc(sizeof(Tree));
	tree->depth = treeDepth;
	tree->root = allocateNodes(treeDepth,keyLength);
	tree->numNodes = 0;
	tree->keyLength = keyLength;
	tree->dataCreator = dataCreator;
	tree->dataModifier = dataModifier;
	tree->dataDestroy = dataDestroy;
	rc = pthread_spin_init(&(tree->nodecountspinlock), 0);
	if (rc != 0) {
        printf("nodecountspinlock Initialization failed at");
    }
	return tree;	
}

void moveNode(Tree *tree, TreeNode * oldNode){
	TreeNode * node = tree->root;
	int treeDepth = tree->depth;
	
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
				node->bigNode = allocateNodes(treeDepth,tree->keyLength);
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
				node->smallNode = allocateNodes(treeDepth,tree->keyLength);
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

void * addData(Tree *tree, kint * key, void * datum){
	TreeNode * node = tree->root;
	int treeDepth = tree->depth;

	pthread_spin_lock(&(node->keyspinlock));
	if (node->createdKL == 0){
		node->createdKL = tree->keyLength;
		copyKey(key, node->key, tree->keyLength);
		node->dataPointer = tree->dataCreator(datum);
		tree->numNodes++;
	}
	int keyCompare = compareKey(key,node->key,tree->keyLength);
	pthread_spin_unlock(&(node->keyspinlock));
	while(keyCompare){
		if (keyCompare == 1) {
			pthread_spin_lock(&(node->bigspinlock));
				if(node->bigNode == NULL){
					node->bigNode = allocateNodes(treeDepth,tree->keyLength);
					(node->bigNode)->createdKL = tree->keyLength;
					copyKey(key, (node->bigNode)->key, tree->keyLength);
					(node->bigNode)->dataPointer = tree->dataCreator(datum);
					pthread_spin_lock(&(tree->nodecountspinlock));
						tree->numNodes++;
					pthread_spin_unlock(&(tree->nodecountspinlock));
				} 
				if ((node->bigNode)->createdKL == 0){
					(node->bigNode)->createdKL = tree->keyLength;
					copyKey(key, (node->bigNode)->key, tree->keyLength);
					(node->bigNode)->dataPointer = tree->dataCreator(datum);
					pthread_spin_lock(&(tree->nodecountspinlock));
						tree->numNodes++;
					pthread_spin_unlock(&(tree->nodecountspinlock));
				}
			pthread_spin_unlock(&(node->bigspinlock));
			node = node->bigNode;
		} else if (keyCompare == -1) {
			pthread_spin_lock(&(node->smallspinlock));
				if(node->smallNode == NULL){
					node->smallNode = allocateNodes(treeDepth,tree->keyLength);
					(node->smallNode)->createdKL = tree->keyLength;
					copyKey(key, (node->smallNode)->key, tree->keyLength);
					(node->smallNode)->dataPointer = tree->dataCreator(datum);
					pthread_spin_lock(&(tree->nodecountspinlock));
						tree->numNodes++;
					pthread_spin_unlock(&(tree->nodecountspinlock));
				} 
				if ((node->smallNode)->createdKL == 0){
					(node->smallNode)->createdKL = tree->keyLength;
					copyKey(key, (node->smallNode)->key, tree->keyLength);
					(node->smallNode)->dataPointer = tree->dataCreator(datum);
					pthread_spin_lock(&(tree->nodecountspinlock));
						tree->numNodes++;
					pthread_spin_unlock(&(tree->nodecountspinlock));
				}
			pthread_spin_unlock(&(node->smallspinlock));
			node = node->smallNode;
		}
		pthread_spin_lock(&(node->keyspinlock));
			keyCompare = compareKey(key,node->key,tree->keyLength);
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

void * getData(Tree *tree, kint *key)
{
	TreeNode * node = tree->root;
	if (node->createdKL == 0){
		return NULL;
	}
	char comparison;
	while((comparison = compareKey(key,node->key,tree->keyLength))){

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
	node = node - (1 << tree->depth) + 1;
	// Free the keyspace.
	free(node->key);
	int n = (1 << (tree->depth+1)) - 1;

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
		freeNodes(tree,tree->root);
		free(tree);
	}
}

void traverseSubtree(TreeNode ** nodeArr, TreeNode *node, Tree *tree)
{
	#ifdef DEBUG
		printf("Working on node %p node\n",node);
	#endif
	int i = 0;
	int nodeDepth = tree->depth;
	node = node - (1 << nodeDepth) + 1;
	int n = (1 << (nodeDepth+1)) - 1;



	for(i=0;i<n;i++){
		
		if(i%2==0 && node[i].smallNode){
			traverseSubtree(nodeArr, node[i].smallNode,tree);
		}		
		if(node[i].dataPointer && node[i].createdKL){
			*nodeArr = node+i;
			nodeArr++;
		}
		if(i%2==0 && node[i].bigNode){
			traverseSubtree(nodeArr, node[i].bigNode,tree);
		}
	}	
}

TreeNode ** getNodeArray(Tree *tree)
{
	TreeNode ** nodePointerArr = malloc(tree->numNodes * sizeof(TreeNode*));
	#ifdef DEBUG
		printf("Root node is %p, numNodes is %u\n",tree->root, tree->numNodes);
	#endif
	traverseSubtree(nodePointerArr, tree->root, tree);
	return nodePointerArr;
}

int accessOrderingCmp (const void * a, const void * b)
{
	struct TreeNode *myA;
	myA = (struct TreeNode *) a;
	struct TreeNode *myB;
	myB = (struct TreeNode *) b;
	return ( myB->dataModifiedCount - myA->dataModifiedCount );
}

int keyOrderingCmp(const void * a, const void * b)
{
	struct TreeNode *myA;
	myA = (struct TreeNode *) a;
	struct TreeNode *myB;
	myB = (struct TreeNode *) b;
	return (int)(compareKey(myA->key,myB->key,myB->createdKL));
}

void recursiveAddNode(TreeNode ** nodeArr, Tree *tree, int recursionDepth)
{
	int k = (1 << (recursionDepth))-1;
	moveNode(tree, nodeArr[k]);
	if(recursionDepth > 0){
		recursiveAddNode(nodeArr, tree, recursionDepth-1);
		recursiveAddNode(nodeArr + k, tree, recursionDepth-1);
	}
}

void balanceAndTrimTree(Tree *tree, int desiredNodeCount)
{
	TreeNode ** nodeArr = getNodeArray(tree);
	int oldNodeCount = tree->numNodes;
	TreeNode * oldRoot = tree->root;
	TreeNode * newRoot = allocateNodes(tree->depth, tree->keyLength);
	tree->numNodes = 0;
	tree->root = newRoot;
	qsort(nodeArr, oldNodeCount, sizeof(TreeNode *), accessOrderingCmp);
	int i = 0;
	int j = 0;
	int recursionDepth = 5;
	int batchConstant = (1 << (recursionDepth+1))-1;
	int finalNodeCount = 0;
	if(desiredNodeCount < oldNodeCount){
		finalNodeCount = desiredNodeCount;
	} else {
		finalNodeCount = oldNodeCount;
	}
	while(i<finalNodeCount )
	{
		qsort(nodeArr+i, batchConstant, sizeof(TreeNode *), keyOrderingCmp);
		recursiveAddNode(nodeArr, tree, recursionDepth);
		i+=batchConstant;
	}
	i-=batchConstant;
	if(i<finalNodeCount)
	{
		for(j=i;j<finalNodeCount;j++){
			moveNode(tree, nodeArr[j]);
		}
	}

	freeNodes(tree,oldRoot);
	free(nodeArr);
}

int * getAccessCounts(Tree *tree)
{
	int i = 0;
	int nodeCount = tree->numNodes;
	TreeNode ** nodeArr = getNodeArray(tree);
	int * accessCounts = malloc(nodeCount*sizeof(int));
	for(i=0;i<nodeCount;i++){
		accessCounts[i] = nodeArr[i]->dataModifiedCount;
	}
	free(nodeArr);
	return accessCounts;
}