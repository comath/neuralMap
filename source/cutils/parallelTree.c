#include "parallelTree.h"

void fillTreeNodes(TreeNode *node, int nodeDepth)
{
	int rc = 0;
	node->dataModifiedCount = 0;
	node->created = 0;
	node->key = 0;
	node->dataPointer = NULL;

	rc = pthread_spin_init(&(node->keyspinlock), 0);
	if (rc != 0) {
        printf("spinlock Initialization failed at %p", (void *) node);
    }
	rc = pthread_spin_init(&(node->smallspinlock), 0);
	if (rc != 0) {
        printf("spinlock Initialization failed at %p", (void *) node);
    }
	rc = pthread_spin_init(&(node->bigspinlock), 0);
	if (rc != 0) {
        printf("spinlock Initialization failed at %p", (void *) node);
    }
	rc = pthread_spin_init(&(node->dataspinlock), 0);
	if (rc != 0) {
        printf("spinlock Initialization failed at %p", (void *) node);
    }

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

TreeNode * allocateNodes(int treeDepth)
{
	const int maxTreeSize = (1 << (treeDepth+1)) - 1;
	struct TreeNode * tree = malloc(maxTreeSize * sizeof(TreeNode));
	
	tree = tree + (1 << treeDepth) - 1;
	fillTreeNodes(tree ,treeDepth-1);
	return tree;
}

Tree * createTree(int treeDepth, uint keyLength, void * (*dataCreator)(void * input),
				void (*dataModifier)(void * input, void * data),void (*dataDestroy)(void * data))
{
	Tree * tree = malloc(sizeof(Tree));
	tree->depth = treeDepth;
	tree->root = allocateNodes(treeDepth);
	tree->numNodes = 0;
	tree->keyLength = keyLength;
	tree->dataCreator = dataCreator;
	tree->dataModifier = dataModifier;
	tree->dataDestroy = dataDestroy;
	return tree;	
}


void * addData(Tree *tree, uint * key, void * datum){
	TreeNode * node = tree->root;
	int treeDepth = tree->depth;
	
	pthread_spin_lock(&(node->keyspinlock));
	if (node->created == 0){
		node->created = 1;
		node->key = key;
		node->dataPointer = tree->dataCreator(datum);
		tree->numNodes++;
	}
	int keyCompare = compareKey(key,node->key,tree->keyLength);
	pthread_spin_unlock(&(node->keyspinlock));
	while(keyCompare){
		if (keyCompare == 1) {
			pthread_spin_lock(&(node->bigspinlock));
			if(node->bigNode == NULL){
				node->bigNode = allocateNodes(treeDepth);
				(node->bigNode)->created = 1;
				(node->bigNode)->key = key;
				(node->bigNode)->dataPointer = tree->dataCreator(datum);
				tree->numNodes++;
			} 
			if ((node->bigNode)->created == 0){
				(node->bigNode)->created = 1;
				(node->bigNode)->key = key;
				(node->bigNode)->dataPointer = tree->dataCreator(datum);
				tree->numNodes++;
			}
			pthread_spin_unlock(&(node->bigspinlock));
			node = node->bigNode;
		} else if (keyCompare == -1) {
			pthread_spin_lock(&(node->smallspinlock));
			if(node->smallNode == NULL){
				node->smallNode = allocateNodes(treeDepth);
				(node->smallNode)->created = 1;
				(node->smallNode)->key = key;
				(node->smallNode)->dataPointer = tree->dataCreator(datum);
				tree->numNodes++;
			} 
			if ((node->smallNode)->created == 0){
				(node->smallNode)->created = 1;
				(node->smallNode)->key = key;
				(node->smallNode)->dataPointer = tree->dataCreator(datum);
				tree->numNodes++;
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
			node->dataModifiedCount++;
		}		
	pthread_spin_unlock(&(node->dataspinlock));
	return (void *) node->dataPointer;
}

void * getData(Tree *tree, uint *key)
{
	TreeNode * node = tree->root;
	if (node->created == 0){
		return NULL;
	}
	char comparison;
	while((comparison = compareKey(key,node->key,tree->keyLength))){

		if (comparison == 1) {
			pthread_spin_lock(&(node->bigspinlock));
			if(node->bigNode == NULL || node->bigNode->created == 0){
				return NULL;
			}
			pthread_spin_unlock(&(node->bigspinlock));
			node = node->bigNode;
		} else if (comparison == -1) {
			pthread_spin_lock(&(node->smallspinlock));
			if(node->smallNode == NULL || node->smallNode->created == 0){
				return NULL;
			}
			pthread_spin_unlock(&(node->smallspinlock));
			node = node->smallNode;
		}
		
	}
	return (void *) node->dataPointer;
}


void freeNode(Tree *tree, TreeNode *node, int nodeDepth)
{
	int i = 0;
	node = node - (1 << nodeDepth) + 1;

	int n = (1 << (nodeDepth+1)) - 1;

	for(i=0;i<n;i=i+2){
		if(node[i].bigNode){
			freeNode(tree,node[i].bigNode,nodeDepth);
		}
		if(node[i].smallNode){
			freeNode(tree,node[i].smallNode,nodeDepth);
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
		freeNode(tree,tree->root,tree->depth);
		free(tree);
	}
}

