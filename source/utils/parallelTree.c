#include "paralleltree.h"

void fillTreeNodes(TreeNode *node, int nodeDepth)
{
	int rc = 0;
	node->dataModifiedCount = 0;
	node->created = 0;
	node->value = 0;
	node->dataPointer = NULL;

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

Tree * createTree(int treeDepth, void * (*dataCreator)(void * input),
				void (*dataModifier)(void * input, void * data),void (*dataDestroy)(void * data))
{
	Tree * tree = malloc(sizeof(Tree));
	tree->depth = treeDepth;
	tree->root = allocateNodes(treeDepth);
	tree->numNodes = 0;
	tree->dataCreator = dataCreator;
	tree->dataModifier = dataModifier;
	tree->dataDestroy = dataDestroy;
	return tree;	
}


char compareKey(Key *x, Key *y)
{
	int i = 0;
	for(i = 0; i < x->length; i++){
		if(x->key[i] > y->key[i]){
			return -1;
		} 
		if (x->key[i] < y->key[i]){
			return 1;
		}
	}
	return 0;
}

void convertToKey(int * raw, Key * key,int length)
{
	
	key->length = (length/DATASIZE);
	if(length % DATASIZE){
		key->length++;
	}
	key->key = calloc(key->length , sizeof(DATATYPE));
	int i = 0,j=0;
	for(i=0;i<length;i++){
		j = i % DATASIZE;
		if(raw[i]){
			key->key[i/DATASIZE] += (1 << (DATASIZE -j -1))	;
		}
		
	}
}

void convertFromKey(Key * key, int * output, int length)
{
	int i = 0,j=0;
	for(i=0;i<length;i++){
		j = i % DATASIZE;
		if(key->key[i/DATASIZE] & (1 << (DATASIZE-1  -j))){
			output[i] = 1;
		} else {
			output[i] = 0;
		}
	}

}

void addData(Tree *tree, int key, void * datum){
	TreeNode * node = tree->root;
	int treeDepth = tree->depth;
	if (node->created == 0){
		node->created = 1;
		node->value = key;
		node->dataPointer = tree->dataCreator(datum);
		tree->numNodes++;
	}
	while(key != node->value){
		if (node->value < key) {
			pthread_spin_lock(&(node->bigspinlock));
			if(node->bigNode == NULL){
				node->bigNode = allocateNodes(treeDepth);
				(node->bigNode)->created = 1;
				(node->bigNode)->value = key;
				(node->bigNode)->dataPointer = tree->dataCreator(datum);
				tree->numNodes++;
			} 
			if ((node->bigNode)->created == 0){
				(node->bigNode)->created = 1;
				(node->bigNode)->value = key;
				(node->bigNode)->dataPointer = tree->dataCreator(datum);
				tree->numNodes++;
			}
			pthread_spin_unlock(&(node->bigspinlock));
			node = node->bigNode;
		} else if (node->value > key) {
			pthread_spin_lock(&(node->smallspinlock));
			if(node->smallNode == NULL){
				node->smallNode = allocateNodes(treeDepth);
				(node->smallNode)->created = 1;
				(node->smallNode)->value = key;
				(node->smallNode)->dataPointer = tree->dataCreator(datum);
				tree->numNodes++;
			} 
			if ((node->smallNode)->created == 0){
				(node->smallNode)->created = 1;
				(node->smallNode)->value = key;
				(node->smallNode)->dataPointer = tree->dataCreator(datum);
				tree->numNodes++;
			}
			pthread_spin_unlock(&(node->smallspinlock));
			node = node->smallNode;
		}
		
	}

	pthread_spin_lock(&(node->dataspinlock));
		tree->dataModifier(datum,node->dataPointer);
		node->dataModifiedCount++;
	pthread_spin_unlock(&(node->dataspinlock));
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
	freeNode(tree,tree->root,tree->depth);
	free(tree);
}

