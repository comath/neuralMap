#include "paralleltree.h"

void fillTreeNodes(TreeNode *tree, int treeDepth)
{
	int rc = 0;
	tree->data = 0;
	tree->created = 0;
	tree->value = 0;

	rc = pthread_spin_init(&(tree->smallspinlock), 0);
	if (rc != 0) {
        printf("spinlock Initialization failed at %p", (void *) tree);
    }
	rc = pthread_spin_init(&(tree->bigspinlock), 0);
	if (rc != 0) {
        printf("spinlock Initialization failed at %p", (void *) tree);
    }
	rc = pthread_spin_init(&(tree->dataspinlock), 0);
	if (rc != 0) {
        printf("spinlock Initialization failed at %p", (void *) tree);
    }

	if(treeDepth > -1) {
		tree->smallNode = tree - (1 << treeDepth);
		tree->bigNode = tree + (1 << treeDepth);
		fillTreeNodes(tree->smallNode,treeDepth-1);
		fillTreeNodes(tree->bigNode,treeDepth-1);
	} else {
		tree->smallNode = NULL;
		tree->bigNode = NULL;
	}

}

TreeNode * allocateTree(int treeDepth)
{
	const int maxTreeSize = (1 << (treeDepth+1)) - 1;
	struct TreeNode * tree = malloc(maxTreeSize * sizeof(TreeNode));
	
	tree = tree + (1 << treeDepth) - 1;
	fillTreeNodes(tree ,treeDepth-1);
	return tree;
}



void addVector(Tree * tree, int vector){
	TreeNode * node = tree->root;
	int treeDepth = tree->depth;
	while(vector != node->value){
		if (node->value < vector) {
			pthread_spin_lock(&(node->bigspinlock));
			if(node->bigNode == NULL){
				node->bigNode = allocateTree(treeDepth);
				(node->bigNode)->created = 1;
				(node->bigNode)->value = vector;
				tree->numNodes++;
			} else if ((node->bigNode)->created == 0){
				(node->bigNode)->created = 1;
				(node->bigNode)->value = vector;
				tree->numNodes++;
			}
			pthread_spin_unlock(&(node->bigspinlock));
			node = node->bigNode;
		} else if (node->value > vector) {
			pthread_spin_lock(&(node->smallspinlock));
			if(node->smallNode == NULL){
				node->smallNode = allocateTree(treeDepth);
				(node->smallNode)->created = 1;
				(node->smallNode)->value = vector;
				tree->numNodes++;
			} else if ((node->smallNode)->created == 0){
				(node->smallNode)->created = 1;
				(node->smallNode)->value = vector;
				tree->numNodes++;
			}
			pthread_spin_unlock(&(node->smallspinlock));
			node = node->smallNode;
		}
		
	}
	pthread_spin_lock(&(node->dataspinlock));
		node->data++;
	pthread_spin_unlock(&(node->dataspinlock));
}



Tree * createTree(int treeDepth)
{
	Tree * tree = malloc(sizeof(Tree));
	tree->depth = treeDepth;
	tree->root = allocateTree(treeDepth);
	tree->numNodes = 0;
	return tree;	
}

void freeNode(TreeNode *node, int nodeDepth)
{
	int i = 0;
	node = node - (1 << nodeDepth) + 1;

	int n = (1 << (nodeDepth+1)) - 1;

	for(i=0;i<n;i=i+2){
		if(node[i].bigNode){
			freeNode(node[i].bigNode,nodeDepth);
		}
		if(node[i].smallNode){
			freeNode(node[i].smallNode,nodeDepth);
		}
	}
	for(i=0;i<n;i++){
		pthread_spin_destroy(&(node[i].bigspinlock));
		pthread_spin_destroy(&(node[i].smallspinlock));
		pthread_spin_destroy(&(node[i].dataspinlock));
	}

	free(node);
}

void freeTree(Tree *tree)
{
	freeNode(tree->root,tree->depth);
	free(tree);
}

void printNode(TreeNode *tree, int treeDepth)
{
	printf("Tree root at: %p\n", (void *) tree);
	tree = tree - (1 << treeDepth) + 1;
	int i = 0;
	int n = (1 << (treeDepth+1)) - 1;
	for(i=0;i<n;i++){
		printf("------------------------\n");
		printf("Node: %d at %p\n", i, (void *) (tree+i));
		printf("The smallNode is at: %p\n", (void *) tree[i].smallNode);
		printf("The bigNode is at: %p\n", (void *) tree[i].bigNode);
		printf("Creation value: %d \n", tree[i].created);
		printf("The value is: %d \n", tree[i].value);
		printf("The data is: %d \n", tree[i].data);
	}
	printf("------------------------- Subtrees: ----------------\n");
	n = (1 << treeDepth) -1;
	int location =0;
	for(i=0;i<n;i++){
		location = (i << 1);
		location = (i << 1);
		if(tree[location].bigNode){
			printNode(tree[location].bigNode,treeDepth);
		}
		if(tree[location].smallNode){
			printNode(tree[location].smallNode,treeDepth);
		}
	}
}

void printTree(Tree *tree)
{
	printNode(tree->root, tree->depth);
}

struct vectorAddThreadArgs {
	int numVectors;
	int *vectors;
	int tid;
	int numThreads;
	Tree *tree;
};

void * addBatch_thread(void *thread_args)
{
	struct vectorAddThreadArgs *myargs;
	myargs = (struct vectorAddThreadArgs *) thread_args;
	Tree *tree = myargs->tree;
	int tid = myargs->tid;
	int numVectors = myargs->numVectors;
	int *vectors = myargs->vectors;
	int numThreads = myargs->numThreads;

	int i = 0;
	for(i=tid;i<numVectors;i=i+numThreads){	
		addVector(tree,vectors[i]);
	}
	pthread_exit(NULL);
}

void addBatch(Tree * tree, int * vectors, int numVectors)
{
	int maxThreads = sysconf(_SC_NPROCESSORS_ONLN);
	int rc =0;
	int i =0;

	struct vectorAddThreadArgs *thread_args = malloc(maxThreads*sizeof(struct vectorAddThreadArgs));

	pthread_t threads[maxThreads];
	pthread_attr_t attr;
	void *status;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	
	for(i=0;i<maxThreads;i++){
		thread_args[i].tree = tree;
		thread_args[i].numVectors = numVectors;
		thread_args[i].vectors = vectors;
		thread_args[i].numThreads = maxThreads;
		thread_args[i].tid = i;
		rc = pthread_create(&threads[i], NULL, addBatch_thread, (void *)&thread_args[i]);
		if (rc){
			printf("Error, unable to create thread\n");
			exit(-1);
		}
	}

	for( i=0; i < maxThreads; i++ ){
		rc = pthread_join(threads[i], &status);
		if (rc){
			printf("Error, unable to join: %d \n", rc);
			exit(-1);
     	}
	}

	free(thread_args);
}	
