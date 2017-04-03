#include <stdio.h>
#include <stdlib.h>
#include "../cutils/parallelTree.h"



typedef struct dataStruct {
	double totalValue;
	int totalCount;
	double errorValue;
	int errorCount;
} dataStruct;

typedef struct dataInput {
	int value;
	char error;
	kint key[1];
} dataInput;

void dataModifier(void * input, void * data)
{
	struct dataInput *myInput;
	myInput = (struct dataInput *) input;
	struct dataStruct *myData;
	myData = (struct dataStruct *) data;

	

	if(myData->totalCount){
		myData->totalValue = ((myData->totalCount)*myData->totalValue + myInput->value) / (myData->totalCount + 1);
		myData->totalCount++;
	} else {
		myData->totalValue = myInput->value;
		myData->totalCount = 1;
	}

	if(myInput->error){
		if(myData->errorCount) {
			myData->errorValue = ((myData->errorCount)*myData->errorValue + myInput->value) / (myData->errorCount + 1);
			myData->errorCount++;
		} else {
			myData->totalValue = myInput->value;
			myData->totalCount = 1;
		}
	}
}

// Data handling function pointers
void * dataCreator(void * input)
{
	struct dataStruct *myData = malloc(sizeof(struct dataStruct));
	myData->totalValue = 0;
	myData->totalCount = 0;
	myData->errorValue = 0;
	myData->errorCount = 0;
	return (void *) myData;
}


void dataDestroy(void * data)
{
	struct dataStruct *myData;
	myData = (struct dataStruct *) data;
	free(myData);
}

dataInput  *createRandomArray(int numElements,  int max)
{
	srand(time(NULL));
	dataInput * randArr = malloc(numElements*sizeof(dataInput));
	int i = 0;

	for(i=0;i<numElements;i++){
		randArr[i].value = (rand() % max);
		randArr[i].error = (rand() % 2);
		randArr[i].key[0] = randArr[i].value % 50;
	}
	return randArr;
}

struct dataAddThreadArgs {
	int numKeys;
	dataInput * data;
	int tid;
	int numThreads;
	Tree *tree;
};

void * addBatch_thread(void *thread_args)
{
	struct dataAddThreadArgs *myargs;
	myargs = (struct dataAddThreadArgs *) thread_args;
	Tree *tree = myargs->tree;
	int tid = myargs->tid;
	int numkeys = myargs->numKeys;
	dataInput * data = myargs-> data;
	int numThreads = myargs->numThreads;

	int i = 0;
	for(i=tid;i<numkeys;i=i+numThreads){
		addData(tree,data[i].key,0, data + i);
	}
	pthread_exit(NULL);
}

void addBatch(Tree * tree, struct dataInput *data, int numKeys)
{
	int maxThreads = sysconf(_SC_NPROCESSORS_ONLN);
	int rc =0;
	int i =0;

	//Add one data to the first node so that we can avoid the race condition.
	

	struct dataAddThreadArgs *thread_args = malloc(maxThreads*sizeof(struct dataAddThreadArgs));

	pthread_t threads[maxThreads];
	pthread_attr_t attr;
	void *status;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	
	for(i=0;i<maxThreads;i++){
		thread_args[i].tree = tree;
		thread_args[i].numKeys = numKeys ;
		thread_args[i].data = data ;
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

typedef struct location {
	kint *key;
	double totalValue;
	int totalCount;
	double errorValue;
	int errorCount;
} location;

void traverseLocationSubtreeTEST(Tree * myTree, location * locArr, TreeNode *node)
{
	#ifdef DEBUG
		printf("Working on node %p node\n",node);
	#endif
	int i = 0;
	int nodeDepth = 2;
	node = node - (1 << nodeDepth) + 1;
	struct dataStruct *myData = NULL;
	int n = (1 << (nodeDepth+1)) - 1;



	for(i=0;i<n;i++){
		#ifdef DEBUG
			printf("smallNode:%p dataPointer: %p bigNode: %p\n",node[i].smallNode,node[i].dataPointer,node[i].bigNode);
		#endif
		if(i%2==0 && node[i].smallNode){
			traverseLocationSubtreeTEST(myTree, locArr, node[i].smallNode);
		}		
		if(node[i].dataPointer){
			printf("Node access: %d\n", node[i].dataModifiedCount);
			locArr[i].key = node[i].key;
			myData = (struct dataStruct *) node[i].dataPointer;
			locArr[i].totalValue = myData->totalValue;
			locArr[i].totalCount = myData->totalCount;
		}
		if(i%2==0 && node[i].bigNode){
			traverseLocationSubtreeTEST(myTree, locArr, node[i].bigNode);
		}
	}	
}

location * getLocationArray(Tree * myTree)
{	
	uint numLoc = myTree->numNodes;
	location * locArr = malloc(numLoc * sizeof(location));
	#ifdef DEBUG
		printf("Root node is %p, numNodes is %u\n",myTree->root, numLoc);
	#endif
	traverseLocationSubtreeTEST(myTree, locArr, myTree->root[0]);
	return locArr;
}





#undef DEBUG

int main(int argc, char* argv[])
{
	const int population = 3000000;
	const int max = 5000;
	printf("Creating an pseudo-random array with %d elements, max val: %d\n",population,max);
	dataInput * randArr;
	randArr = createRandomArray(population,max);
	printf("Success!\n");
	uint keyLength = calcKeyLen(1);
	printf("Creating the tree and allocating the first node.\n");
	Tree *tree = createTree(keyLength, 1, dataCreator,dataModifier,dataDestroy);
	printf("Success!\n");
	
	printf("Adding %d nodes with addVector\n", population);
	addBatch(tree, randArr, population);
	printf("Success!\n");

	location *locArr = getLocationArray(tree);
	printf("Location 2 is %d\n", locArr[2].totalValue);
	
	printf("We have %d nodes. Removing half.\n", tree->numNodes);
	balanceAndTrimTree(tree, tree->numNodes/2);

	addBatch(tree, randArr, population);
	printf("We have %d nodes. Removing half.\n", tree->numNodes);
	balanceAndTrimTree(tree, tree->numNodes/2);

	addBatch(tree, randArr, population);
	printf("We have %d nodes. Removing half.\n", tree->numNodes);
	balanceAndTrimTree(tree, tree->numNodes/2);

	addBatch(tree, randArr, population);
	printf("We have %d nodes. Removing half.\n", tree->numNodes);
	balanceAndTrimTree(tree, tree->numNodes/2);

	free(locArr);
	free(randArr);
	freeTree(tree);
	return 0;
}