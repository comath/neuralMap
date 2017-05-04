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
		randArr[i].key[0] = randArr[i].value % 5000;
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
		addData(tree,data[i].key,0, data + i, data[i].value % 10);
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





int main(int argc, char* argv[])
{
	printf("Size of kint: %lu\n", sizeof(kint));
	const int population = 3000000;
	const int max = 50000;
	printf("Creating an pseudo-random array with %d elements, max val: %d\n",population,max);
	dataInput * randArr;
	randArr = createRandomArray(population,max);
	printf("Success!\n");
	uint keyLength = calcKeyLen(1);
	printf("Creating the tree and allocating the first node. keyLength: %u\n", keyLength);
	Tree *tree = createTree(keyLength, 1, 10,250,dataCreator,dataModifier,dataDestroy);
	printf("Success!\n");
	
	printf("Adding %d nodes with addVector\n", population);
	addBatch(tree, randArr, population);
	printf("Success!\n");

	
	
	printf("We are using %d memory. Removing half.\n", tree->currentMemoryUseage);
	balanceAndTrimTree(tree, tree->currentMemoryUseage/2);
	printf("We are using %d memory, after trim.\n", tree->currentMemoryUseage);

	printf("Adding %d nodes with addVector\n", population);
	addBatch(tree, randArr, population);
	printf("Success!\n");

	
	
	printf("We are using %d memory. Removing half.\n", tree->currentMemoryUseage);
	balanceAndTrimTree(tree, tree->currentMemoryUseage/2);
	printf("We are using %d memory, after trim.\n", tree->currentMemoryUseage);

	printf("Adding %d nodes with addVector\n", population);
	addBatch(tree, randArr, population);
	printf("Success!\n");

	
	
	printf("We are using %d memory. Removing half.\n", tree->currentMemoryUseage);
	balanceAndTrimTree(tree, tree->currentMemoryUseage/2);
	printf("We are using %d memory, after trim.\n", tree->currentMemoryUseage);

	printf("Adding %d nodes with addVector\n", population);
	addBatch(tree, randArr, population);
	printf("Success!\n");

	
	
	printf("We are using %d memory. Removing half.\n", tree->currentMemoryUseage);
	balanceAndTrimTree(tree, tree->currentMemoryUseage/2);
	printf("We are using %d memory, after trim.\n", tree->currentMemoryUseage);
	
	

	free(randArr);
	freeTree(tree);
	return 0;
}