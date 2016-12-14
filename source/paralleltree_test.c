#include <stdio.h>
#include <stdlib.h>
#include "paralleltree.h"



struct dataStruct {
	double totalValue;
	int totalCount;
	double errorValue;
	int errorCount;
};

struct dataInput {
	double value;
	char error;
};

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

int *createRandomArray(int numElements, int min, int max)
{
	srand(time(NULL));
	int * randArr = malloc(numElements*sizeof(int));
	int i = 0;
	int range = max - min;
	if(range <= 0){
		range = RAND_MAX;
	}
	for(i=0;i<numElements;i++){
		randArr[i] = (rand() % (max)) + min;
	}
	return randArr;
}

int main(int argc, char* argv[])
{
	const int population = 300000;
	const int max = 10000;
	const int min = -10000;
	const unsigned int treeDepth = 3;
	printf("Creating an pseudo-random array with %d elements, max val: %d, min val: %d\n",population,max,min);
	int * randArr;
	randArr = createRandomArray(population,min,max);
	printf("Success!\n");
	printf("Creating the tree and allocating the first node.\n");
	Tree *tree = createTree(treeDepth);
	printf("Success!\n");
	
	int numToAdd = population;
	printf("Adding %d nodes with addVector\n", numToAdd);
	addBatch(tree, randArr, numToAdd);
	printf("Success!\n");
	
	free(randArr);
	freeTree(tree);

}