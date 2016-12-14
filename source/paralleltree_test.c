#include <stdio.h>
#include <stdlib.h>
#include "paralleltree.h"

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