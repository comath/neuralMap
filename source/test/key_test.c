#include <stdio.h>
#include <stdlib.h>
#include "../utils/paralleltree.h"

char compareArr(int * x, int * y, int length)
{
	int i = 0;
	for(i = 0; i < length; i++){
		if(x[i]<y[i]){
			return 1;
		} 
		if (x[i]>y[i]){
			return -1;
		}
	}
	return 0;
}

int * createRandomBinaryArray(int numElements)
{
	
	int * randArr = malloc(numElements*sizeof(int));
	int i = 0;

	for(i=0;i<numElements;i++){
		randArr[i] = (rand() % 2);
	}
	return randArr;
}

void printIntArr(int * arr, int numElements){
	int i = 0;
	printf("[");
	for(i=0;i<numElements-1;i++){
		printf("%u,",arr[i]);
	}
	printf("%d", arr[numElements-1]);
	printf("]\n");
}



void printKeyArr(Key *x){
	int i = 0;
	printf("[");
	for(i=0;i<x->length;i++){
		printf("%u,",x->key[i]);
	}
	printf("%u", x->key[x->length]);
	printf("]\n");
}

void printCharArr(char * arr, int numElements){
	int i = 0;
	printf("[");
	for(i=0;i<numElements-1;i++){
		printf("%u,",arr[i]);
	}
	printf("%u", arr[numElements-1]);
	printf("]\n");
}

void testCompareKey()
{
	
	int arrLength = 600;
	int *arr = createRandomBinaryArray(arrLength);
	int *arr2 = createRandomBinaryArray(arrLength);

	Key *testKey = malloc(sizeof(Key));
	Key *testKey2 = malloc(sizeof(Key));
	convertToKey(arr,testKey,arrLength);
	convertToKey(arr2,testKey2,arrLength);
	if(compareKey( testKey, testKey2) != compareArr(arr,arr2,arrLength)){
		printf("Faliure\n");
	}
	free(arr);
	free(arr2);
	free(testKey->key);
	free(testKey2->key);
	free(testKey);
	free(testKey2);
}

void testRebuildKey()
{
	
	int arrLength = 600;
	int *arr = createRandomBinaryArray(arrLength);
	int *RecreateArr = malloc(arrLength*sizeof(int));

	Key *testKey = malloc(sizeof(Key));
	convertToKey(arr,testKey,arrLength);
	convertFromKey(testKey,RecreateArr,arrLength);
	if(compareArr(arr,RecreateArr,arrLength) != 0){
		printf("Faliure\n");
	}


	free(arr);
	free(RecreateArr);
	free(testKey->key);
	free(testKey);
}

int main(int argc, char* argv[])
{
	srand(time(NULL));
	int i = 0;
	printf("If no faliures are printed then we are fine.\n");
	printf("testCompareKey:\n");
	for(i=0;i<100;i++){
		testCompareKey();
	}
	printf("testRebuildKey:\n");
	for(i=0;i<100;i++){
		testRebuildKey();
	} 
	return 0;
}