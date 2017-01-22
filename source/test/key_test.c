#include <stdio.h>
#include <stdlib.h>
#include "../cutils/parallelTree.h"

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


void printCharArr(char * arr, int numElements){
	int i = 0;
	printf("[");
	for(i=0;i<numElements-1;i++){
		printf("%u,",arr[i]);
	}
	printf("%u", arr[numElements-1]);
	printf("]\n");
}

void testKeyLen()
{
	uint arrLen = 600;
	uint keyLen = calcKeyLen(arrLen);
	uint keyLen2 = (arrLen/32);
	if(arrLen % 32){
		keyLen2++;
	}
	if(keyLen != keyLen2){
		printf("Faliure\n");
	}
}

void testCompareKey()
{
	
	uint arrLength = 30;
	int *arr = createRandomBinaryArray(arrLength);
	int *arr2 = createRandomBinaryArray(arrLength);
	uint keyLen = calcKeyLen(arrLength);

	uint *testKey = malloc(keyLen*sizeof(uint));
	uint *testKey2 = malloc(keyLen*sizeof(uint));
	convertToKey(arr,testKey,arrLength);
	convertToKey(arr2,testKey2,arrLength);
	if(compareKey( testKey, testKey2,keyLen) != compareArr(arr,arr2,arrLength)){
		printf("Faliure\n");
	}
	free(arr);
	free(arr2);
	free(testKey);
	free(testKey2);
}

void testRebuildKey()
{
	
	int arrLength = 30;
	int *arr = createRandomBinaryArray(arrLength);
	int *RecreateArr = malloc(arrLength*sizeof(int));
	
	uint keyLen = calcKeyLen(arrLength);
	uint *testKey = malloc(keyLen*sizeof(uint));

	convertToKey(arr,testKey,arrLength);
	convertFromKey(testKey,RecreateArr,arrLength);
	if(compareArr(arr,RecreateArr,arrLength) != 0){
		printf("Faliure, Recreated Arr:\n");
		printIntArr(RecreateArr,arrLength);
		printf("Original one:\n");
		printIntArr(arr,arrLength);
	}


	free(arr);
	free(RecreateArr);
	free(testKey);
}

int main(int argc, char* argv[])
{
	srand(time(NULL));
	int i = 0;
	printf("If no faliures are printed then we are fine.\n");
	printf("test keyLen\n");
	void testKeyLen();
	printf("testCompareKey:\n");
	testCompareKey();
	for(i=0;i<100;i++){	testCompareKey(); }
	printf("testRebuildKey:\n");
	for(i=0;i<100;i++){	testRebuildKey(); } 
	return 0;
}