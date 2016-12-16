#include <stdio.h>
#include <stdlib.h>
#include "paralleltree.h"

#define DATASIZE 32
#define DATATYPE unsigned int

void convertToKey(int * raw, Key * key,int length)
{
	
	key->length = (length/DATASIZE);
	if(length % DATASIZE){
		key->length++;
	}
	key->key = calloc(key->length, sizeof(DATATYPE));
	int i =0,j=0;
	for(i=length-1;i>-1;i--){
		if(raw[i]){
			key->key[i/DATASIZE] += (1 << (i % DATASIZE));
		}
	}
}

void convertFromKey(Key * key, int * output, int length)
{
	int i = 0;
	for(i=length-1;i>-1;i--){
		if(key->key[i/DATASIZE] & (1 << (i % DATASIZE))){
			output[i] = 1;
		} else {
			output[i] = 0;
		}
	}
}

int * createRandomBinaryArray(int numElements)
{
	srand(time(NULL));
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
		printf("%d,",arr[i]);
	}
	printf("%d", arr[numElements-1]);
	printf("]\n");
}

void printCharArr(unsigned char * arr, int numElements){
	int i = 0;
	printf("[");
	for(i=0;i<numElements-1;i++){
		printf("%u,",arr[i]);
	}
	printf("%u", arr[numElements-1]);
	printf("]\n");
}

int main(int argc, char* argv[])
{
	int arrLength = 30;
	int *arr = createRandomBinaryArray(arrLength);
	printIntArr(arr,arrLength);

	Key *testKey = malloc(sizeof(Key));
	convertToKey(arr,testKey,arrLength);
	printCharArr(testKey->key, testKey-> length);
	convertFromKey(testKey,arr,arrLength);
	printIntArr(arr,arrLength);

	return 0;
}