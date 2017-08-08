#include "key.h"
#include <stdint.h>
#include <string.h>

#define DATASIZE 32

int checkOffByNArray(kint * keyArray, kint* testKey, uint numKeys, uint keyLen, uint n)
{
	if(n == 1){
		for(uint i = 0;i<numKeys;i++){
			if(offByOne(keyArray + i*keyLen, testKey, keyLen)){
				return i;
			}
		}
		return -1;
	} else {
		uint diffSize = 0;
		for(uint i = 0;i<numKeys;i++){
			diffSize = numberOfDiff(keyArray + i*keyLen, testKey, keyLen);
			if(diffSize <= n){
				return i;
			}
		}
		return -1;
	}
}

struct offByNThreadArgs {
	uint tid;
	uint numThreads;

	kint * keyArray;
	kint* testKeys;
	uint numKeys;
	uint keyLen;
	uint n;
	uint numTestKeys;
	int * results;
};

void * offByN_thread(void *thread_args)
{
	struct offByNThreadArgs *myargs;
	myargs = (struct offByNThreadArgs *) thread_args;

	uint tid = myargs->tid;	
	uint numThreads = myargs->numThreads;

	kint * keyArray = myargs->keyArray;
	kint* testKeys = myargs->testKeys;
	uint numKeys = myargs->numKeys;
	uint keyLen = myargs->keyLen;
	uint n = myargs->n;
	uint numTestKeys = myargs->numTestKeys;
	int * results = myargs->results;

	uint i = 0;
	for(i=tid;i<numTestKeys;i=i+numThreads){
		results[i] = checkOffByNArray(keyArray,testKeys + i*keyLen,numKeys,keyLen,n);
		//printf("Thread %d at mutex with %u nodes \n",tid,tc->bases->numNodes);	
	}
	pthread_exit(NULL);
}

void batchCheckOffByN(kint * keyArray, kint* testKeys, uint numKeys, uint keyLen, uint n, uint numTestKeys, int * results, int numProc)
{
	int maxThreads = numProc;
	int rc =0;
	int i =0;
	struct offByNThreadArgs *thread_args = malloc(maxThreads*sizeof(struct offByNThreadArgs));
	
	pthread_t threads[maxThreads];
	pthread_attr_t attr;
	void *status;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	for(i=0;i<maxThreads;i++){
		thread_args[i].keyArray = keyArray;
		thread_args[i].testKeys = testKeys;
		thread_args[i].numKeys = numKeys;
		thread_args[i].keyLen = keyLen;
		thread_args[i].n = n;
		thread_args[i].numTestKeys = numTestKeys;
		thread_args[i].results = results;
		thread_args[i].numThreads = maxThreads;
		thread_args[i].tid = i;
		rc = pthread_create(&threads[i], NULL, offByN_thread, (void *)&thread_args[i]);
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

int getMinGraphDistance(kint * keyArray, kint* testKey, uint numKeys, uint keyLen, uint n)
{	
	uint diffSize = 0;
	uint currentMinDiffSize = keyLen * DATASIZE;
	for(uint i = 0;i<numKeys;i++){
		diffSize = numberOfDiff(keyArray + i*keyLen, testKey, keyLen);
		if(diffSize <= currentMinDiffSize){
			currentMinDiffSize = diffSize;
		}
	}
	return currentMinDiffSize;
}

struct getMinGraphDistThreadArgs {
	uint tid;
	uint numThreads;

	kint * keyArray;
	kint* testKeys;
	uint numKeys;
	uint keyLen;
	uint n;
	uint numTestKeys;
	int * results;
};

void * getMinGraphDistanceThread_thread(void *thread_args)
{
	struct offByNThreadArgs *myargs;
	myargs = (struct offByNThreadArgs *) thread_args;

	uint tid = myargs->tid;	
	uint numThreads = myargs->numThreads;

	kint * keyArray = myargs->keyArray;
	kint* testKeys = myargs->testKeys;
	uint numKeys = myargs->numKeys;
	uint keyLen = myargs->keyLen;
	uint n = myargs->n;
	uint numTestKeys = myargs->numTestKeys;
	int * results = myargs->results;

	uint i = 0;
	for(i=tid;i<numTestKeys;i=i+numThreads){
		results[i] = getMinGraphDistance(keyArray,testKeys + i*keyLen,numKeys,keyLen,n);
		//printf("Thread %d at mutex with %u nodes \n",tid,tc->bases->numNodes);	
	}
	pthread_exit(NULL);
}

void batchGetMinGraphDistance(kint * keyArray, kint* testKeys, uint numKeys, uint keyLen, uint n, uint numTestKeys, int * results, int numProc)
{
	int maxThreads = numProc;
	int rc =0;
	int i =0;
	struct getMinGraphDistThreadArgs *thread_args = malloc(maxThreads*sizeof(struct getMinGraphDistThreadArgs));
	
	pthread_t threads[maxThreads];
	pthread_attr_t attr;
	void *status;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	for(i=0;i<maxThreads;i++){
		thread_args[i].keyArray = keyArray;
		thread_args[i].testKeys = testKeys;
		thread_args[i].numKeys = numKeys;
		thread_args[i].keyLen = keyLen;
		thread_args[i].n = n;
		thread_args[i].numTestKeys = numTestKeys;
		thread_args[i].results = results;
		thread_args[i].numThreads = maxThreads;
		thread_args[i].tid = i;
		rc = pthread_create(&threads[i], NULL, getMinGraphDistanceThread_thread, (void *)&thread_args[i]);
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

void getGraphDist(kint * keyArray, kint* testKey, uint numKeys, uint keyLen, int * results)
{
	for(uint i = 0;i<numKeys;i++){
		results[i] = numberOfDiff(keyArray + i*keyLen, testKey, keyLen);
	}
}


uint calcKeyLen(uint dataLen)
{
	uint keyLen = (dataLen/DATASIZE);
	if(dataLen % DATASIZE){
		keyLen++;
	}
	return keyLen;
}

uint checkIndex(kint *key, uint i)
{
	if(key[i/DATASIZE] & (1 << (DATASIZE-1-(i % DATASIZE)))){
		return 1;
	} else {
		return 0;
	}
}

void convertFromKeyToInt(kint *key, int * raw, uint dataLen)
{
	uint i = 0;
	for(i=0;i<dataLen;i++){
		if(checkIndex(key,i)){
			raw[i] = 1;
		} else {
			raw[i] = 0;
		}
	}
}

void convertFromKeyToChar(kint *key, char * raw, uint dataLen)
{
	uint i = 0;
	for(i=0;i<dataLen;i++){
		if(checkIndex(key,i)){
			raw[i] = 1;
		} else {
			raw[i] = 0;
		}
	}
}

void convertFromIntToKey(int * raw, kint *key,uint dataLen)
{

	uint keyLen = calcKeyLen(dataLen);
	clearKey(key, keyLen);
	uint i = 0,j=0;
	for(i=0;i<dataLen;i++){
		j = i % DATASIZE;
		if(raw[i]){
			key[i/DATASIZE] += (1 << (DATASIZE - j -1))	;
		}	
	}
}

int compareKey(kint *x, kint *y, uint keyLen)
{
	//return memcmp (x,y, keyLen*sizeof(kint));

	uint i = 0;
	
	for(i = 0; i < keyLen; i++){
		if(x[i] > y[i]){
			return -1;
		} 
		if (x[i] < y[i]){
			return 1;
		}
	}
	return 0;

}

int isPowerOfTwo (kint x)
{
  return ((x != 0) && !(x & (x - 1)));
}

int offByOne(kint *x, kint *y, uint keyLen)
{
	kint cmp = 0;
	uint i = 0;
	uint j = 0;
	// Find the first non-zero difference
	while(i<keyLen && cmp == 0){
		cmp = x[i] - y[i];
		i++; 
	}
	// check that it's the only non-zero difference 
	for(j = i; j<keyLen;j++){
		if(x[j] - y[j]){
			return 0;
		}
	}
	// cmp should contain the only non-zero entry of the difference
	// Checking that it's a power of two
	return isPowerOfTwo(cmp);
}

unsigned int numberOfOneBits(kint *x, uint keyLen)
{
	unsigned int numberOfOneBits = 0;
	kint xi;
	uint i = 0;
	for(i = 0; i<keyLen;i++){
		xi = x[i]; // Copy x[i]
		while(xi){
			if ((xi & 1) == 1) 
			  numberOfOneBits++;
			xi >>= 1;          
		}
	}

	return numberOfOneBits; 
}

unsigned int numberOfDiff(kint *x, kint *y, uint keyLen)
{
	unsigned int numberOfOneBits = 0;
	kint zj;
	for(uint j = 0;j<keyLen;j++){
		zj = x[j] ^ y[j];
		while(zj){
			if ((zj & 1) == 1) 
			  numberOfOneBits++;
			zj >>= 1;          
		}
	}
	return numberOfOneBits;
}


int evalSig(kint *key, float *selectionVec, float selectionBias, uint dataLen)
{
	uint i = 0;
	float result = selectionBias;
	for(i=0;i<dataLen;i++){
		if(checkIndex(key,i)){
			result += selectionVec[i];
		} 
	}
	if(result > 0){
		return 1;
	} else if(result < 0){
		return -1;
	} else {
		return 0;
	}
}

int checkEmptyKey(kint *key,uint keyLen)
{
	uint i = 0;
	for(i = 0; i < keyLen; i++){
		if(checkIndex(key,i) == 0){
			return 1;
		}
	}
	return 0;
}


void batchConvertFromIntToKey(int * raw, kint *key,uint dataLen, uint numData){
	uint i =0;
	uint keyLen = calcKeyLen(dataLen);
	for(i=0;i<numData;i++){
		convertFromIntToKey(raw + i*dataLen, key +i*keyLen,dataLen);
	}
}



void addIndexToKey(kint *key, uint index)
{
	int j = index % DATASIZE;
	if(checkIndex(key,index) == 0){
		key[index/DATASIZE] += (1 << (DATASIZE-1-j));
	}
}
void removeIndexFromKey(kint *key, uint index)
{
	int j = index % DATASIZE;
	if(checkIndex(key,index)){
		key[index/DATASIZE] -= (1 << (DATASIZE-1-j));
	}
}



void clearKey(kint *key, uint keyLen)
{
	uint i = 0;
	for(i=0;i<keyLen;i++){
		key[i] = 0;
	}
}

void printKeyArr(kint *key, uint length){
	uint i = 0;
	printf("[");
	for(i=0;i<length-1;i++){
		printf("%u,",key[i]);
	}
	printf("%u", key[length-1]);
	printf("]\n");
}

void printIntArr(int *arr, uint length){
	uint i = 0;
	printf("[");
	for(i=0;i<length-1;i++){
		printf("%d,",arr[i]);
	}
	printf("%d", arr[length-1]);
	printf("]\n");
}

void printKey(kint *key, uint dataLen){
	uint i=0;
	printf("[");
	for(i=0;i<dataLen;i++){
		if(checkIndex(key,i)){
			printf("%u,",i);
		}
	}
	printf("]\n");
}



void copyKey(kint *key1, kint *key2, uint keyLen)
{
	memcpy(key2, key1, keyLen*sizeof(kint));
}

void batchConvertFromKeyToInt(kint *key, int * raw, uint dataLen,uint numData){
	uint i =0;
	uint keyLen = calcKeyLen(dataLen);
	printf("Converting a key. Data length: %u, numData: %u\n",dataLen,numData );
	for(i=0;i<numData;i++){
		convertFromKeyToInt(key + i*keyLen, raw +i*dataLen,dataLen);
	}
}

void batchConvertFromKeyToChar(kint *key, char * raw, uint dataLen,uint numData)
{
	uint i =0;
	uint keyLen = calcKeyLen(dataLen);
	printf("Converting a key. Data length: %u, numData: %u\n",dataLen,numData );
	for(i=0;i<numData;i++){
		convertFromKeyToChar(key + i*keyLen, raw +i*dataLen,dataLen);
	}
}

void chromaticKey(kint* key, float *rgb, uint dataLen)
{
	rgb[0] = 0;
	rgb[1] = 0;
	rgb[2] = 0;
	uint i = 0;
	for(i =0;i<dataLen;i++){
		if(checkIndex(key,i)){
			if(i % 3 == 0){ rgb[0]+= 1.0f/ (1 << (int)(i/3+1)); }
			if(i % 3 == 1){ rgb[1]+= 1.0f/ (1 << (int)(i/3+1)); }
			if(i % 3 == 2){ rgb[2]+= 1.0f/ (1 << (int)(i/3+1)); }
		}		
	}
}

void batchChromaticKey(kint* key, float *rgb, uint dataLen, uint numData){
	uint i =0;
	uint keyLen = calcKeyLen(dataLen);
	for(i=0;i<numData;i++){
		chromaticKey(key + i*keyLen, rgb +i*3,dataLen);
	}
}

void convertFromFloatToKey(float * raw, kint *key,uint dataLen)
{
	uint keyLen = calcKeyLen(dataLen);
	clearKey(key, keyLen);
	uint i = 0,j=0;
	for(i=0;i<dataLen;i++){
		j = i % DATASIZE;
		if(raw[i]>0){
			key[i/DATASIZE] += (1 << (DATASIZE -j -1))	;
		}	
	}
}

void convertFromKeyToFloat(kint *key, float * raw, uint dataLen)
{
	uint i = 0;
	for(i=0;i<dataLen;i++){
		if(checkIndex(key,i)){
			raw[i] = 1.0;
		} else {
			raw[i] = 0.0;
		}
	}
}