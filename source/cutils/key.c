#include "key.h"
#include <stdint.h>

#define DATASIZE 32


uint calcKeyLen(uint dataLen)
{
	uint keyLen = (dataLen/DATASIZE);
	if(dataLen % DATASIZE){
		keyLen++;
	}
	return keyLen;
}

char compareKey(kint *x, kint *y, uint keyLen)
{
	//return memcmp ( x,y, keyLen*sizeof(kint) );
	
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

char checkEmptyKey(kint *key,uint keyLength)
{
	uint i = 0;
	for(i = 0; i < keyLength; i++){
		if(checkIndex(key,i) == 0){
			return 1;
		}
	}
	return 0;
}

void convertToKey(int * raw, kint *key,uint dataLen)
{

	uint keyLen = calcKeyLen(dataLen);
	clearKey(key, keyLen);
	uint i = 0,j=0;
	for(i=0;i<dataLen;i++){
		j = i % DATASIZE;
		if(raw[i]){
			key[i/DATASIZE] += (1 << (DATASIZE -j -1))	;
		}	
	}
}



void batchConvertToKey(int * raw, kint *key,uint dataLen, uint numData){
	uint i =0;
	uint keyLen = calcKeyLen(dataLen);
	for(i=0;i<numData;i++){
		convertToKey(raw + i*dataLen, key +i*keyLen,dataLen);
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

uint checkIndex(kint *key, uint index)
{
	if(key[index/DATASIZE] & (1 << (DATASIZE-1-((kint)index % DATASIZE)))){
		return 1;
	} else {
		return 0;
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
		printf("%lu,",key[i]);
	}
	printf("%lu", key[length-1]);
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

void convertFromKey(kint *key, int * raw, uint dataLen)
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

void copyKey(kint *key1, kint *key2, uint keyLen)
{
	uint i = 0;
	for(i=0;i<keyLen;i++){
		key2[i] = key1[i];
	}
}

void batchConvertFromKey(kint *key, int * raw, uint dataLen,uint numData){
	uint i =0;
	uint keyLen = calcKeyLen(dataLen);
	printf("Converting a key. Data length: %u, numData: %u\n",dataLen,numData );
	for(i=0;i<numData;i++){
		convertFromKey(key + i*keyLen, raw +i*dataLen,dataLen);
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

void convertFloatToKey(float * raw, kint *key,uint dataLen)
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