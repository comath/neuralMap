#include "key.h"

#define DATASIZE 32

uint calcKeyLen(uint dataLen)
{
	uint keyLen = (dataLen/DATASIZE);
	if(dataLen % DATASIZE){
		keyLen++;
	}
	return keyLen;
}

char compareKey(uint *x, uint *y, uint keyLen)
{
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

void convertToKey(int * raw, uint *key,uint dataLen)
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

void convertFromKey(uint *key, int * raw, uint dataLen)
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

void addIndexToKey(uint * key, uint index)
{
	int j = index % DATASIZE;
	if(checkIndex(key,index)){
		key[index/DATASIZE] += (1 << (DATASIZE-1-j));
	}
}

uint checkIndex(uint * key, uint index)
{
	return key[index/DATASIZE] & (1 << (DATASIZE-1-(index % DATASIZE)));
}

void clearKey(uint *key, uint keyLen)
{
	uint i = 0;
	for(i=0;i<keyLen;i++){
		key[i] = 0;
	}
}

void printKeyArr(uint *key, uint length){
	uint i = 0;
	printf("[");
	for(i=0;i<length-1;i++){
		printf("%u,",key[i]);
	}
	printf("%u", key[length-1]);
	printf("]\n");
}