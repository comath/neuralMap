#include "key.h"

#define DATASIZE 32

uint calcKeyLen(uint dataLen)
{
	uint keyLen = (dataLen/DATASIZE);
	if(dataLen % DATASIZE){
		keyLen++;
	}
}

char compareKey(uint *x, uint *y, uint length)
{
	int i = 0;
	for(i = 0; i < length; i++){
		if(x->key[i] > y->key[i]){
			return -1;
		} 
		if (x->key[i] < y->key[i]){
			return 1;
		}
	}
	return 0;
}

uint convertToKey(int * raw, uint *key,uint dataLen)
{
	uint keyLen = calc(dataLen);
	key->key = calloc(keyLen , sizeof(DATATYPE));
	int i = 0,j=0;
	for(i=0;i<dataLen;i++){
		j = i % DATASIZE;
		if(raw[i]){
			key->key[i/DATASIZE] += (1 << (DATASIZE -j -1))	;
		}	
	}
	return keyLen;
}

void convertFromKey(uint *key, int * raw, uint dataLen)
{
	uint keyLen = calc(dataLen);
	int i = 0,j=0;
	for(i=0;i<dataLen;i++){
		j = i % DATASIZE;
		if(checkIndex(key,index)){
			raw[i] = 1;
		} else {
			raw[i] = 0;
		}
	}

}

void addIndexToKey(uint * key, uint index)
{
	if(checkIndex(key,index)){
		key->key[index/DATASIZE] += (1 << (DATASIZE-1-j));
	}
}

char checkIndex(uint * key, uint index)
{
	int j = index % DATASIZE;
	return key->key[index/DATASIZE] & (1 << (DATASIZE-1-j));
}

void clearKey(uint *key, uint dataLen)
{
	uint keyLen = calcKeyLen(dataLen);
	uint i = 0;
	for(i=0;i<keyLen;i++){
		key[i] = 0;
	}
}