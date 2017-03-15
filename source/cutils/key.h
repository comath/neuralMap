#ifndef _key_h
#define _key_h
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <error.h>
#include <unistd.h>



char compareKey(uint *x, uint *y, uint keyLength);
char emptyKey(uint key,uint keyLength);

void convertToKey(int * raw, uint *key,uint dataLen);
void convertFloatToKey(float * raw, uint *key,uint dataLen);

void convertFromKey(uint *key, int * output, uint dataLen);
void convertFromKeyToFloat(uint *key, float * output, uint dataLen);

void copyKey(uint *key1, uint * key2, uint keyLen);

void chromaticKey(uint* key, float *rgb, uint dataLen);

void batchConvertToKey(int * raw, uint *key,uint dataLen, uint numData);
void batchConvertFromKey(uint *key, int * output, uint dataLen,uint numData);
void batchChromaticKey(uint* key, float *rgb, uint dataLen, uint numData);

uint calcKeyLen(uint dataLen);

void addIndexToKey(uint * key, uint index);
void removeIndexFromKey(uint * key, uint index);
uint checkIndex(uint * key, uint index);
void clearKey(uint *key, uint keyLength);

void printKeyArr(uint *key, uint length);
void printKey(uint* key, uint dataLen);




#endif