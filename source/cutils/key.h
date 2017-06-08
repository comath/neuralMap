#ifndef _key_h
#define _key_h
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <error.h>
#include <unistd.h>

#include <stdint.h>

#define kint uint32_t

int compareKey(kint *x, kint *y, uint keyLength);
int checkEmptyKey(kint *key, uint keyLength);

int offByOne(kint *x, kint *y, uint dataLen);
unsigned int numberOfOneBits(kint *x, int keyLength);
int evalSig(kint *regSig, float *selectionVec, float selectionBias, uint dataLen);


void convertToKey(int * raw, kint *key,uint dataLen);
void convertFloatToKey(float * raw, kint *key,uint dataLen);

void convertFromKey(kint *key, int * output, uint dataLen);
void convertFromKeyChar(kint *key, char * output, uint dataLen);
void convertFromKeyToFloat(kint *key, float * output, uint dataLen);

void copyKey(kint *key1, kint * key2, uint keyLen);

void chromaticKey(kint *key, float *rgb, uint dataLen);

void batchConvertToKey(int *raw, kint *key,uint dataLen, uint numData);
void batchConvertFromKey(kint *key, int * output, uint dataLen,uint numData);
void batchConvertFromKeyChar(kint *key, char * output, uint dataLen,uint numData);
void batchChromaticKey(kint *key, float *rgb, uint dataLen, uint numData);

uint calcKeyLen(uint dataLen);

void addIndexToKey(kint * key, uint index);
void removeIndexFromKey(kint * key, uint index);
uint checkIndex(kint * key, uint index);
void clearKey(kint *key, uint keyLength);

void printKeyArr(kint *key, uint length);
void printKey(kint *key, uint dataLen);




#endif