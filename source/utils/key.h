#ifndef _key_h
#define _key_h
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <error.h>
#include <unistd.h>

char compareKey(uint *x, uint *y, uint length);
uint convertToKey(int * raw, uint *key,uint dataLen);
void convertFromKey(uint *key, int * output, uint dataLen);
uint calcKeyLen(uint dataLen);
void addIndexToKey(uint * key, uint index);
char checkIndex(uint * key, uint index);
void clearKey(uint *key, uint dataLen);


#endif