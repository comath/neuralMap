#ifndef _key_h
#define _key_h
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <error.h>
#include <unistd.h>

#include <stdint.h>
/*
We store the sets bit packed into unsigned 32 bit integers. This makes all future operations faster and simpler to write out
*/
#define kint uint32_t  //Definition of kint, makes it easier to resize later if needed

// Functions to handle key interactions
int compareKey(kint *x, kint *y, uint keyLength); // Lexographically compares two keys. See keyTest for the required behavior.

uint calcKeyLen(uint dataLen); // Returns the length of key needed to store a 
void addIndexToKey(kint * key, uint index); // Changes the bit at index i from 0 to 1. Does nothing if bit is already 1
void removeIndexFromKey(kint * key, uint index); // Changes the bit at index i from 1 to 0. Does nothing if bit is already 0
uint checkIndex(kint * key, uint index); // Returns a positive integer if the bit at the provided index is set
void clearKey(kint *key, uint keyLength);  // Sets a key to 0
int checkEmptyKey(kint *key, uint keyLength); // Returns 1 if the key is empty
void copyKey(kint *key1, kint * key2, uint keyLen); // Copies from Key1 into key2, though I don't call this anywhere I think.
int offByOne(kint *x, kint *y, uint dataLen); // Returns 1 if x and y differ by a single bit
unsigned int numberOfOneBits(kint *x, int keyLength); // Returns the order of a set

void printKeyArr(kint *key, uint length); // Prints the raw key. For debugging purposes.
void printKey(kint *key, uint dataLen); // Prints the set associated to the key. If dataLen is unknown at time of call, just call with 32*keyLen

// Coverts the keys between the bitpacking and the various formats.
void convertFromIntToKey(int * raw, kint *key,uint dataLen);
void batchConvertFromIntToKey(int *raw, kint *key,uint dataLen, uint numData);
void convertFromFloatToKey(float * raw, kint *key,uint dataLen);
void convertFromKeyToFloat(kint *key, float * output, uint dataLen);
void convertFromKeyToInt(kint *key, int * output, uint dataLen);
void batchConvertFromKeyToInt(kint *key, int * output, uint dataLen,uint numData);
void convertFromKeyToChar(kint *key, char * output, uint dataLen);
void batchConvertFromKeyToChar(kint *key, char * output, uint dataLen,uint numData);


void chromaticKey(kint *key, float *rgb, uint dataLen); // For some python scripts. Returns a RGB pixel associated to the signature
void batchChromaticKey(kint *key, float *rgb, uint dataLen, uint numData);

/* 
Given a selectionVector and Bias returns if this region is selected for or against. 
Returns 1 if selected for, -1 selected against, 0 for neither (if by some random chance the gradient is completely 0 here).
*/
int evalSig(kint *regSig, float *selectionVec, float selectionBias, uint dataLen); 







#endif