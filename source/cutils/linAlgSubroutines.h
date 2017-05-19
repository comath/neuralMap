#ifndef _linAlgSubroutines_h
#define _linAlgSubroutines_h
#include "structDefs.h"
ipCacheData * solve(ipMemory *mb);

void printFloatArr(float * arr, uint length);
void printMatrix(float * arr, uint inDim, uint outDim);
#endif