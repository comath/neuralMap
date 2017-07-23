import cython
import numpy as np
cimport numpy as np
import multiprocessing
from libc.stdlib cimport malloc, free
from libc cimport stdint

ctypedef stdint.uint32_t kint

#to print to stderr to warn the user of usage errors

import sys


cdef extern from "../cutils/key.h":
	cdef int checkOffByNArray(kint * keyArray, kint* testKey, unsigned int numKeys, unsigned int keyLength, unsigned int n)
	cdef void batchCheckOffByN(kint * keyArray, kint* testKeys, unsigned int numKeys, unsigned int keyLength, unsigned int n, int numTestKeys, int * results, int numProc)

	cdef int getMinGraphDistance(kint * keyArray, kint* testKey, unsigned int numKeys, unsigned int keyLength, unsigned int n)
	cdef void batchGetMinGraphDistance(kint * keyArray, kint* testKeys, unsigned int numKeys, unsigned int keyLength, unsigned int n, int numTestKeys, int * results, int numProc)

	cdef void getGraphDist(kint * keyArray, kint* testKey, unsigned int numKeys, unsigned int keyLength, unsigned int n, int * results);

	cdef char compareKey(kint *x, kint *y, unsigned int keyLength)
	
	cdef void convertFromIntToKey(int * raw, kint *key,unsigned int dataLen)
	cdef void convertFromKeyToInt(kint *key, int * output, unsigned int dataLen)
	cdef void chromaticKey(kint *key, float *rgb, unsigned int dataLen)

	cdef void batchConvertFromIntToKey(int * raw, kint *key,unsigned int dataLen, unsigned int numData)
	cdef void batchConvertFromKeyToInt(kint *key, int * output, unsigned int dataLen,unsigned int numData)
	cdef void batchConvertFromKeyToChar(kint *key, char * output, unsigned int dataLen,unsigned int numData)
	cdef void batchChromaticKey(kint *key, float *rgb, unsigned int dataLen, unsigned int numData)

	cdef unsigned int calcKeyLen(unsigned int dataLen)
	cdef void addIndexToKey(kint * key, unsigned int index)
	cdef unsigned int checkIndex(kint * key, unsigned int index)
	cdef void clearKey(kint *key, unsigned int keyLength)
	cdef void printKeyArr(kint *key, unsigned int length)

def offByNArray(np.ndarray[np.uint32_t,ndim=2,mode="c"] keyArray not None,np.ndarray[np.uint32_t,ndim=1,mode="c"] testKey not None, int n):
	cdef unsigned int keyLength = keyArray.shape[1]
	cdef unsigned int numKeys = keyArray.shape[0]
	assert(keyLength == testKey.shape[0])
	assert(n > -1)
	return checkOffByNArray(<kint *> keyArray.data, <kint *> testKey.data, numKeys, keyLength, n)

def offByNArrayBatch(np.ndarray[np.uint32_t,ndim=2,mode="c"] keyArray not None,np.ndarray[np.uint32_t,ndim=2,mode="c"] testKeys not None, int n,**kwarg):
	cdef unsigned int keyLength = keyArray.shape[1]
	cdef unsigned int numKeys = keyArray.shape[0]
	cdef unsigned int numTestKeys = testKeys.shape[0]

	if(kwarg and kwarg['numProc'] != None):
			numProc = kwarg['numProc']
	else:
		numProc = multiprocessing.cpu_count()

	assert(keyLength == testKeys.shape[1])
	assert(n > -1)
	cdef np.ndarray[np.int32_t,ndim=1] results = np.zeros(numTestKeys, dtype=np.int32)
	return batchCheckOffByN(<kint *> keyArray.data, <kint *> testKeys.data, numKeys, keyLength, n, numTestKeys, <int *> results.data, numProc)

def getMinGraphDist(np.ndarray[np.uint32_t,ndim=2,mode="c"] keyArray not None,np.ndarray[np.uint32_t,ndim=1,mode="c"] testKey not None, int n):
	cdef unsigned int keyLength = keyArray.shape[1]
	cdef unsigned int numKeys = keyArray.shape[0]
	assert(keyLength == testKey.shape[0])
	assert(n > -1)
	return getMinGraphDistance(<kint *> keyArray.data, <kint *> testKey.data, numKeys, keyLength, n)

def getMinGraphDistBatch(np.ndarray[np.uint32_t,ndim=2,mode="c"] keyArray not None,np.ndarray[np.uint32_t,ndim=2,mode="c"] testKeys not None, int n,**kwarg):
	cdef unsigned int keyLength = keyArray.shape[1]
	cdef unsigned int numKeys = keyArray.shape[0]
	cdef unsigned int numTestKeys = testKeys.shape[0]

	if(kwarg and kwarg['numProc'] != None):
			numProc = kwarg['numProc']
	else:
		numProc = multiprocessing.cpu_count()

	assert(keyLength == testKeys.shape[1])
	assert(n > -1)
	cdef np.ndarray[np.int32_t,ndim=1] results = np.zeros(numTestKeys, dtype=np.int32)
	return batchGetMinGraphDistance(<kint *> keyArray.data, <kint *> testKeys.data, numKeys, keyLength, n, numTestKeys, <int *> results.data, numProc)

def getGraphDistances(np.ndarray[np.uint32_t,ndim=2,mode="c"] keyArray not None,np.ndarray[np.uint32_t,ndim=1,mode="c"] testKey not None, int n,**kwarg):
	cdef unsigned int keyLength = keyArray.shape[1]
	cdef unsigned int numKeys = keyArray.shape[0]

	assert(keyLength == testKeys.shape[1])
	cdef np.ndarray[np.int32_t,ndim=1] results = np.zeros(numKeys, dtype=np.int32)
	getGraphDist(<kint *> keyArray.data, <kint *> testKeys.data, numKeys, keyLength, <int *> results.data)



def convertToRGB(np.ndarray[int,ndim=1,mode="c"] b not None):
	cdef unsigned int dim
	dim = b.shape[0]
	keyLen = calcKeyLen(dim)
	cdef kint *_key = <kint *>malloc(keyLen * sizeof(kint))
	if not _key:
		raise MemoryError()		
	convertFromIntToKey(&b[0],_key,dim)
	cdef np.ndarray[np.float32_t,ndim=1] chromaKey = np.zeros([3], dtype=np.float32)

	chromaticKey(_key, <float *> chromaKey.data, dim)
	free(_key)
	return chromaKey

def readKey(np.ndarray[np.uint32_t,ndim=1,mode="c"] compressedKey not None,unsigned int dataLen):
	cdef np.ndarray[np.int32_t,ndim=1] rawKey = np.zeros([dataLen], dtype=np.int32)
	convertFromKeyToInt(<kint *> compressedKey.data, <int *> rawKey.data, dataLen)
	return rawKey

def pyCalcKeyLen(unsigned int dataLen):
	return calcKeyLen(dataLen)