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

def readKey(np.ndarray[np.uint64_t,ndim=1,mode="c"] compressedKey not None,unsigned int dataLen):
	cdef np.ndarray[np.int32_t,ndim=1] rawKey = np.zeros([dataLen], dtype=np.int32)
	convertFromKeyToInt(<kint *> compressedKey.data, <int *> rawKey.data, dataLen)
	return rawKey

def pyCalcKeyLen(unsigned int dataLen):
	return calcKeyLen(dataLen)