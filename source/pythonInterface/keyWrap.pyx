import cython
import numpy as np
cimport numpy as np
import multiprocessing
from libc.stdlib cimport malloc, free

#to print to stderr to warn the user of usage errors

import sys


cdef extern from "../cutils/key.h":
	cdef char compareKey(unsigned int *x, unsigned int *y, unsigned int keyLength)
	
	cdef void convertToKey(int * raw, unsigned int *key,unsigned int dataLen)
	cdef void convertFromKey(unsigned int *key, int * output, unsigned int dataLen)
	cdef void chromaticKey(unsigned int* key, float *rgb, unsigned int dataLen)

	cdef void batchConvertToKey(int * raw, unsigned int *key,unsigned int dataLen, unsigned int numData)
	cdef void batchConvertFromKey(unsigned int *key, int * output, unsigned int dataLen,unsigned int numData)
	cdef void batchChromaticKey(unsigned int* key, float *rgb, unsigned int dataLen, unsigned int numData)

	cdef unsigned int calcKeyLen(unsigned int dataLen)
	cdef void addIndexToKey(unsigned int * key, unsigned int index)
	cdef unsigned int checkIndex(unsigned int * key, unsigned int index)
	cdef void clearKey(unsigned int *key, unsigned int keyLength)
	cdef void printKeyArr(unsigned int *key, unsigned int length)

def convertToRGB(np.ndarray[int,ndim=1,mode="c"] b not None):
	cdef unsigned int dim
	dim = b.shape[0]
	keyLen = calcKeyLen(dim)
	cdef unsigned int *_key = <unsigned int *>malloc(keyLen * sizeof(unsigned int))
	if not _key:
		raise MemoryError()		
	convertToKey(&b[0],_key,dim)
	cdef np.ndarray[np.float32_t,ndim=1] chromaKey = np.zeros([3], dtype=np.float32)

	chromaticKey(_key, <float *> chromaKey.data, dim)
	free(_key)
	return chromaKey