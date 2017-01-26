from __future__ import print_function
import cython
import numpy as np
cimport numpy as np
import multiprocessing
from libc.stdlib cimport malloc, free

#to print to stderr to warn the user of usage errors

import sys

def eprint(*args, **kwargs):
	print(*args, file=sys.stderr, **kwargs)

include "nnLayerUtilsWrap.pyx"

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
	
cdef extern from "../cutils/ipCalculator.h":
	ctypedef struct ipCache:
		pass
	ipCache * allocateCache(nnLayer *layer0, float threshold)
	void freeCache(ipCache *cache)
	void getInterSig(ipCache * cache, float *data, unsigned int *ipSignature)
	void getInterSigBatch(ipCache *cache, float *data, unsigned int *ipSignature, unsigned int numData, unsigned int numProc)

cdef class ipCalculator:
	cdef ipCache * cache
	cdef nnLayer * layer
	cdef unsigned int keyLen
	def __cinit__(self,np.ndarray[float,ndim=2,mode="c"] A not None, np.ndarray[float,ndim=1,mode="c"] b not None, float threshold):
		cdef unsigned int outDim = A.shape[0]
		cdef unsigned int inDim  = A.shape[1]
		self.layer = createLayer(&A[0,0],&b[0],outDim,inDim)
		self.cache = allocateCache(self.layer,threshold)

		if not self.cache:
			raise MemoryError()


	def calculate(self,np.ndarray[float,ndim=1,mode="c"] b not None):
		cdef unsigned int dim
		dim = b.shape[0]
		keyLen = calcKeyLen(dim)
		cdef unsigned int *ipSignature_key = <unsigned int *>malloc(keyLen * sizeof(unsigned int))
		if not ipSignature_key:
			raise MemoryError()		
		cdef int *ipSignature = <int *>malloc(dim * sizeof(unsigned int))
		if not ipSignature:
			raise MemoryError()
		try:	        
			getInterSig(self.cache,&b[0],ipSignature_key)
			convertFromKey(ipSignature_key, ipSignature, dim)
			return [ ipSignature[i] for i in range(dim) ]
		finally:
			free(ipSignature)
			free(ipSignature_key)

	#Batch calculate, this is multithreaded and you can specify the number of threads you want to use.
	#It defaults, and takes a maximum of 
	def batchCalculate(self,np.ndarray[float,ndim=2,mode="c"] data not None, numProc=None):
		if numProc == None:
			numProc = multiprocessing.cpu_count()
		if numProc > multiprocessing.cpu_count():
			eprint("WARNING: Specified too many cores. Reducing to the number you actually have.")
			numProc = multiprocessing.cpu_count()

		cdef unsigned int dim
		dim = data.shape[0]
		numData = data.shape[1]
		keyLen = calcKeyLen(dim)
		cdef unsigned int *ipSignature_key = <unsigned int *>malloc(numData * keyLen * sizeof(unsigned int))
		if not ipSignature_key:
			raise MemoryError()

		cdef np.ndarray[np.int32_t,ndim=2] ipSignature = np.zeros([dim,numData], dtype=np.int32)


		try:	        
			getInterSigBatch(self.cache,&data[0,0],ipSignature_key, numData, numProc)
			batchConvertFromKey(ipSignature_key, <int *> ipSignature.data, dim,numData)
			return ipSignature
		finally:
			free(ipSignature_key)

	def chromaCalculate(self,np.ndarray[float,ndim=1,mode="c"] b not None):
		cdef unsigned int dim
		dim = b.shape[0]
		keyLen = calcKeyLen(dim)
		cdef unsigned int *ipSignature_key = <unsigned int *>malloc(keyLen * sizeof(unsigned int))
		if not ipSignature_key:
			raise MemoryError()		
		cdef float *ipSignature = <float *>malloc(dim * sizeof(float))
		if not ipSignature:
			raise MemoryError()
		try:	        
			getInterSig(self.cache,&b[0],ipSignature_key)
			chromaticKey(ipSignature_key, ipSignature, dim)
			return [ ipSignature[i] for i in range(dim) ]
		finally:
			free(ipSignature)
			free(ipSignature_key)

	def batchChromaCalculate(self,np.ndarray[float,ndim=2,mode="c"] data not None, numProc=None):
		if numProc == None:
			numProc = multiprocessing.cpu_count()
		if numProc > multiprocessing.cpu_count():
			eprint("WARNING: Specified too many cores. Reducing to the number you actually have.")
			numProc = multiprocessing.cpu_count()

		cdef unsigned int dim
		dim = data.shape[1]
		numData = data.shape[0]
		keyLen = calcKeyLen(dim)
		cdef unsigned int *ipSignature_key = <unsigned int *>malloc(numData * keyLen * sizeof(unsigned int))
		if not ipSignature_key:
			raise MemoryError()

		cdef np.ndarray[np.float32_t,ndim=2] chromaipSignature = np.zeros([numData,3], dtype=np.float32)


		try:	        
			getInterSigBatch(self.cache,&data[0,0],ipSignature_key, numData, numProc)
			batchChromaticKey(ipSignature_key, <float *> chromaipSignature.data, dim,numData)
			return chromaipSignature
		finally:
			free(ipSignature_key)


	def __dealloc__(self):
		freeCache(self.cache)
		freeLayer(self.layer)