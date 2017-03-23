# cython: profile=True
from __future__ import print_function
import cython
import numpy as np
cimport numpy as np
import multiprocessing
from libc.stdlib cimport malloc, free
import psutil


#to print to stderr to warn the user of usage errors

import sys


def eprint(*args, **kwargs):
	print(*args, file=sys.stderr, **kwargs)

include "nnLayerUtilsWrap.pyx"


	
cdef extern from "../cutils/ipCalculator.h":
	ctypedef struct ipCache:
		pass
	ipCache * allocateCache(nnLayer *layer0, float threshold, int depthRestriction, long unsigned int)
	void freeCache(ipCache *cache)
	void getInterSig(ipCache * cache, float *data, kint *ipSignature)
	void getInterSigBatch(ipCache *cache, float *data, kint *ipSignature, unsigned int numData, unsigned int numProc)

cdef class ipCalculator:
	cdef ipCache * cache
	cdef nnLayer * layer
	cdef unsigned int keyLen
	cdef unsigned int outDim
	cdef unsigned int inDim
	def __cinit__(self,np.ndarray[float,ndim=2,mode="c"] A not None, np.ndarray[float,ndim=1,mode="c"] b not None, float threshold=2,int depthRestriction=3):
		self.outDim = A.shape[1]
		self.inDim  = A.shape[0]
		self.layer = createLayer(&A[0,0],&b[0],self.outDim,self.inDim)
		freeMemory = psutil.virtual_memory().free
		self.cache = allocateCache(self.layer,threshold,depthRestriction,freeMemory)

		if not self.cache:
			raise MemoryError()

	def batchCalculateUncompressed(self,np.ndarray[float,ndim=2,mode="c"] data not None, numProc=None):
		if numProc == None:
			numProc = multiprocessing.cpu_count()
		if numProc > multiprocessing.cpu_count():
			eprint("WARNING: Specified too many cores. Reducing to the number you actually have.")
			numProc = multiprocessing.cpu_count()

		cdef unsigned int dim
		dim = data.shape[1]
		numData = data.shape[0]

		keyLen = calcKeyLen(dim)
		cdef np.ndarray[np.uint64_t,ndim=2] ipSignature = np.zeros([numData,keyLen], dtype=np.uint64)        
		getInterSigBatch(self.cache,<float *> data.data,<kint * >ipSignature.data, numData, numProc)
		return ipSignature
		

	def calculate(self,np.ndarray[float,ndim=1,mode="c"] b not None):
		cdef unsigned int dim
		dim = b.shape[0]
		keyLen = calcKeyLen(dim)
		cdef kint *ipSignature_key = <kint *>malloc(keyLen * sizeof(kint))
		if not ipSignature_key:
			raise MemoryError()		
		cdef int *ipSignature = <int *>malloc(dim * sizeof(int))
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
		dim = data.shape[1]
		numData = data.shape[0]

		keyLen = calcKeyLen(dim)
		cdef kint *ipSignature_key = <kint *>malloc(numData * keyLen * sizeof(kint))
		if not ipSignature_key:
			raise MemoryError()

		cdef np.ndarray[np.int32_t,ndim=2] ipSignature = np.zeros([numData,dim], dtype=np.int32)
		

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
		cdef kint *ipSignature_key = <kint *>malloc(keyLen * sizeof(kint))
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
		cdef kint *ipSignature_key = <kint *>malloc(numData * keyLen * sizeof(kint))
		if not ipSignature_key:
			raise MemoryError()

		cdef np.ndarray[np.float32_t,ndim=2] chromaipSignature = np.zeros([numData,3], dtype=np.float32)


		try:	        
			getInterSigBatch(self.cache,&data[0,0],ipSignature_key, numData, numProc)
			batchChromaticKey(ipSignature_key, <float *> chromaipSignature.data, dim,numData)
			return chromaipSignature
		finally:
			free(ipSignature_key)

	def visualize2d(self,np.ndarray[float,ndim=3,mode="c"] data not None, numProc=None):
		if numProc == None:
			numProc = multiprocessing.cpu_count()
		if numProc > multiprocessing.cpu_count():
			eprint("WARNING: Specified too many cores. Reducing to the number you actually have.")
			numProc = multiprocessing.cpu_count()
		if not data.shape[2] == 2:
			print("Incorrect data shape")
		
		xDim = data.shape[0]
		yDim = data.shape[1]
		cdef kint *ipSignature_key = <kint *>malloc(xDim * yDim * sizeof(kint))
		if not ipSignature_key:
			raise MemoryError()

		cdef np.ndarray[np.float32_t,ndim=3] chromaipSignature = np.zeros([xDim,yDim,3], dtype=np.float32)


		try:	        
			getInterSigBatch(self.cache,<float *>data.data,ipSignature_key, xDim*yDim, numProc)
			batchChromaticKey(ipSignature_key, <float *> chromaipSignature.data,self.outDim+1,xDim*yDim)
			return chromaipSignature
		finally:
			free(ipSignature_key)


	def __dealloc__(self):
		freeCache(self.cache)
		freeLayer(self.layer)