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
	ipCache * allocateCache(nnLayer *layer0, float threshold, long long int free)
	void freeCache(ipCache *cache)
	void getInterSigBatch(ipCache *cache, float *data, kint *ipSignature, unsigned int numData, unsigned int numProc)
	void traceDistsSigBatch(ipCache *cache, float *data, kint *ipSigTraces, float * dists, unsigned int numData, unsigned int numProc);


cdef class ipCalculator:
	cdef ipCache * cache
	cdef nnLayer * layer
	cdef unsigned int keyLen
	cdef unsigned int outDim
	cdef unsigned int inDim
	def __cinit__(self,np.ndarray[float,ndim=2,mode="c"] A not None, np.ndarray[float,ndim=1,mode="c"] b not None, float threshold):
		self.outDim = A.shape[1]
		self.inDim  = A.shape[0]
		self.keyLen = calcKeyLen(self.inDim)
		print(self.keyLen)
		self.layer = createLayer(&A[0,0],&b[0],self.outDim,self.inDim)
		freeMemory = psutil.virtual_memory().free
		self.cache = allocateCache(self.layer,threshold,freeMemory)

		if not self.cache:
			raise MemoryError()

	def calculateUncompressed(self,np.ndarray[float,ndim=1,mode="c"] data not None):
		cdef np.ndarray[np.uint64_t,ndim=1] ipSignature = np.zeros([self.keyLen], dtype=np.uint64)        
		getInterSigBatch(self.cache,<float *> data.data,<kint * >ipSignature.data, 1, 1)
		return ipSignature

	def batchCalculateUncompressed(self,np.ndarray[float,ndim=2,mode="c"] data not None, numProc=None):
		if numProc == None:
			numProc = multiprocessing.cpu_count()
		if numProc > multiprocessing.cpu_count():
			eprint("WARNING: Specified too many cores. Reducing to the number you actually have.")
			numProc = multiprocessing.cpu_count()



		numData = data.shape[0]
		cdef np.ndarray[np.uint64_t,ndim=2] ipSignature = np.zeros([numData,self.keyLen], dtype=np.uint64)        
		getInterSigBatch(self.cache,<float *> data.data,<kint * >ipSignature.data, numData, numProc)
		return ipSignature
	
	def traceCalculateUncompressed(self,np.ndarray[float,ndim=1,mode="c"] data not None):
		cdef np.ndarray[np.uint64_t,ndim=1] ipSigTrace = np.zeros([self.outDim,self.keyLen], dtype=np.uint64)
		cdef np.ndarray[np.float32_t,ndim=1] dists = np.zeros([self.outDim], dtype=np.float32)
		traceDistsSigBatch(self.cache,<float *> data.data,<kint * >ipSigTrace.data,<float *>dists.data, 1, 1)
		return ipSigTrace, dists

	def batchTraceCalculateUncompressed(self,np.ndarray[float,ndim=2,mode="c"] data not None, numProc=None):
		if numProc == None:
			numProc = multiprocessing.cpu_count()
		if numProc > multiprocessing.cpu_count():
			eprint("WARNING: Specified too many cores. Reducing to the number you actually have.")
			numProc = multiprocessing.cpu_count()



		numData = data.shape[0]
		cdef np.ndarray[np.uint64_t,ndim=2] ipSigTraces = np.zeros([numData,self.outDim,self.keyLen], dtype=np.uint64)        
		cdef np.ndarray[np.float32_t,ndim=2] dists = np.zeros([numData,self.outDim], dtype=np.float32)
		traceDistsSigBatch(self.cache,<float *> data.data,<kint * >ipSigTraces.data, <float *>dists.data, numData, numProc)
		return ipSigTraces, dists
	

	#Batch calculate, this is multithreaded and you can specify the number of threads you want to use.
	#It defaults, and takes a maximum of 
	def batchCalculate(self,np.ndarray[float,ndim=2,mode="c"] data not None, numProc=None):
		if numProc == None:
			numProc = multiprocessing.cpu_count()
		if numProc > multiprocessing.cpu_count():
			eprint("WARNING: Specified too many cores. Reducing to the number you actually have.")
			numProc = multiprocessing.cpu_count()

		numData = data.shape[0]

		cdef kint *ipSignature_key = <kint *>malloc(numData * self.keyLen * sizeof(kint))
		if not ipSignature_key:
			raise MemoryError()

		cdef np.ndarray[np.int32_t,ndim=2] ipSignature = np.zeros([numData,self.outDim], dtype=np.int32)
		

		try:	        
			getInterSigBatch(self.cache,&data[0,0],ipSignature_key, numData, numProc)
			batchConvertFromKey(ipSignature_key, <int *> ipSignature.data, self.outDim,numData)
			return ipSignature
		finally:
			free(ipSignature_key)

	

	def batchChromaCalculate(self,np.ndarray[float,ndim=2,mode="c"] data not None, numProc=None):
		if numProc == None:
			numProc = multiprocessing.cpu_count()
		if numProc > multiprocessing.cpu_count():
			eprint("WARNING: Specified too many cores. Reducing to the number you actually have.")
			numProc = multiprocessing.cpu_count()

		numData = data.shape[0]
		cdef kint *ipSignature_key = <kint *>malloc(numData * self.keyLen * sizeof(kint))
		if not ipSignature_key:
			raise MemoryError()

		cdef np.ndarray[np.float32_t,ndim=2] chromaipSignature = np.zeros([numData,3], dtype=np.float32)


		try:	        
			getInterSigBatch(self.cache,&data[0,0],ipSignature_key, numData, numProc)
			batchChromaticKey(ipSignature_key, <float *> chromaipSignature.data, self.outDim,numData)
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
		free(self.layer)