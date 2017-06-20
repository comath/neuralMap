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
	
cdef extern from "../cutils/ipTrace.h":
	ctypedef struct traceMemory:
		pass
	ctypedef struct traceCache:
		pass
	ctypedef struct distanceWithIndex:
		int index
		float dist
	traceCache * allocateTraceCache(nnLayer * layer)
	void freeTraceCache(traceCache * tc)

	traceMemory * allocateTraceMB(int m, int n)
	void freeTraceMB(traceMemory * tc)

	void fullTrace(traceCache * tc, traceMemory * tm, float * point, float * dists, kint * intersections)
	void batchFullTrace(traceCache * tc, float * point, float * dists, kint * intersections, int numData, int numProc)

	void ipCalc(traceCache * tc, traceMemory * tm, float * point, float threshold, kint *ipSig)
	void batchIpCalc(traceCache * tc, float * data,  float threshold, kint * ipSigs, int numData, int numProc)

cdef class traceCalc:
	cdef traceCache * tc
	cdef traceMemory * tm
	cdef nnLayer * layer
	cdef unsigned int keyLen
	cdef unsigned int m
	cdef unsigned int n
	def __cinit__(self,np.ndarray[float,ndim=2,mode="c"] A not None, np.ndarray[float,ndim=1,mode="c"] b not None):
		self.m = A.shape[1]
		self.n  = A.shape[0]
		self.keyLen = calcKeyLen(self.m)
		print(self.keyLen)
		self.layer = createLayer(&A[0,0],&b[0],self.m,self.n)
		freeMemory = psutil.virtual_memory().free
		self.tc = allocateTraceCache(self.layer)
		self.tm = allocateTraceMB(self.m,self.n)

		if not (self.tc and self.tm):
			raise MemoryError()

	def getFullTrace(self,np.ndarray[float,ndim=1,mode="c"] data not None, **kwarg):
		cdef np.ndarray[np.uint32_t,ndim=2] ipSigTraces
		cdef np.ndarray[np.float32_t,ndim=2] dists
		cdef np.ndarray[np.float32_t,ndim=2] chromaipSignatures
		cdef np.ndarray[np.int8_t,ndim=2] ipSignaturesUncompressed

		
		if(data.shape[0] != self.m):
			raise ValueError('The data is an incorrect dimension')
		ipSigTraces = np.zeros([self.m,self.keyLen], dtype=np.uint32)
		dists = np.zeros([self.m], dtype=np.float32)        
		fullTrace(self.tc,self.tm,<float *> data.data,<float *> dists.data,<kint * >ipSigTraces.data)
		if(kwarg and kwarg['returnType'] != None):
			if(kwarg['returnType'] =='color'):
				chromaipSignatures = np.zeros([self.m,3], dtype=np.float32)
				batchChromaticKey(<kint * >ipSigTraces.data, <float *> chromaipSignatures.data, self.m,self.m)
				return dists,chromaipSignatures
			if(kwarg['returnType'] =='uncompressed'):
				ipSignaturesUncompressed = np.zeros([self.m,self.m], dtype=np.int8)
				batchConvertFromKeyToChar(<kint * >ipSigTraces.data, <char *> ipSignaturesUncompressed.data, self.m,self.m)
				return dists,ipSignaturesUncompressed
		return dists,ipSigTraces

	def getFullTraces(self,np.ndarray[float,ndim=2,mode="c"] data not None, **kwarg):
		cdef np.ndarray[np.uint32_t,ndim=3] ipSigTraces
		cdef np.ndarray[np.float32_t,ndim=3] dists
		cdef np.ndarray[np.float32_t,ndim=3] chromaipSignatures
		cdef np.ndarray[np.int8_t,ndim=3] ipSignaturesUncompressed

		
		if(data.shape[1] != self.m):
			raise ValueError('The data is of an incorrect dimension')
		if(kwarg and kwarg['numProc'] != None):
			numProc = kwarg['numProc']
		else:
			numProc = multiprocessing.cpu_count()

		numData = data.shape[0]
		ipSigTraces = np.zeros([numData,self.m,self.keyLen], dtype=np.uint32)        
		dists = np.zeros([numData,self.m], dtype=np.float32)
		batchFullTrace(self.tc, <float *> data.data, <float *> dists.data, <kint * >ipSigTraces.data, numData, numProc)
		if(kwarg and kwarg['returnType'] != None):
			if(kwarg['returnType'] =='color'):
				chromaipSignatures = np.zeros([numData,self.m,3], dtype=np.float32)
				batchChromaticKey(<kint * >ipSigTraces.data, <float *> chromaipSignatures.data, self.m,numData*self.m)
				return dists,chromaipSignatures
			if(kwarg['returnType'] =='uncompressed'):
				ipSignaturesUncompressed = np.zeros([numData,self.m,self.m], dtype=np.int8)
				batchConvertFromKeyToChar(<kint * >ipSigTraces.data, <char *> ipSignaturesUncompressed.data, self.m,numData*self.m)
				return dists,ipSignaturesUncompressed
		return dists,ipSigTraces
		
	def getIntersection(self,np.ndarray[float, ndim=1,mode="c"] data not None,float threshold, **kwarg):
		cdef np.ndarray[np.uint32_t,ndim =1] ipSig
		cdef np.ndarray[np.float32_t,ndim =1] chromaipSignatures
		cdef np.ndarray[np.int8_t,ndim=1] ipSignaturesUncompressed
		
		if(data.shape[0] != self.m):
			raise ValueError('The data is an incorrect dimension')
		ipSig = np.zeros([self.keyLen], dtype=np.uint32)
		ipCalc(self.tc,self.tm,<float *> data.data, threshold,<kint * >ipSig.data)
		if(kwarg):
			if(kwarg['output_type']=='color'):
				chromaipSignatures = np.zeros([3], dtype=np.float32)
				batchChromaticKey(<kint * >ipSig.data, <float *> chromaipSignatures.data, self.m,1)
				return chromaipSignatures
			if(kwarg['output_type']=='uncompressed'):
				ipSignaturesUncompressed = np.zeros([self.m], dtype=np.int8)
				batchConvertFromKeyToChar(<kint * >ipSig.data, <char *> ipSignaturesUncompressed.data, self.m,1)
				return ipSignaturesUncompressed
		return ipSig

	def getIntersections(self,np.ndarray[float, ndim=2,mode="c"] data not None,float threshold, *returnType, **kwarg):
		
		cdef np.ndarray[np.float32_t, ndim=2] chromaipSignatures
		cdef np.ndarray[np.int8_t,ndim =2] ipSignaturesUncompressed
		cdef int numProc = 1
		if(not (data.shape[1] == self.m)):
			raise ValueError('The data is an incorrect dimension')
		
		

		if(kwarg and kwarg['numProc'] != None):
			numProc = kwarg['numProc']
		else:
			numProc = multiprocessing.cpu_count()

		cdef int numData = data.shape[0]
		cdef np.ndarray[np.uint32_t, ndim=2] ipSig = np.zeros([numData,self.keyLen], dtype=np.uint32)        
		batchIpCalc(self.tc, <float *> data.data, threshold, <kint * >ipSig.data, numData, numProc)
		print('Done with map')
		if(kwarg and kwarg['returnType'] != None):
			if(kwarg['returnType'] =='color'):
				print('Color Coding')
				chromaipSignatures = np.zeros([numData,3], dtype=np.float32)
				batchChromaticKey(<kint * >ipSig.data, <float *> chromaipSignatures.data, self.m,numData*self.m)
				return chromaipSignatures
			if(kwarg['returnType'] =='uncompressed'):
				print('Decompressing')
				ipSignaturesUncompressed = np.zeros([numData,self.m], dtype=np.int8)
				batchConvertFromKeyToChar(<kint * >ipSig.data, <char *> ipSignaturesUncompressed.data, self.m,numData)
				return ipSignaturesUncompressed
		
		return ipSig
		

	def __dealloc__(self):
		freeTraceCache(self.tc)
		freeTraceMB(self.tm)
		free(self.layer)