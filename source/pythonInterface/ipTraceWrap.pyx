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

	void ipCalc(traceCache * tc, traceMemory * tm, float * point, kint *ipSig, float threshold)
	void batchIpCalc(traceCache * tc, float * data, kint * ipSigs, float threshold, int numData, int numProc)

cdef class traceCalc:
	cdef traceCache * tc
	cdef traceMemory * tm
	cdef nnLayer * layer
	cdef unsigned int keyLen
	cdef unsigned int m
	cdef unsigned int n
	def __cinit__(self,np.ndarray[float,ndim=2,mode="c"] A not None, np.ndarray[float,ndim=1,mode="c"] b not None, float threshold):
		self.m = A.shape[1]
		self.n  = A.shape[0]
		self.keyLen = calcKeyLen(self.n)
		print(self.keyLen)
		self.layer = createLayer(&A[0,0],&b[0],self.m,self.n)
		freeMemory = psutil.virtual_memory().free
		self.tc = allocateTraceCache(self.layer)
		self.tm = allocateTraceMB(self.m,self.n)

		if not (self.tc and self.tm):
			raise MemoryError()

	def getFullTrace(self,np.ndarray[float,mode="c"] data not None, **kwarg):
		cdef np.ndarray[np.uint64_t] ipSigTraces
		cdef np.ndarray[np.float32_t] dists
		cdef np.ndarray[np.float32_t] chromaipSignatures
		cdef np.ndarray[np.int8_t] ipSignaturesUncompressed

		if(data.ndim==1):
			if(data.shape[0] != self.n):
				raise ValueError('The data is an incorrect dimension')
			ipSigTraces = np.zeros([self.outDim,self.keyLen], dtype=np.uint32)
			dists = np.zeros([self.outDim], dtype=np.float32)        
			fullTrace(self.tc,self.tm,<float *> data.data,<float *> dists.data,<kint * >ipSigTraces.data)
			if(kwarg):
				if(kwarg['output_type']=='color'):
					chromaipSignatures = np.zeros([self.outDim,3], dtype=np.float32)
					batchChromaticKey(<kint * >ipSigTraces.data, <float *> chromaipSignatures.data, self.outDim,self.outDim)
					return dists,chromaipSignatures
				if(kwarg['output_type']=='uncompressed'):
					ipSignaturesUncompressed = np.zeros([self.outDim,self.outDim], dtype=np.char)
					batchConvertFromKeyChar(<kint * >ipSigTraces.data, <char *> ipSignaturesUncompressed.data, self.outDim,self.outDim)
					return dists,ipSignaturesUncompressed
			return dists,ipSigTraces
		elif (data.ndim==2):
			if(data.shape[1] != self.n):
				raise ValueError('The data is of an incorrect dimension')
			if(kwarg and kwarg['numProc'] != None):
				numProc = kwarg['numProc']
			else:
				numProc = multiprocessing.cpu_count()

			numData = data.shape[0]
			ipSigTraces = np.zeros([numData,self.outDim,self.keyLen], dtype=np.uint32)        
			dists = np.zeros([numData,self.outDim], dtype=np.float32)
			batchFullTrace(self.tc, <float *> data.data, <float *> dists.data, <kint * >ipSigTraces.data, numData, numProc)
			if(kwarg):
				if(kwarg['output_type']=='color'):
					chromaipSignatures = np.zeros([numData,self.outDim,3], dtype=np.float32)
					batchChromaticKey(<kint * >ipSigTraces.data, <float *> chromaipSignatures.data, self.outDim,numData*self.outDim)
					return dists,chromaipSignatures
				if(kwarg['output_type']=='uncompressed'):
					ipSignaturesUncompressed = np.zeros([numData,self.outDim,self.outDim], dtype=np.char)
					batchConvertFromKeyChar(<kint * >ipSigTraces.data, <char *> ipSignaturesUncompressed.data, self.outDim,numData*self.outDim)
					return dists,ipSignaturesUncompressed
			return dists,ipSigTraces
		else:
			raise ValueError('The data must be either a rank 2 batch, or a single data vector')

	def getIntersection(self,np.ndarray[float,mode="c"] data not None,float threshold, **kwarg):
		cdef np.ndarray[np.uint64_t] ipSig
		cdef np.ndarray[np.float32_t] chromaipSignatures
		cdef np.ndarray[np.int8_t] ipSignaturesUncompressed
		if(data.ndim==1):
			if(data.shape[0] != self.n):
				raise ValueError('The data is an incorrect dimension')
			ipSig = np.zeros([self.keyLen], dtype=np.uint32)
			ipCalc(self.tc,self.tm,<float *> data.data,<kint * >ipSig.data, threshold)
			if(kwarg):
				if(kwarg['output_type']=='color'):
					chromaipSignatures = np.zeros([3], dtype=np.float32)
					batchChromaticKey(<kint * >ipSig.data, <float *> chromaipSignatures.data, self.outDim,1)
					return chromaipSignatures
				if(kwarg['output_type']=='uncompressed'):
					ipSignaturesUncompressed = np.zeros([self.outDim], dtype=np.char)
					batchConvertFromKeyChar(<kint * >ipSig.data, <char *> ipSignaturesUncompressed.data, self.outDim,1)
					return ipSignaturesUncompressed
			return ipSig
		elif (data.ndim==2):
			if(data.shape[1] != self.n):
				raise ValueError('The data is an incorrect dimension')
			if(kwarg and kwarg['numProc'] != None):
				numProc = kwarg['numProc']
			else:
				numProc = multiprocessing.cpu_count()

			numData = data.shape[0]
			ipSig = np.zeros([numData,self.keyLen], dtype=np.uint32)        
			batchIpCalc(self.tc, <float *> data.data, <kint * >ipSig.data, threshold, numData, numProc)
			if(kwarg):
				if(kwarg['output_type']=='color'):
					chromaipSignatures = np.zeros([numData,3], dtype=np.float32)
					batchChromaticKey(<kint * >ipSig.data, <float *> chromaipSignatures.data, self.outDim,numData*self.outDim)
					return chromaipSignatures
				if(kwarg['output_type']=='uncompressed'):
					ipSignaturesUncompressed = np.zeros([numData,self.outDim], dtype=np.char)
					batchConvertFromKeyChar(<kint * >ipSig.data, <char *> ipSignaturesUncompressed.data, self.outDim,numData)
					return ipSignaturesUncompressed
			return ipSig
		else:
			raise ValueError('The data must be either a batch of data vectors, or a single data vector')


	def __dealloc__(self):
		freeTraceCache(self.tc)
		freeTraceMB(self.tm)
		free(self.layer)