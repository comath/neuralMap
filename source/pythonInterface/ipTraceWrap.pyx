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
	void batchIpCalc(traceCache * tc, float * data, float * dists, kint * ipSigs, float threshold, int numData, int numProc)

cdef class traceCalc:
	cdef traceCache * tc
	cdef traceMemory * tm
	cdef unsigned int keyLen
	cdef unsigned int m
	cdef unsigned int n
	def __cinit__(self,np.ndarray[float,ndim=2,mode="c"] A not None, np.ndarray[float,ndim=1,mode="c"] b not None, float threshold):
		self.m = A.shape[1]
		self.n  = A.shape[0]
		self.keyLen = calcKeyLen(self.n)
		print(self.keyLen)
		layer = createLayer(&A[0,0],&b[0],self.m,self.n)
		freeMemory = psutil.virtual_memory().free
		self.tc = allocateTraceCache(layer)
		self.tm = allocateTraceMB(m,n)

		if not (self.tc and self.tm):
			raise MemoryError()

	def getTrace(self,np.ndarray[float,mode="c"] data not None, **kwarg):
		if(len(data.shape)==1):
			if(data.shape[0] != self.n):
				raise ValueError('The data is an incorrect dimension')
			cdef np.ndarray[np.uint32_t,ndim=2] ipSigTrace = np.zeros([self.outDim,self.keyLen], dtype=np.uint32)
			cdef np.ndarray[np.float32_t,ndim=1] dists = np.zeros([self.outDim], dtype=np.float32)        
			fullTrace(self.tc,self.tm,<float *> data.data,<float *> dists.data,<kint * >ipSigTrace.data)
			if(kwarg):
				if(kwarg['output_type']=='color'):
					cdef np.ndarray[np.float32_t,ndim=2] chromaipSignature = np.zeros([self.outDim,3], dtype=np.float32)
					batchChromaticKey(<kint * >ipSigTrace.data, <float *> chromaipSignature.data, self.outDim,self.outDim)
					return dists,chromaipSignature
				if(kwarg['output_type']=='uncompressed'):
					cdef np.ndarray[np.char_t,ndim=2] ipSignatureUncompressed = np.zeros([self.outDim,self.outDim], dtype=np.char)
					batchConvertFromKey(<kint * >ipSigTrace.data, <char *> ipSignatureUncompressed.data, self.outDim,self.outDim)
					return dists,ipSignatureUncompressed
			return dists,ipSigTrace
		elif (len(data.shape)==2):
			if(data.shape[1] != self.n):
				raise ValueError('The data is an incorrect dimension')
			if numProc == None:
				numProc = multiprocessing.cpu_count()
			if numProc > multiprocessing.cpu_count():
				eprint("WARNING: Specified too many cores. Reducing to the number you actually have.")
				numProc = multiprocessing.cpu_count()

			numData = data.shape[0]
			cdef np.ndarray[np.uint32_t,ndim=3] ipSigTraces = np.zeros([numData,self.outDim,self.keyLen], dtype=np.uint32)        
			cdef np.ndarray[np.float32_t,ndim=2] dists = np.zeros([numData,self.outDim], dtype=np.float32)
			batchFullTrace(self.tc, <float *> data.data, <float *> dists.data, <kint * >ipSigTraces.data, numData, numProc)
			if(kwarg):
				if(kwarg['output_type']=='color'):
					cdef np.ndarray[np.float32_t,ndim=3] chromaipSignature = np.zeros([numData,self.outDim,3], dtype=np.float32)
					batchChromaticKey(<kint * >ipSigTrace.data, <float *> chromaipSignature.data, self.outDim,numData*self.outDim)
					return dists,chromaipSignature
				if(kwarg['output_type']=='uncompressed'):
					cdef np.ndarray[np.char_t,ndim=3] ipSignatureUncompressed = np.zeros([numData,self.outDim,self.outDim], dtype=np.char)
					batchConvertFromKey(<kint * >ipSigTrace.data, <char *> ipSignatureUncompressed.data, self.outDim,numData*self.outDim)
					return dists,ipSignatureUncompressed
			return dists,ipSigTraces
		else:
			raise ValueError('The data must be either a rank 2 batch, or a single data vector')

	def getIntersection(self,np.ndarray[float,mode="c"] data not None,float threshold):
		if(len(data.shape)==1):
			if(data.shape[0] != self.n):
				raise ValueError('The data is an incorrect dimension')
			cdef np.ndarray[np.uint32_t,ndim=1] ipSig = np.zeros([self.keyLen], dtype=np.uint32)
			fullTrace(self.tc,self.tm,<float *> data.data,<float *> dists.data,<kint * >ipSigTrace.data)
			if(kwarg):
				if(kwarg['output_type']=='color'):
					cdef np.ndarray[np.float32_t,ndim=3] chromaipSignature = np.zeros([3], dtype=np.float32)
					batchChromaticKey(<kint * >ipSigTrace.data, <float *> chromaipSignature.data, self.outDim,1)
					return dists,chromaipSignature
				if(kwarg['output_type']=='uncompressed'):
					cdef np.ndarray[np.char_t,ndim=3] ipSignatureUncompressed = np.zeros([numData,self.outDim,self.outDim], dtype=np.char)
					batchConvertFromKey(<kint * >ipSigTrace.data, <char *> ipSignatureUncompressed.data, self.outDim,1)
					return dists,ipSignatureUncompressed
			return dists,ipSigTrace
		elif (len(data.shape)==2):
			if(data.shape[1] != self.n):
				raise ValueError('The data is an incorrect dimension')
			if numProc == None:
				numProc = multiprocessing.cpu_count()
			if numProc > multiprocessing.cpu_count():
				eprint("WARNING: Specified too many cores. Reducing to the number you actually have.")
				numProc = multiprocessing.cpu_count()

			numData = data.shape[0]
			cdef np.ndarray[np.uint32_t,ndim=2] ipSig = np.zeros([numData,self.keyLen], dtype=np.uint32)        
			batchFullTrace(self.tc, <float *> data.data, <float *> dists.data, <kint * >ipSigTraces.data, numData, numProc)
			if(kwarg):
				if(kwarg['output_type']=='color'):
					cdef np.ndarray[np.float32_t,ndim=3] chromaipSignature = np.zeros([numData,3], dtype=np.float32)
					batchChromaticKey(<kint * >ipSigTrace.data, <float *> chromaipSignature.data, self.outDim,numData*self.outDim)
					return dists,chromaipSignature
				if(kwarg['output_type']=='uncompressed'):
					cdef np.ndarray[np.char_t,ndim=3] ipSignatureUncompressed = np.zeros([numData,self.outDim], dtype=np.char)
					batchConvertFromKey(<kint * >ipSigTrace.data, <char *> ipSignatureUncompressed.data, self.outDim,numData)
					return dists,ipSignatureUncompressed
			return dists,ipSigTraces
		else:
			raise ValueError('The data must be either a rank 2 batch, or a single data vector')


	def __dealloc__(self):
		freeTraceCache(self.tc)
		freeTraceMB(self.tm)
		free(self.layer)