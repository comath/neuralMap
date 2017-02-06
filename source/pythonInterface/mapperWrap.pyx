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
include "keyWrap.pyx"

cdef extern from "../cutils/mapper.h":
	ctypedef struct location:
		unsigned int *ipSig
		unsigned int *regSig
		unsigned int numPoints
		unsigned int numErrorPoints
		float *avgPoint
		float *avgErrorPoint

	ctypedef struct _nnMap:
		pass

	cdef _nnMap * allocateMap(nnLayer *layer0, nnLayer *layer1, float threshold, float errorThreshhold)
	cdef void freeMap(_nnMap * internalMap)

	cdef void addDatumToMap(_nnMap * internalMap, float *datum, float errorMargin)
	cdef void addDataToMapBatch(_nnMap * internalMap, float *data, float * errorMargins, unsigned int numData, unsigned int numProc)

	cdef unsigned int numLoc(_nnMap * internalMap)
	cdef location getMaxErrorLoc(_nnMap * internalMap)
	cdef location * getLocationArray(_nnMap * internalMap)

cdef class nnMap:
	cdef _nnMap * internalMap
	cdef nnLayer * layer0
	cdef nnLayer * layer1
	cdef unsigned int keyLen
	cdef unsigned int dim
	cdef unsigned int numLocations
	cdef location * locArr

	def __cinit__(self,np.ndarray[float,ndim=2,mode="c"] A0 not None, np.ndarray[float,ndim=1,mode="c"] b0 not None,
					   np.ndarray[float,ndim=2,mode="c"] A1 not None, np.ndarray[float,ndim=1,mode="c"] b1 not None,
					   float threshold, float errorThreshhold):
		cdef unsigned int outDim0 = A0.shape[0]
		cdef unsigned int inDim0  = A0.shape[1]
		dim = inDim0
		cdef unsigned int outDim1 = A1.shape[0]
		cdef unsigned int inDim1  = A1.shape[1]
		self.layer0 = createLayer(&A0[0,0],&b0[0],outDim0,inDim0)
		if not self.layer0:
			raise MemoryError()
		self.layer1 = createLayer(&A1[0,0],&b1[0],outDim1,inDim1)
		if not self.layer1:
			raise MemoryError()
		self.internalMap = allocateMap(self.layer0,self.layer1,threshold,errorThreshhold)
		if not self.internalMap:
			raise MemoryError()


	def add(self,np.ndarray[float,ndim=1,mode="c"] b not None, float errorMargin):       
		addDatumToMap(self.internalMap,<float *> b.data, errorMargin)
		self.numLocations = numLoc(self.internalMap)	

	#Batch calculate, this is multithreaded and you can specify the number of threads you want to use.
	#It defaults to the number of virtual cores you have
	def batchAdd(self,np.ndarray[float,ndim=2,mode="c"] data not None, np.ndarray[float,ndim=1,mode="c"] errorMargins not None, numProc=None):
		if numProc == None:
			numProc = multiprocessing.cpu_count()
		if numProc > multiprocessing.cpu_count():
			eprint("WARNING: Specified too many cores. Reducing to the number you actually have.")
			numProc = multiprocessing.cpu_count()

		numData = data.shape[0]       
		addDataToMapBatch(self.internalMap,<float *> data.data, <float *> errorMargins.data, numData, numProc)
		self.numLocations = numLoc(self.internalMap)

	def location(self,int i):
		if not self.locArr:
			self.numLocations = numLoc(self.internalMap)
			self.locArr = getLocationArray(self.internalMap)
			if not self.locArr:
				raise MemoryError()
		if(i > self.numLocations):
			eprint("Index out of bounds")
			raise MemoryError()
		cdef location thisLoc = self.locArr[i]
		cdef np.ndarray[np.int32_t,ndim=1] ipSignature = np.zeros([self.dim], dtype=np.int32)
		cdef np.ndarray[np.int32_t,ndim=1] regSignature = np.zeros([self.dim], dtype=np.int32)
		cdef np.ndarray[np.float32_t,ndim=1] avgPoint = np.zeros([self.dim], dtype=np.float32)
		cdef np.ndarray[np.float32_t,ndim=1] avgErrorPoint = np.zeros([self.dim], dtype=np.float32)
		convertFromKey(thisLoc.ipSig, <int *> ipSignature.data, self.dim)
		convertFromKey(thisLoc.regSig, <int *> regSignature.data, self.dim)
		for i in range(self.dim):
			avgPoint[i] = thisLoc.avgPoint[i]
			avgErrorPoint[i] = thisLoc.avgErrorPoint[i]
		return ipSignature,regSignature,avgPoint,avgErrorPoint,thisLoc.numPoints,thisLoc.numErrorPoints
	
#	def __dealloc__(self):
#		freeMap(self.internalMap)
#		freeLayer(self.layer0)
#		freeLayer(self.layer1)
#		if self.locArr:
#			free(self.locArr)
	