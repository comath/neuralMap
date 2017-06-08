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

cdef extern from "../cutils/vector.h":
	ctypedef struct vector:
		void ** items
		int capacity
		int total
	cdef void vector_init(vector *)
	cdef int vector_total(vector *)
	cdef void vector_add(vector *, void *)
	cdef void vector_set(vector *, int, void *)
	cdef void *vector_get(vector *, int)
	cdef void vector_delete(vector *, int)
	cdef void vector_free(vector *)

cdef extern from "../cutils/location.h":
	ctypedef struct pointInfo:
		int * traceRaw
		float * traceDists
		long int index
	ctypedef struct location:
		float *traceDists
		int *traceRaws
		int *pointIndexes
		int m
		int capacity
		int total
	cdef void location_init(location *)
	cdef int location_total(location *)
	cdef void location_add(location *, pointInfo *)
	cdef void location_set(location *, int, pointInfo *)
	cdef pointInfo location_get(location *, int)
	cdef void location_delete(location *, int)
	cdef void location_free(location *)

cdef extern from "../cutils/mapper.h":
	ctypedef struct _nnMap:
		pass

	cdef _nnMap * allocateMap(nnLayer *layer0, float threshold, float errorThreshhold)
	cdef void freeMap(_nnMap * internalMap)

	cdef void addDatumToMap(_nnMap * internalMap, float *datum, float errorMargin)
	cdef void addDataToMapBatch(_nnMap * internalMap, float *data, float * errorMargins, unsigned int numData, unsigned int numProc)

	cdef location getPointsAt(_nnMap *map, kint *keyPair)
	cdef unsigned int numLoc(_nnMap * map)
	cdef mapTreeNode ** getLocations(_nnMap *map, char orderBy)

cdef extern from "../cutils/adaptiveTools.h":
	ctypedef struct maxPopGroupData:
		mapTreeNode ** nodes
		kint * hpCrossed
		int count
		int selectionIndex

	cdef maxPopGroupData * refineMapAndGetMax(mapTreeNode ** locArr, int maxLocIndex, nnLayer * selectionLayer)

cdef class _location:
	cdef unsigned int outDim 
	cdef unsigned int inDim 
	cdef location * thisLoc
	@staticmethod
	cdef _location create(location* ptr, unsigned int outDim, unsigned int inDim):
		obj = _location() # create instance without calling __init__
		obj.thisLoc = ptr
		obj.outDim = outDim
		obj.inDim = inDim
		return obj
	def ipSig(self):
		cdef np.ndarray[np.int32_t,ndim=1] ipSignature = np.zeros([self.outDim], dtype=np.int32)
		convertFromKey(self.thisLoc.ipSig, <int *> ipSignature.data, self.outDim)
		return ipSignature
	def regSig(self):
		cdef np.ndarray[np.int32_t,ndim=1] regSignature = np.zeros([self.outDim], dtype=np.int32)
		convertFromKey(self.thisLoc.regSig, <int *> regSignature.data, self.outDim)
		return regSignature
	def pointIndexes(self):
		numPoints = location_total(thisloc)
		cdef np.ndarray[np.float32_t,ndim=2] points = np.zeros([numPoints,self.inDim], dtype=np.float32)
		pointsAsArray(thisloc,points.data,inDim)
		return points
	

cdef class nnMap:
	cdef _nnMap * internalMap
	cdef nnLayer * layer0
	cdef nnLayer * layer1
	cdef unsigned int keyLen
	cdef unsigned int numLoc
	cdef location * locArr

	def __cinit__(self,np.ndarray[float,ndim=2,mode="c"] A0 not None, np.ndarray[float,ndim=1,mode="c"] b0 not None,
					   float threshold, float errorThreshhold):
		cdef unsigned int outDim0 = A0.shape[1]
		cdef unsigned int inDim0  = A0.shape[0]
		self.layer0 = createLayer(&A0[0,0],&b0[0],outDim0,inDim0)
		if not self.layer0:
			raise MemoryError()
		self.threshold = threshold
		self.internalMap = allocateMap(self.layer0)
		if not self.internalMap:
			raise MemoryError()


	def add(self,np.ndarray[float,ndim=1,mode="c"] b not None, float errorMargin):
		if self.locArr:
			self.numLoc = 0
			free(self.locArr)   
		addDatumToMap(self.internalMap,<float *> b.data, errorMargin)
		
	#Batch calculate, this is multithreaded and you can specify the number of threads you want to use.
	#It defaults to the number of virtual cores you have
	def batchAdd(self,np.ndarray[float,ndim=2,mode="c"] data not None, np.ndarray[float,ndim=1,mode="c"] errorMargins not None, numProc=None):
		if numProc == None:
			numProc = multiprocessing.cpu_count()
		if numProc > multiprocessing.cpu_count():
			eprint("WARNING: Specified too many cores. Reducing to the number you actually have.")
			numProc = multiprocessing.cpu_count()
		if self.locArr:
			self.numLoc = 0
			free(self.locArr)

		numData = data.shape[0]
		if(data.shape[1] != self.layer0.inDim):
			eprint("Data is of the wrong dimension.")   
		addDataToMapBatch(self.internalMap,<float *> data.data, self.threshold,numData, numProc)
		
	def numLocations(self):
		if not self.locArr:
			self.numLoc = numLoc(self.internalMap)
			self.locArr = getLocationArray(self.internalMap)
			if not self.locArr:
				raise MemoryError()
		return self.numLoc

	def location(self,int i):
		if not self.locArr:
			self.numLoc = numLoc(self.internalMap)
			self.locArr = getLocationArray(self.internalMap)
			if not self.locArr:
				raise MemoryError()
		if(i > self.numLoc):
			eprint("Index out of bounds")
			raise MemoryError()
		return _location.create( &(self.locArr[i]) , self.layer0.outDim, self.layer0.inDim)
	
	def __dealloc__(self):
		freeMap(self.internalMap)
		if self.locArr:
			free(self.locArr)
	