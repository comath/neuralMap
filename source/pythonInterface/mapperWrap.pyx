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

cdef extern from "pthread.h" nogil:
	ctypedef struct pthread_mutex_t:
		pass

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

cdef extern from "../cutils/mapperTree.h":
	ctypedef struct mapTreeNode:
		pass
	cdef void nodeGetIPKey(mapTreeNode * node, int * ipKey, unsigned int outDim)
	cdef void nodeGetRegKey(mapTreeNode * node, int * regKey, unsigned int outDim)
	cdef void nodeGetPointIndexes(mapTreeNode * node, int *indexHolder)
	cdef int nodeGetTotal(mapTreeNode * node)

cdef extern from "../cutils/mapper.h":
	ctypedef struct _nnMap:
		int createdKL
		kint *ipKey
		kint *regKey

		pthread_mutex_t datamutex
		location loc

	cdef _nnMap * allocateMap(nnLayer *layer0)
	cdef void freeMap(_nnMap * internalMap)

	cdef void addPointToMap(_nnMap * map, float *point, int pointIndex, float threshold)
	cdef void addDataToMapBatch(_nnMap * map, float *data, int *indexes, float threshold, unsigned int numData, unsigned int numProc)

	cdef location getPointsAt(_nnMap *map, kint *keyPair)
	cdef unsigned int numLoc(_nnMap * map)
	cdef mapTreeNode ** getLocations(_nnMap *map, char orderBy)
	void location_get_indexes(location *v, int *indexHolder)


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
	cdef mapTreeNode * thisLoc
	@staticmethod
	cdef _location create(mapTreeNode* ptr, unsigned int outDim, unsigned int inDim):
		obj = _location() # create instance without calling __init__
		obj.thisLoc = ptr
		obj.outDim = outDim
		obj.inDim = inDim
		return obj

	def ipSig(self):
		cdef np.ndarray[np.int32_t,ndim=1] ipSignature = np.zeros([self.outDim], dtype=np.int32)
		nodeGetIPKey(self.thisLoc, <int *> ipSignature.data, self.outDim)
		return ipSignature

	def regSig(self):
		cdef np.ndarray[np.int32_t,ndim=1] regSignature = np.zeros([self.outDim], dtype=np.int32)
		nodeGetRegKey(self.thisLoc, <int *> regSignature.data, self.outDim)
		return regSignature
		
	def pointIndexes(self):
		numPoints =	nodeGetTotal(self.thisLoc)
		cdef np.ndarray[np.float32_t,ndim=2] points = np.zeros([numPoints,self.inDim], dtype=np.float32)
		nodeGetPointIndexes(self.thisLoc, <int *>points.data)
		return points
	

cdef class nnMap:
	cdef _nnMap * internalMap
	cdef nnLayer * layer0
	cdef nnLayer * layer1
	cdef unsigned int keyLen
	cdef unsigned int numLoc
	cdef mapTreeNode ** locArr

	def __cinit__(self,np.ndarray[float,ndim=2,mode="c"] A0 not None, np.ndarray[float,ndim=1,mode="c"] b0 not None,
					   float threshold):
		cdef unsigned int outDim0 = A0.shape[1]
		cdef unsigned int inDim0  = A0.shape[0]
		self.layer0 = createLayer(&A0[0,0],&b0[0],outDim0,inDim0)
		if not self.layer0:
			raise MemoryError()
		self.threshold = threshold
		self.internalMap = allocateMap(self.layer0)
		if not self.internalMap:
			raise MemoryError()


	def add(self,np.ndarray[float,ndim=1,mode="c"] b not None, int pointIndex, int errorClass):
		if self.locArr:
			self.numLoc = 0
			free(self.locArr)
		addPointToMap(self.internalMap,<float *> b.data, pointIndex, errorClass, self.threshold)
		
	#Batch calculate, this is multithreaded and you can specify the number of threads you want to use.
	#It defaults to the number of virtual cores you have
	def batchAdd(self,np.ndarray[float,ndim=2,mode="c"] data not None, np.ndarray[int,ndim=1,mode="c"] indexes not None, 
		np.ndarray[int,ndim=1,mode="c"] errorClass not None, numProc=None):
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
		addDataToMapBatch(self.internalMap,<float *> data.data, <int *> indexes.data,<int *> errorClass.data, self.threshold,numData, numProc)

	def numLocations(self):
		if not self.locArr:
			self.numLoc = numLoc(self.internalMap)
			self.locArr = getLocations(self.internalMap,'i')
			if not self.locArr:
				raise MemoryError()
		return self.numLoc

	def location(self,int i):
		if not self.locArr:
			self.numLoc = numLoc(self.internalMap)
			self.locArr = getLocations(self.internalMap,'i')
			if not self.locArr:
				raise MemoryError()
		if(i > self.numLoc):
			eprint("Index out of bounds")
			raise MemoryError()
		thisLoc = _location.create( self.locArr[i] , self.layer0.outDim, self.layer0.inDim)
		return thisLoc
	
	def __dealloc__(self):
		freeMap(self.internalMap)
		if self.locArr:
			free(self.locArr)

	def adaptiveStep(self, np.ndarray[float,ndim=2,mode="c"] data not None, 
					np.ndarray[float,ndim=2,mode="c"] A1 not None, 
					np.ndarray[float,ndim=1,mode="c"] b1 not None):
		cdef unsigned int outDim1 = A1.shape[1]
		cdef unsigned int inDim1  = A1.shape[0]
		layer1 = createLayer(&A1[0,0],&b1[0],outDim1,inDim1)
		if not layer1:
			raise MemoryError()
		if not self.locArr:
			self.numLoc = numLoc(self.internalMap)
			self.locArr = getLocations(self.internalMap,'i')
			if not self.locArr:
				raise MemoryError()
		cdef maxPopGroupData * maxErrorGroup = refineMapAndGetMax(self.locArr, self.numLoc, layer1)
		cdef np.ndarray[np.float32_t,ndim=1] newHPVec = np.zeros([self.inDim], dtype=np.float32)
		cdef np.ndarray[np.float32_t,ndim=1] newHPoff = np.zeros([1], dtype=np.float32)

		createNewHP(maxErrorGroup,<float *>newHPVec.data,<float *>newHPoff.data)

		cdef int regCount = createTrainingData(self.locArr)