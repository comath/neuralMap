from __future__ import print_function
import cython
import numpy as np
cimport numpy as np
import multiprocessing
from libc.stdlib cimport malloc, free

#to print to stderr to warn the user of usage errors
include "ipTrace.pyx"
import sys

def eprint(*args, **kwargs):
	print(*args, file=sys.stderr, **kwargs)



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
	cdef void nodeGetIPKey(mapTreeNode * node, int * ipKey, unsigned int outDim, char compressed)
	cdef void nodeGetRegKey(mapTreeNode * node, int * regKey, unsigned int outDim, char compressed)
	cdef void nodeGetPointIndexes(mapTreeNode * node, int errorClass, int *indexHolder)
	cdef int nodeGetTotal(mapTreeNode * node, int errorClass)

cdef extern from "../cutils/mapper.h":
	ctypedef struct _nnMap:
		int createdKL
		kint *ipKey
		kint *regKey

		pthread_mutex_t datamutex
		location loc

	cdef _nnMap * allocateMap(nnLayer *layer0) with gil
	cdef void freeMap(_nnMap * internalMap) 

	cdef void addPointToMap(_nnMap * map, float *point, int pointIndex, int errorClass, float threshold) with gil 
	cdef void addPointsToMapBatch(_nnMap * map, float *data, int *indexes, int * errorClasses, float threshold, unsigned int numData, unsigned int numProc) with gil

	cdef location getPointsAt(_nnMap *map, kint *keyPair)
	cdef unsigned int numLoc(_nnMap * map)
	cdef mapTreeNode ** getLocations(_nnMap *map, char orderBy) with gil
	void location_get_indexes(location *v, int *indexHolder)


cdef extern from "../cutils/adaptiveTools.h":
	ctypedef struct maxErrorCorner:
		mapTreeNode ** nodes
		kint * hpCrossed
		int count
		int selectionIndex

	cdef maxErrorCorner * refineMapAndGetMax(mapTreeNode ** locArr, int maxLocIndex, nnLayer * selectionLayer)
	
	cdef void getAverageError(maxErrorCorner * maxErrorGroup, float *data, float * avgError, int dim) with gil
	cdef void createNewHPVec(maxErrorCorner * maxErrorGroup, float * avgError, float *solution, nnLayer *hpLayer, float *newVec, float *newOff) with gil

	cdef vector * getRegSigs(mapTreeNode ** locArr, int numNodes) with gil
	#cdef void unpackRegSigs(vector * regSigs, unsigned int dim, float * unpackedSigs)
	cdef void createData(maxErrorCorner *maxErrorGroup, nnLayer *selectionLayer, int selectionIndex, vector *regSigs, float *unpackedSigs, int * labels) with gil


	cdef float *getSolutionPointer(_nnMap *map) with gil
	cdef int getSelectionIndex(maxErrorCorner * maxGroup) with gil


cdef extern from "../cutils/selectionTrainer.h":
	cdef void trainNewSelector(nnLayer *selectionLayer, mapTreeNode **locArr, int maxLocIndex, maxErrorCorner *maxGroup, float * newSelectionWeight, float * newSelectionBias) with gil



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

	def ipSig(self,compressed=True):
		cdef np.ndarray[np.uint32_t,ndim=1] ipSignature = np.zeros([pyCalcKeyLen(self.outDim)], dtype=np.uint32)
		nodeGetIPKey(self.thisLoc, <int *> ipSignature.data, self.outDim, 1)
		return ipSignature

	def regSig(self,compressed=True):
		cdef np.ndarray[np.uint32_t,ndim=1] regSignature = np.zeros([pyCalcKeyLen(self.outDim)], dtype=np.uint32)
		nodeGetRegKey(self.thisLoc, <int *> regSignature.data, self.outDim, 1)		
		return regSignature
		
	def pointIndexes(self, int errorClass):
		numPoints =	nodeGetTotal(self.thisLoc, errorClass)
		cdef np.ndarray[np.int32_t,ndim=1] pointI = np.zeros([numPoints], dtype=np.int32)
		nodeGetPointIndexes(self.thisLoc,errorClass, <int *>pointI.data)
		return pointI
	
	def all(self):
		return self.ipSig(), self.regSig(), self.pointIndexes(0), self.pointIndexes(1)

cdef class cy_nnMap:
	cdef _nnMap * internalMap
	cdef nnLayer * layer0
	cdef nnLayer * layer1
	cdef long int outDim0
	cdef long int inDim0
	cdef unsigned int keyLen
	cdef unsigned int numLoc
	cdef float threshold
	cdef mapTreeNode ** locArr

	def __cinit__(self,np.ndarray[float,ndim=2,mode="c"] A0 not None, np.ndarray[float,ndim=1,mode="c"] b0 not None,
					   float threshold):
		self.outDim0 = A0.shape[0]
		self.inDim0  = A0.shape[1]
		print("Creating map structure, outdim %(n)d, inDim %(m)d"%{'n':self.outDim0,'m':self.inDim0})
		self.layer0 = createLayer(<float *> A0.data,<float *> b0.data,self.inDim0,self.outDim0)
		if not self.layer0:
			raise MemoryError()
		self.threshold = threshold
		self.internalMap = allocateMap(self.layer0)
		if not self.internalMap:
			raise MemoryError()

	def __dealloc__(self):
		freeMap(self.internalMap)
		if self.locArr:
			free(self.locArr)

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
		addPointsToMapBatch(self.internalMap,<float *> data.data, <int *> indexes.data,<int *> errorClass.data, self.threshold,numData, numProc)

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

	def adaptiveStep(self, np.ndarray[float,ndim=2,mode="c"] data not None, 
					np.ndarray[float,ndim=2,mode="c"] A1 not None, 
					np.ndarray[float,ndim=1,mode="c"] b1 not None):
		cdef long int outDim1 = A1.shape[0]
		cdef long int inDim1  = A1.shape[1]
		if(not inDim1 == self.outDim0):
			ValueError("the selection matrix input dim is different from the hyperplane matrix output dim")
		
		layer1 = createLayer(&A1[0,0],&b1[0],inDim1,outDim1)
		if not layer1:
			raise MemoryError()
		if not self.locArr:
			self.numLoc = numLoc(self.internalMap)
			self.locArr = getLocations(self.internalMap,'i')
			if not self.locArr:
				raise MemoryError()
		print("Starting the adaptive step.")
		print("Searching for the corner with the most error")
		cdef maxErrorCorner * maxErrorGroup = refineMapAndGetMax(self.locArr, self.numLoc, layer1)

		cdef np.ndarray[np.float32_t,ndim=1] avgError = np.zeros([self.inDim0], dtype=np.float32)
		cdef float * solution = getSolutionPointer(self.internalMap)
		cdef np.ndarray[np.float32_t,ndim=2] newHPVec = np.zeros([1,self.inDim0], dtype=np.float32)
		cdef np.ndarray[np.float32_t,ndim=1] newHPoff = np.zeros([1], dtype=np.float32)
		newDim = inDim1 + 1
		cdef np.ndarray[np.float32_t,ndim=2] newSelectionWeight = np.zeros([outDim1,newDim], dtype=np.float32)
		cdef np.ndarray[np.float32_t,ndim=1] newSelectionBias = np.zeros([outDim1], dtype=np.float32)
		
		if(getSelectionIndex(maxErrorGroup) >= 0):
			print("Aquiring average error in the max error corner with %(baseInDim)d"%{'baseInDim':self.inDim0})
			getAverageError(maxErrorGroup, <float *> data.data, <float *> avgError.data, self.inDim0)

			print("Creating new hyperplane.")
			createNewHPVec(maxErrorGroup, <float *>avgError.data, solution, self.layer0, <float *>newHPVec.data,<float *> newHPoff.data);

			print("Creating new selection Layer.")
			trainNewSelector(layer1, self.locArr, self.numLoc, maxErrorGroup, <float *>newSelectionWeight.data, <float *>newSelectionBias.data);
			
			print("Returning everything")
		
			return newDim, newHPVec, newHPoff, newSelectionWeight, newSelectionBias
		else:
			return inDim1, newHPVec, newHPoff, newSelectionWeight, newSelectionBias
		

