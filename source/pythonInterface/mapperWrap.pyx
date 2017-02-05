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
	cdef void freeMap(_nnMap *map)

	cdef void addDatumToMap(_nnMap * map, float *datum, float errorMargin)
	cdef void addDataToMapBatch(_nnMap * map, float *data, float * errorMargins, unsigned int numData, unsigned int numProc)

	cdef location getMaxErrorLoc(_nnMap * map);
	cdef location * getLocationArray(_nnMap * map);
	cdef location getLocationByKey(_nnMap *map, unsigned int * ipSig, unsigned int * regSig);

cdef class nnMap:
	cdef _nnMap * map
	cdef nnLayer * layer0
	cdef nnLayer * layer1
	cdef unsigned int keyLen

	def __cinit__(self,np.ndarray[float,ndim=2,mode="c"] A0 not None, np.ndarray[float,ndim=1,mode="c"] b0 not None,
					   np.ndarray[float,ndim=2,mode="c"] A1 not None, np.ndarray[float,ndim=1,mode="c"] b1 not None,
					   float threshold, float errorThreshhold):
		cdef unsigned int outDim0 = A0.shape[0]
		cdef unsigned int inDim0  = A0.shape[1]
		cdef unsigned int outDim1 = A1.shape[0]
		cdef unsigned int inDim1  = A1.shape[1]
		self.layer0 = createLayer(&A0[0,0],&b0[0],outDim0,inDim0)
		if not self.layer0:
			raise MemoryError()
		self.layer1 = createLayer(&A1[0,0],&b1[0],outDim1,inDim1)
		if not self.layer1:
			raise MemoryError()
		self.map = allocateMap(self.layer0,self.layer1,threshold,errorThreshhold)
		if not self.map:
			raise MemoryError()


	def add(self,np.ndarray[float,ndim=1,mode="c"] b not None, float errorMargin):       
		addDatumToMap(self.map,<float *> b.data, errorMargin)
		

	#Batch calculate, this is multithreaded and you can specify the number of threads you want to use.
	#It defaults, and takes a maximum of 
	def batchAdd(self,np.ndarray[float,ndim=2,mode="c"] data not None, np.ndarray[float,ndim=1,mode="c"] errorMargins not None, numProc=None):
		if numProc == None:
			numProc = multiprocessing.cpu_count()
		if numProc > multiprocessing.cpu_count():
			eprint("WARNING: Specified too many cores. Reducing to the number you actually have.")
			numProc = multiprocessing.cpu_count()

		numData = data.shape[0]       
		addDataToMapBatch(self.map,<float *> data.data, <float *> errorMargins.data, numData, numProc)


	def __dealloc__(self):
		freeMap(self.map)
		freeLayer(self.layer0)
		freeLayer(self.layer1)