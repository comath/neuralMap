import cython
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

include "nnLayerUtilsWrap.pyx"

cdef extern from "../utils/key.h":
	cdef char compareKey(unsigned int *x, unsigned int *y, unsigned int keyLength)
	cdef void convertToKey(int * raw, unsigned int *key,unsigned int dataLen)
	cdef void convertFromKey(unsigned int *key, int * output, unsigned int dataLen)
	cdef unsigned int calcKeyLen(unsigned int dataLen)
	cdef void addIndexToKey(unsigned int * key, unsigned int index)
	cdef unsigned int checkIndex(unsigned int * key, unsigned int index)
	cdef void clearKey(unsigned int *key, unsigned int keyLength)
	
cdef extern from "../utils/ipCalculator.h":
	ctypedef struct ipCache:
		pass
	ipCache * allocateCache(nnLayer *layer0, float threshold)
	void freeCache(ipCache *cache)
	void getInterSig(float *p, unsigned int *ipSignature, ipCache * cache)

cdef class ipCalculator:
	cdef ipCache * cache
	cdef nnLayer * layer
	cdef unsigned int keyLen
	def __cinit__(self,np.ndarray[float,ndim=2,mode="c"] A not None, np.ndarray[float,ndim=1,mode="c"] b not None, float threshold):
		print "Called cinit for ipCalculator"
		cdef unsigned int outDim = A.shape[0]
		cdef unsigned int inDim  = A.shape[1]
		self.layer = createLayer(&A[0,0],&b[0],outDim,inDim)
		self.cache = allocateCache(self.layer,threshold)

		if not self.cache:
			raise MemoryError()


	def calculate(self,np.ndarray[float,ndim=1,mode="c"] b not None):
		cdef unsigned int dim
		dim = b.shape[0]
		keyLen = calcKeyLen(dim)
		cdef unsigned int *ipSignature_key = <unsigned int *>malloc(keyLen * sizeof(unsigned int))
		if not ipSignature_key:
			raise MemoryError()		
		cdef int *ipSignature = <int *>malloc(dim * sizeof(unsigned int))
		if not ipSignature:
			raise MemoryError()
		try:	        
			getInterSig(&b[0],ipSignature_key, self.cache)
			convertFromKey(ipSignature_key, ipSignature, dim)
			return [ ipSignature[i] for i in range(dim) ]
		finally:
			free(ipSignature)
			free(ipSignature_key)


	def __dealloc__(self):
		print "Calling dealloc for ipCalculator"
		freeCache(self.cache)
		freeLayer(self.layer)