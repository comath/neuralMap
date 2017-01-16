import cython
import numpy as np
cimport numpy as np

cdef extern from "../utils/nnLayerUtils.h":
	ctypedef struct nnLayer:
		float * A
		float * b
		unsigned int inDim
		unsigned int outdim
	nnLayer * createLayer(float *A, float *b, uint outDim, uint inDim)
	void freeLayer(nnLayer * layer)

cdef extern from "../utils/ipCalculator.h":
	ctypedef struct ipCache:
		nnLayer *layer0;
		Tree *bases;
		float *hpOffsetVecs;
		float *hpNormals;
		float threshold;
	ipCache * allocateCache(nnLayer *layer0, float threshold)
	void freeCache(ipCache *cache)
	void getInterSig(float *p, unsigned int *ipSignature, ipCache * cache)

def calcKeyLen(dataLen):
	keyLen = (dataLen/32)
	if(dataLen % 32){
		keyLen++
	}
	return keyLen

class intersectionPoset:
	def __cinit__(np.ndarray[float,ndim=2,mode="c"] A not None, np.ndarray[float,ndim=1,mode="c"] b not None, float threshold):
		cdef unsigned int inDim, outDim
		outDim,inDim = A.shape[0], A.shape[1]
		layer = createLayer(&A[0,0],&b[0],outDim,inDim)
		ipCache * self.cache = allocateCache(layer,threshold)

	def calculate(np.ndarray[float,ndim=1,mode="c"] b not None):
		cdef unsigned int dim
		dim = b.shape[0]
		unsigned int keyLen = calcKeyLen(dim)
		cdef unsigned int *ipSignature = <unsigned int *>malloc(keyLen * sizeof(unsigned int))
		if not ipSignature:
            raise MemoryError()

		try:	        
			getInterSig(&b[0],ipSignature, self.cache)
	        return [ ipSignature[i] for i in range(number) ]
	    finally:
	        # return the previously allocated memory to the system
	        free(ipSignature)

	def __dealloc__(self):
        freeCache(self.cache) 