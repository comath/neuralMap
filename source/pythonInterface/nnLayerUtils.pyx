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

class :
	def __cinit__(np.ndarray[float,ndim=2,mode="c"] A not None, np.ndarray[float,ndim=1,mode="c"] b not None):
		cdef unsigned int inDim, outDim
		outDim,inDim = A.shape[0], A.shape[1]
		this.layer = createLayer(&A[0,0],&b[0],outDim,inDim)

	def __dealloc__(self):
        freelayer(self.cache) 