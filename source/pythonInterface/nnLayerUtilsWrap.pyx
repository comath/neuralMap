import cython
import numpy as np
cimport numpy as np

cdef extern from "../utils/nnLayerUtils.h":
	ctypedef struct nnLayer:
		float * A
		float * b
		unsigned int inDim
		unsigned int outDim
	nnLayer * createLayer(float *A, float *b, unsigned int outDim, unsigned int inDim)
	void freeLayer(nnLayer * layer)
	void evalLayer(nnLayer *layer, float * input, float * output)

cdef class neuralLayer:
	cdef nnLayer * _layer
	
	def __cinit__(self,np.ndarray[float,ndim=2,mode="c"] A not None, np.ndarray[float,ndim=1,mode="c"] b not None):
		cdef unsigned int inDim, outDim
		outDim,inDim = A.shape[0], A.shape[1]
		self._layer = <nnLayer *>createLayer(&A[0,0],&b[0],outDim,inDim)

	def eval(self,np.ndarray[float,ndim=2,mode="c"] x not None):
		cdef np.nparray output = np.zeros([self.layer.outDim], dtype=DTYPE)
		cdef np.n

		x_c = np.ascontiguousarray(c, dtype=np.float)
		output_c = np.ascontiguousarray(output, dtype=np.float)
		evalLayer(self._layer,&x_c[0],&output_c[0])
		return output

	def __dealloc__(self):
		freeLayer(self._layer) 