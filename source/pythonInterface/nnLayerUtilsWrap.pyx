import cython
import numpy as np
cimport numpy as np

include "keyWrap.pyx"

cdef extern from "../cutils/nnLayerUtils.h":
	ctypedef struct nnLayer:
		float * A
		float * b
		unsigned int inDim
		unsigned int outDim
	nnLayer * createLayer(float *A, float *b, unsigned int outDim, unsigned int inDim)
	void freeLayer(nnLayer * layer)
	void evalLayer(nnLayer *layer, float * input, float * output)
	void getRegSig(nnLayer *layer, float *p, kint * regSig)
	void getRegSigBatch(nnLayer *layer, float *data, kint *regSig, unsigned numData, unsigned numProc)

cdef class neuralLayer:
	cdef nnLayer * _c_layer
	
	def __cinit__(self,np.ndarray[float,ndim=2,mode="c"] A not None, np.ndarray[float,ndim=1,mode="c"] b not None):
		cdef unsigned int inDim, outDim
		outDim,inDim = A.shape[1], A.shape[0]
		self._c_layer = <nnLayer *>createLayer( <float *> A.data,<float *>  b.data,outDim,inDim)

	def eval(self, np.ndarray[float,ndim=2,mode="c"] x not None):
		cdef np.ndarray output = np.zeros([self.layer.outDim], dtype=np.float32)
		cdef np.ndarray x_c = np.ascontiguousarray(x, dtype=np.float)
		evalLayer(self._c_layer , <float *> x_c.data, <float *> output.data)
		return output

	def __dealloc__(self):
		freeLayer(self._c_layer)

	def batchCalculateUncompressed(self,np.ndarray[float,ndim=2,mode="c"] data not None, numProc=None):
		if numProc == None:
			numProc = multiprocessing.cpu_count()
		if numProc > multiprocessing.cpu_count():
			eprint("WARNING: Specified too many cores. Reducing to the number you actually have.")
			numProc = multiprocessing.cpu_count()

		cdef unsigned int dim
		dim = data.shape[1]
		numData = data.shape[0]

		keyLen = calcKeyLen(dim)
		cdef np.ndarray[np.uint64_t,ndim=2] regSignature = np.zeros([numData,keyLen], dtype=np.uint64)
		getRegSigBatch(self._c_layer,<float *>data.data,<kint *>regSignature.data, numData, numProc)
		return regSignature