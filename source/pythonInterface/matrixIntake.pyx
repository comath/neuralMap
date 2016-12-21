import cython
import numpy as np
cimport numpy as np

cdef extern void c_printMatrix(double *arr, int m, int n)
cdef extern float c_dot(float *v, float *u, int n)

@cython.boundscheck(False)
@cython.wraparound(False)

def printMatrix(np.ndarray[double,ndim=2,mode="c"] A not None):
	cdef int n, m
	m,n = A.shape[0], A.shape[1]
	c_printMatrix(&A[0,0],m,n)
	return None

def customDot(np.ndarray[float,ndim=1,mode="c"] v not None,np.ndarray[float,ndim=1,mode="c"] u not None):
	cdef int n
	n = v.shape[0]
	assert(n == u.shape[0])
	return c_dot(&v[0],&u[0],n)