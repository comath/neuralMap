import cython
import numpy as np
cimport numpy as np

cdef extern void c_printMatrix(double *arr, int m, int n)
cdef extern float c_dot(float *v, float *u, int n)

