#!/usr/bin/env python

from distutils.core import setup, Extension
from Cython.Build import cythonize

import numpy

BIN 	= "./build/"
wrapDir = "./source/pythonInterface/"
utilsDir 	= "./source/utils/"
TEST 	= "./source/test/"
mklRootDir = "/opt/intel/compilers_and_libraries/linux/mkl"

mklIncDir = mklRootDir + "/include/"
mklLibDir = [mklRootDir + "/lib/intel64/libmkl_intel_ilp64.a",  mklRootDir + "/lib/intel64/libmkl_sequential.a",  mklRootDir + "/lib/intel64/libmkl_core.a"]

nnLayerExt = Extension(name='nnLayerWrap',
						sources=[wrapDir+"nnLayerUtilsWrap.pyx"],
						include_dirs = [numpy.get_include(),utilsDir,mklIncDir],
						library_dirs = mklLibDir,
						extra_compile_args=['-DMKL', "-O2"])

setup(ext_modules = cythonize(nnLayerExt))
	