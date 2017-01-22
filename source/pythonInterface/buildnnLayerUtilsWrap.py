#!/usrbin/env python

from distutils.core import setup, Extension
from Cython.Build import cythonize

import numpy

import sys

print 'Number of arguments:', len(sys.argv), 'arguments.'
print 'Argument List:', str(sys.argv)

BIN 	= "./build/"
wrapDir = "./source/pythonInterface/"
utilsDir 	= "./source/cutils/"
testDir 	= "./source/test/"
mklRootDir = "/opt/intel/compilers_and_libraries/linux/mkl"

mklIncDir =  mklRootDir + "/include/"
mklLibDir = [mklRootDir + "/lib/intel64/libmkl_intel_ilp64.a",  
			 mklRootDir + "/lib/intel64/libmkl_sequential.a",  
			 mklRootDir + "/lib/intel64/libmkl_core.a"]

nnLayerExt = Extension(name='nnLayerUtilsWrap',
						sources=[wrapDir+"nnLayerUtilsWrap.pyx",utilsDir+"nnLayerUtils.c"],
						include_dirs = [numpy.get_include(),utilsDir,mklIncDir],
						extra_objects = mklLibDir,
						extra_compile_args=['-DMKL', "-O2"]
						)

setup(ext_modules = cythonize(nnLayerExt))
	