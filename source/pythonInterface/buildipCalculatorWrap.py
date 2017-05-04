#!/usr/bin/env python

from distutils.core import setup, Extension
from Cython.Build import cythonize

import numpy
import os

import sys


buildDir 	= "build/"
wrapDir 	= "source/pythonInterface/"
utilsDir 	= "source/cutils/"
pythonDir 	= "source/python/"
testDir 	= "source/test/"
mklRootDir 	= "/opt/intel/compilers_and_libraries_2017.2.174/linux/mkl/"

mklIncDir =  mklRootDir + "include/"
mklLibDir = mklRootDir+'lib/intel64'
intelLinLibDir = '/opt/intel/compilers_and_libraries/linux/lib/intel64_lin'
mklStaticLib = [
				mklLibDir + "/libmkl_intel_ilp64.a",  
			    mklLibDir + "/libmkl_sequential.a",  
			    mklLibDir + "/libmkl_core.a"
			    ]

ipCalculatorExt = Extension(name='ipCalculatorWrap',
						sources=[wrapDir+"ipCalculatorWrap.pyx",
								 utilsDir+"ipCalculator.c",
								 utilsDir+"parallelTree.c",
								 utilsDir+"key.c",
								 utilsDir+"nnLayerUtils.c"],
						include_dirs = [numpy.get_include(),utilsDir,mklIncDir],
						extra_objects = mklStaticLib,
						libraries=['pthread', 'm', 'dl'],
						extra_compile_args=[ "-O2", '-m64']
						,extra_link_args=[]
						,define_macros=[('MKL_ILP64',None)]
						)

setup(ext_modules = cythonize(ipCalculatorExt,gdb_debug=True))


os.rename("ipCalculatorWrap.so","nnMapper/ipCalculatorWrap.so")