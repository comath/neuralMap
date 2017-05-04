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
mklRootDir 	= "/opt/intel/compilers_and_libraries/linux/mkl"

mklIncDir =  mklRootDir + "/include/"
mklLibDir = mklRootDir+'/lib/intel64'
intelLinLibDir = '/opt/intel/compilers_and_libraries/linux/lib/intel64_lin'
mklStaticLib = [mklLibDir + "/libmkl_intel_ilp64.a",  
			    mklLibDir + "/libmkl_sequential.a",  
			    mklLibDir + "/libmkl_core.a"]

traceExt = Extension(name='ipTrace',
						sources=[wrapDir+"ipTraceWrap.pyx",
								 utilsDir+"ipTrace.c",
								 utilsDir+"key.c",
								 utilsDir+"nnLayerUtils.c"],
						include_dirs = [numpy.get_include(),utilsDir,mklIncDir],
						#extra_objects = mklStaticLib ,
						library_dirs=[mklLibDir,intelLinLibDir],
						libraries=['mkl_rt',
									'pthread', 'm', 'dl'],
						extra_compile_args=[ '-DMKL_ILP64','-DMKL', "-O2", '-m64'],
						extra_link_args=['-Wl,--no-as-needed']
						#,define_macros=[('DEBUG',None)]
						)

setup(ext_modules = cythonize(traceExt,gdb_debug=True))


os.rename("ipTrace.so","nnMapper/ipTrace.so")