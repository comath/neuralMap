#!/usr/bin/env python
from setuptools import setup
from distutils.core import Extension
from Cython.Build import cythonize

import numpy
import os

import sys


buildDir 	= "build/"
wrapDir = "source/pythonInterface/"
utilsDir 	= "source/cutils/"
testDir 	= "source/test/"
useMKL = False
useBLAS = True

if(useMKL):
	mklRootDir = "/opt/intel/compilers_and_libraries/linux/mkl"
	mklIncDir =  mklRootDir + "/include/"
	mklLibDir = mklRootDir+'/lib/intel64'
	intelLinLibDir = '/opt/intel/compilers_and_libraries/linux/lib/intel64_lin'
	blasLibDir = [mklLibDir,intelLinLibDir]
	mklStaticLib = [mklLibDir + "/libmkl_intel_ilp64.a",  
				    mklLibDir + "/libmkl_sequential.a",  
				    mklLibDir + "/libmkl_core.a"]
	libs=['mkl_rt','pthread', 'm', 'dl',]
	inc_dirs = [numpy.get_include(),utilsDir,mklIncDir]
	extra_compile_args=[ '-DMKL_ILP64','-DUSE_MKL', "-O2", '-m64']
	extra_link_args=['-Wl,--no-as-needed' ]


if(useBLAS):
	blasLibDir = []
	inc_dirs = [numpy.get_include(),utilsDir]
	libs=['pthread', 'm', 'dl','openblas','lapacke']
	extra_compile_args=['-DUSE_OPENBLAS', "-O2", '-m64']
	extra_link_args=[]

mapperExt = Extension(name='mapperWrap',
						sources=[wrapDir+"mapperWrap.pyx",
								 utilsDir+"mapper.c",
								 utilsDir+"ipTrace.c",
								 utilsDir+"mapperTree.c",
								 utilsDir+"key.c",
								 utilsDir+"vector.c",
								 utilsDir+"location.c",
								 utilsDir+"adaptiveTools.c",
								 utilsDir+"selectionTrainer.c",
								 utilsDir+"nnLayerUtils.c"],
						include_dirs = inc_dirs,
						#extra_objects = mklStaticLib ,
						library_dirs= blasLibDir,
						libraries=libs,
						extra_compile_args=extra_compile_args,
						extra_link_args=extra_link_args
						#,define_macros=[('DEBUG',None)]
						)

setup(name='nnMap',
      version='0.1',
      description='',
      author='Sven Cattell',
      author_email='scattell@gmail.com',
      url='comathematician.net',
      package_dir = {'': 'source/python'},
      packages=['nnMap'],
	ext_modules = cythonize(mapperExt,gdb_debug=False))


