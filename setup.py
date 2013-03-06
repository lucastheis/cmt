#!/usr/bin/env python

import os
import sys
import numpy

sys.path.append('./code')

from distutils.core import setup, Extension
from distutils.ccompiler import CCompiler, get_default_compiler
from utils import parallelCCompiler
from numpy.distutils.intelccompiler import IntelCCompiler
from numpy import any

# heuristic for figuring out which compiler is being used (icc, gcc)
if any(['intel' in arg for arg in sys.argv]) or 'intel' in get_default_compiler():
	# icc-specific options
	include_dirs=[
		'/opt/intel/mkl/include']
	library_dirs=[
		'/opt/intel/mkl/lib/intel64',
		'/opt/intel/composer_xe_2013.1.117/compiler/lib/intel64']
	libraries = [
		'mkl_intel_lp64',
		'mkl_intel_thread',
		'mkl_core',
		'mkl_def',
		'iomp5']
	extra_compile_args = [
		'-DEIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS',
		'-DEIGEN_USE_MKL_ALL',
		'-wd1224']
else:
	# gcc-specific options
	include_dirs = []
	library_dirs = []
	libraries = [
		'gomp']
	extra_compile_args = [
		'-Wno-cpp']

if sys.platform != 'darwin':
	# c++0x is used for random number generation on linux
	extra_compile_args += ['-std=c++0x']

modules = [
	Extension('mcm',
		language='c++',
		sources=[
			'code/mcm/src/mcgsm.cpp',
			'code/mcm/src/mcgsminterface.cpp',
			'code/mcm/src/affinetransform.cpp',
			'code/mcm/src/lineartransform.cpp',
			'code/mcm/src/identitytransform.cpp',
			'code/mcm/src/whiteningtransform.cpp',
			'code/mcm/src/pcatransform.cpp',
			'code/mcm/src/transforminterface.cpp',
			'code/mcm/src/whiteningpreconditioner.cpp',
			'code/mcm/src/preconditionerinterface.cpp',
			'code/mcm/src/tools.cpp',
			'code/mcm/src/toolsinterface.cpp',
			'code/mcm/src/pyutils.cpp',
			'code/mcm/src/utils.cpp',
			'code/mcm/src/module.cpp',
			'code/mcm/src/callbacktrain.cpp',
			'code/mcm/src/conditionaldistribution.cpp'],
		include_dirs=[
			'code',
			'code/mcm/include',
			'code/liblbfgs/include',
			os.path.join(numpy.__path__[0], 'core/include/numpy')] + include_dirs,
		library_dirs=[] + library_dirs,
		libraries=[] + libraries,
		extra_link_args=[
			'-fPIC',
			'code/liblbfgs/lib/.libs/liblbfgs.a'],
		extra_compile_args=[
			'-fopenmp',
			'-Wno-sign-compare',
			'-Wno-parentheses',
			'-Wno-write-strings'] + extra_compile_args)]

# enable parallel compilation
CCompiler.compile = parallelCCompiler

setup(
	name='mcm',
	version='0.0.1',
	author='Lucas Theis',
	author_email='lucas@theis.io',
	description='A C++ implementation of conditional models such as the MCGSM.',
	url='http://github.com/lucastheis/mcm',
	license='MIT',
	ext_modules=modules)
