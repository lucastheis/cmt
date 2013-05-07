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
		'/opt/intel/mkl/lib',
		'/opt/intel/lib']
	libraries = [
		'mkl_intel_lp64',
		'mkl_intel_thread',
		'mkl_core',
		'mkl_def',
		'iomp5']
	extra_compile_args = [
		'-DEIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS',
		'-DEIGEN_USE_MKL_ALL',
		'-Wno-deprecated',
		'-wd1224',
		'-openmp']
	extra_link_args = []

	for path in ['/opt/intel/mkl/lib/intel64', '/opt/intel/lib/intel64']:
		if os.path.exists(path):
			library_dirs += [path]
else:
	# gcc-specific options
	include_dirs = []
	library_dirs = []
	libraries = []
	extra_compile_args = []
	extra_link_args = []

	if sys.platform != 'darwin':
		libraries += [
			'gomp']
		extra_compile_args += [
			'-Wno-cpp',
			'-fopenmp']

if sys.platform != 'darwin':
	extra_compile_args += [
		'-std=c++0x']

modules = [
	Extension('cmt',
		language='c++',
		sources=[
			'code/cmt/src/affinepreconditioner.cpp',
			'code/cmt/src/affinetransform.cpp',
			'code/cmt/src/callbackinterface.cpp',
			'code/cmt/src/conditionaldistribution.cpp',
			'code/cmt/src/conditionaldistributioninterface.cpp',
			'code/cmt/src/distribution.cpp',
			'code/cmt/src/distributioninterface.cpp',
			'code/cmt/src/fvbninterface.cpp',
			'code/cmt/src/mcgsm.cpp',
			'code/cmt/src/mcgsminterface.cpp',
			'code/cmt/src/mcbm.cpp',
			'code/cmt/src/mcbminterface.cpp',
			'code/cmt/src/module.cpp',
			'code/cmt/src/patchmodel.cpp',
			'code/cmt/src/patchmodelinterface.cpp',
			'code/cmt/src/pcapreconditioner.cpp',
			'code/cmt/src/pcatransform.cpp',
			'code/cmt/src/preconditioner.cpp',
			'code/cmt/src/preconditionerinterface.cpp',
			'code/cmt/src/pyutils.cpp',
			'code/cmt/src/tools.cpp',
			'code/cmt/src/toolsinterface.cpp',
			'code/cmt/src/trainableinterface.cpp',
			'code/cmt/src/utils.cpp',
			'code/cmt/src/whiteningpreconditioner.cpp',
			'code/cmt/src/whiteningtransform.cpp'],
		include_dirs=[
			'code',
			'code/cmt/include',
			'code/liblbfgs/include',
			os.path.join(numpy.__path__[0], 'core/include/numpy')] + include_dirs,
		library_dirs=[] + library_dirs,
		libraries=[] + libraries,
		extra_link_args=[
			'-fPIC',
			'code/liblbfgs/lib/.libs/liblbfgs.a'] + extra_link_args,
		extra_compile_args=[
			'-Wno-sign-compare',
			'-Wno-parentheses',
			'-Wno-write-strings'] + extra_compile_args)]

# enable parallel compilation
CCompiler.compile = parallelCCompiler

setup(
	name='cmt',
	version='0.2.0',
	author='Lucas Theis',
	author_email='lucas@theis.io',
	description='C++ implementations of conditional probabilistic models.',
	url='http://github.com/lucastheis/cmt',
	license='MIT',
	ext_modules=modules)
