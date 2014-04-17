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

INTEL_PATH = '/opt/intel/'

# heuristic for figuring out which compiler is being used (icc, gcc)
if any(['intel' in arg for arg in sys.argv]) or 'intel' in get_default_compiler():
	# icc-specific options
	include_dirs=[
		os.path.join(INTEL_PATH, 'mkl/include')]
	library_dirs=[
		os.path.join(INTEL_PATH, 'mkl/lib'),
		os.path.join(INTEL_PATH, 'lib')]
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
		'-openmp',
		'-std=c++0x']
	extra_link_args = []

	for path in [os.path.join(INTEL_PATH, 'mkl/lib/intel64'), os.path.join(INTEL_PATH, 'lib/intel64')]:
		if os.path.exists(path):
			library_dirs += [path]

elif sys.platform == 'darwin':
	# clang-specific options
	include_dirs = []
	library_dirs = []
	libraries = []
	extra_compile_args = ['-std=c++0x', '-stdlib=libc++']
	extra_link_args = []

	os.environ['CC'] = 'clang++'
	os.environ['CXX'] = 'clang++'
	os.environ['MACOSX_DEPLOYMENT_TARGET'] = '10.7'

else:
	# gcc-specific options
	include_dirs = []
	library_dirs = []
	libraries = ['gomp']
	extra_compile_args = ['-std=c++0x', '-Wno-cpp', '-fopenmp']
	extra_link_args = []


modules = [
	Extension('_cmt',
		language='c++',
		sources=[
			'code/cmt/python/src/callbackinterface.cpp',
			'code/cmt/python/src/conditionaldistributioninterface.cpp',
			'code/cmt/python/src/distributioninterface.cpp',
			'code/cmt/python/src/fvbninterface.cpp',
			'code/cmt/python/src/glminterface.cpp',
			'code/cmt/python/src/gsminterface.cpp',
			'code/cmt/python/src/mcbminterface.cpp',
			'code/cmt/python/src/mcgsminterface.cpp',
			'code/cmt/python/src/module.cpp',
			'code/cmt/python/src/mixtureinterface.cpp',
			'code/cmt/python/src/mlrinterface.cpp',
			'code/cmt/python/src/nonlinearitiesinterface.cpp',
			'code/cmt/python/src/patchmodelinterface.cpp',
			'code/cmt/python/src/preconditionerinterface.cpp',
			'code/cmt/python/src/pyutils.cpp',
			'code/cmt/python/src/stminterface.cpp',
			'code/cmt/python/src/toolsinterface.cpp',
			'code/cmt/python/src/trainableinterface.cpp',
			'code/cmt/python/src/univariatedistributionsinterface.cpp',
			'code/cmt/src/affinepreconditioner.cpp',
			'code/cmt/src/affinetransform.cpp',
			'code/cmt/src/binningtransform.cpp',
			'code/cmt/src/conditionaldistribution.cpp',
			'code/cmt/src/distribution.cpp',
			'code/cmt/src/glm.cpp',
			'code/cmt/src/gsm.cpp',
			'code/cmt/src/mcbm.cpp',
			'code/cmt/src/mcgsm.cpp',
			'code/cmt/src/mixture.cpp',
			'code/cmt/src/mlr.cpp',
			'code/cmt/src/nonlinearities.cpp',
			'code/cmt/src/patchmodel.cpp',
			'code/cmt/src/pcapreconditioner.cpp',
			'code/cmt/src/pcatransform.cpp',
			'code/cmt/src/preconditioner.cpp',
			'code/cmt/src/regularizer.cpp',
			'code/cmt/src/stm.cpp',
			'code/cmt/src/tools.cpp',
			'code/cmt/src/trainable.cpp',
			'code/cmt/src/utils.cpp',
			'code/cmt/src/univariatedistributions.cpp',
			'code/cmt/src/whiteningpreconditioner.cpp',
			'code/cmt/src/whiteningtransform.cpp'],
		include_dirs=[
			'code',
			'code/cmt/include',
			'code/cmt/python/include',
			'code/liblbfgs/include',
			os.path.join(numpy.__path__[0], 'core/include/numpy')] + include_dirs,
		library_dirs=[] + library_dirs,
		libraries=[] + libraries,
		extra_link_args=[
			'-fPIC',
			'code/liblbfgs/lib/.libs/liblbfgs.a'] + extra_link_args,
		extra_compile_args=[
			'-DEIGEN_NO_DEBUG',
			'-Wno-sign-compare',
			'-Wno-parentheses',
			'-Wno-write-strings'] + extra_compile_args)]

# enable parallel compilation
CCompiler.compile = parallelCCompiler

setup(
	name='cmt',
	version='0.5.0',
	author='Lucas Theis',
	author_email='lucas@theis.io',
	description='Fast implementations of different probabilistic models.',
	url='http://github.com/lucastheis/cmt',
	license='MIT',
	ext_modules=modules,
	package_dir={'cmt': 'code/cmt/python'},
	packages=[
		'cmt',
		'cmt.models',
		'cmt.transforms',
		'cmt.tools',
		'cmt.utils',
		'cmt.nonlinear'])
