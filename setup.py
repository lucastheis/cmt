#!/usr/bin/env python

import os
import sys
import numpy
from distutils.core import setup, Extension

modules = [
	Extension('mcm',
		language='c++',
		sources=[
			'code/mcm/src/mcgsminterface.cpp',
			'code/mcm/src/mcgsm.cpp',
			'code/mcm/src/pyutils.cpp',
			'code/mcm/src/utils.cpp',
			'code/mcm/src/module.cpp',
			'code/mcm/src/conditionaldistribution.cpp'],
		include_dirs=[
			'code',
			'code/mcm/include',
			'code/liblbfgs/include',
			os.path.join(numpy.__path__[0], 'core/include/numpy')],
		library_dirs=[],
		libraries=[
			'gomp'],
		extra_link_args=[
			'code/liblbfgs/lib/.libs/liblbfgs.a'],
		extra_compile_args=[
			'-fopenmp',
			'-Wno-parentheses',
			'-Wno-write-strings'] + ['-std=c++0x'] if sys.platform != 'darwin' else [])]

setup(
	name='mcm',
	version='0.0.1',
	author='Lucas Theis',
	author_email='lucas@theis.io',
	description='A C++ implementation of conditional models such as the MCGSM.',
	url='http://github.com/lucastheis/mcm',
	license='MIT',
	ext_modules=modules)
