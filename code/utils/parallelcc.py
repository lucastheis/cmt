from multiprocessing import pool, cpu_count
from random import random
from time import sleep

def parallelCCompiler(self, sources, output_dir=None, macros=None, include_dirs=None, debug=0,
	extra_preargs=None, extra_postargs=None, depends=None):
	"""
	Monkey-patch for parallel compilation with distutils.
	"""

	# those lines are copied from distutils.ccompiler.CCompiler directly
	macros, objects, extra_postargs, pp_opts, build = self._setup_compile(output_dir, macros, include_dirs, sources, depends, extra_postargs)
	cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)

	def _single_compile(obj):
		# wait a random amount of time so that compiler outputs don't interfere
		sleep(random() / 20. * cpu_count())
		try:
			src, ext = build[obj]
		except KeyError:
			return
		self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)

	# convert to list, imap is evaluated on-demand
	list(pool.ThreadPool(cpu_count()).imap(_single_compile, objects))

	return objects
