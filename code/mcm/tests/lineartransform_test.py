import sys
import unittest

sys.path.append('./code')
sys.path.append('./build/lib.macosx-10.6-intel-2.7')
sys.path.append('./build/lib.linux-x86_64-2.7')

from mcm import LinearTransform
from numpy import *
from numpy.random import *
from numpy import max
from pickle import dump, load
from tempfile import mkstemp

class Tests(unittest.TestCase):
	def test_transform(self):
		A = randn(5, 5)

		lt = LinearTransform(A)

		# basic sanity checks
		self.assertLess(max(abs(lt.A - A)), 1e-20)
		self.assertLess(max(abs(lt.inverse(lt.A) - eye(5))), 1e-10)

		lt.A = randn(5, 5)

		# inverse should be recalculated after changing matrix
		self.assertLess(max(abs(lt.inverse(lt.A) - eye(5))), 1e-10)

		X = randn(5, 20)

		lt = LinearTransform(randn(10, 5))

		# test inverse
		self.assertLess(max(abs(lt.inverse(lt(X)) - X)), 1e-10)



	def test_pickle(self):
		lt0 = LinearTransform(randn(5, 10))

		tmp_file = mkstemp()[1]

		# store transformation
		with open(tmp_file, 'w') as handle:
			dump({'lt': lt0}, handle)

		# load transformation
		with open(tmp_file) as handle:
			lt1 = load(handle)['lt']

		# make sure linear transformation hasn't changed
		self.assertLess(max(abs(lt0.A - lt1.A)), 1e-20)



if __name__ == '__main__':
	unittest.main()
