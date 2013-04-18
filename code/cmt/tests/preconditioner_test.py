import sys
import unittest

sys.path.append('./code')

from cmt import AffinePreconditioner
from numpy import *
from numpy import max, round
from numpy.random import *
from numpy.linalg import inv, slogdet
from pickle import dump, load
from tempfile import mkstemp

class Tests(unittest.TestCase):
	def test_inverse(self):
		X = dot(randn(5, 5), randn(5, 1000)) + randn(5, 1)
		Y = dot(randn(2, 2), randn(2, 1000)) + dot(randn(2, 5), X)

		meanIn = randn(5, 1)
		meanOut = randn(2, 1)
		preIn = randn(5, 5)
		preOut = randn(2, 2)
		predictor = randn(2, 5)

		pre = AffinePreconditioner(
			meanIn,
			meanOut,
			preIn,
			preOut,
			predictor)

		# test inverse
		Xp, Yp = pre(X, Y)
		Xr, Yr = pre.inverse(Xp, Yp)

		self.assertLess(max(abs(Xr - X)), 1e-10)
		self.assertLess(max(abs(Yr - Y)), 1e-10)



	def test_pickle(self):
		meanIn = randn(5, 1)
		meanOut = randn(2, 1)
		preIn = randn(5, 5)
		preOut = randn(2, 2)
		predictor = randn(2, 5)

		pre0 = AffinePreconditioner(
			meanIn,
			meanOut,
			preIn,
			preOut,
			predictor)


		tmp_file = mkstemp()[1]

		# store transformation
		with open(tmp_file, 'w') as handle:
			dump({'pre': pre0}, handle)

		# load transformation
		with open(tmp_file) as handle:
			pre1 = load(handle)['pre']

		X, Y = randn(5, 100), randn(2, 100)

		X0, Y0 = pre0(X, Y)
		X1, Y1 = pre1(X, Y)

		# make sure linear transformation hasn't changed
		self.assertLess(max(abs(X0 - X1)), 1e-20)
		self.assertLess(max(abs(Y0 - Y1)), 1e-20)



	def test_logjacobian(self):
		meanIn = randn(5, 1)
		meanOut = randn(2, 1)
		preIn = randn(5, 5)
		preOut = randn(2, 2)
		predictor = randn(2, 5)

		pre = AffinePreconditioner(
			meanIn,
			meanOut,
			preIn,
			preOut,
			predictor)

		self.assertAlmostEqual(mean(pre.logjacobian(randn(5, 10), randn(2, 10))), slogdet(preOut)[1])



if __name__ == '__main__':
	unittest.main()
