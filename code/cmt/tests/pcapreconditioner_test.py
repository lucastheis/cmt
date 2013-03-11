import sys
import unittest

sys.path.append('./code')

from cmt import PCAPreconditioner
from numpy import *
from numpy import max, round
from numpy.random import *
from numpy.linalg import inv, slogdet
from pickle import dump, load
from tempfile import mkstemp

class Tests(unittest.TestCase):
	def test_transform(self):
		X = dot(randn(5, 5), randn(5, 1000)) + randn(5, 1)
		Y = dot(randn(2, 2), randn(2, 1000)) + dot(randn(2, 5), X)

		wt = PCAPreconditioner(X, Y, num_pcs=X.shape[0])

		# joint covariance
		C = cov(vstack(wt(X, Y)), bias=True)

		self.assertLess(max(abs(C - eye(7))), 1e-8)

		# test inverse
		Xw, Yw = wt(X, Y)
		Xr, Yr = wt.inverse(Xw, Yw)

		self.assertLess(max(abs(Xr - X)), 1e-10)
		self.assertLess(max(abs(Yr - Y)), 1e-10)

		pca = PCAPreconditioner(X, Y, num_pcs=3)

		Xp, Yp = pca(X, Y)



	def test_pickle(self):
		wt0 = PCAPreconditioner(randn(5, 1000), randn(2, 1000), num_pcs=3)

		tmp_file = mkstemp()[1]

		# store transformation
		with open(tmp_file, 'w') as handle:
			dump({'wt': wt0}, handle)

		# load transformation
		with open(tmp_file) as handle:
			wt1 = load(handle)['wt']

		X, Y = randn(5, 100), randn(2, 100)

		X0, Y0 = wt0(X, Y)
		X1, Y1 = wt1(X, Y)

		# make sure linear transformation hasn't changed
		self.assertLess(max(abs(X0 - X1)), 1e-20)
		self.assertLess(max(abs(Y0 - Y1)), 1e-20)



	def test_logjacobian(self):
		eigenvalues = rand(5) + .5
		meanIn = randn(5, 1)
		meanOut = randn(2, 1)
		whiteIn = randn(5, 5)
		whiteIn = dot(whiteIn, whiteIn.T)
		whiteOut = randn(2, 2)
		whiteOut = dot(whiteOut, whiteOut.T)
		predictor = randn(2, 5)

		wt = PCAPreconditioner(
			eigenvalues,
			meanIn,
			meanOut,
			whiteIn,
			inv(whiteIn),
			whiteOut,
			inv(whiteOut),
			predictor)

		self.assertAlmostEqual(mean(wt.logjacobian(randn(5, 10), randn(2, 10))), slogdet(whiteOut)[1])



if __name__ == '__main__':
	unittest.main()
