import sys
import unittest

from numpy import *
from numpy import sum, max, round
from numpy.random import *
from numpy.linalg import inv, slogdet
from pickle import dump, load
from tempfile import mkstemp
from cmt.transforms import AffinePreconditioner, WhiteningPreconditioner, PCAPreconditioner
from cmt.transforms import AffineTransform, WhiteningTransform, PCATransform
from cmt.transforms import BinningTransform

class Tests(unittest.TestCase):
	def test_adjust_gradient(self):
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

		f = lambda X, Y: sum(hstack([p.ravel() for p in pre(X, Y)]))

		# compute analytic gradient
		dfdX, dfdY = pre.adjust_gradient(ones_like(X), ones_like(Y))

		# compute numerical gradient
		h = 0.1
		dfdXn = zeros_like(X)
		X_copy = X.copy()
		for i in range(X.shape[0]):
			for j in range(X.shape[1]):
				X_copy[i, j] = X[i, j] + h
				fp = f(X_copy, Y)

				X_copy[i, j] = X[i, j] - h
				fn = f(X_copy, Y)

				dfdXn[i, j] = (fp - fn) / (2. * h)

				X_copy[i, j] = X[i, j]

		self.assertLess(max(abs(dfdXn - dfdX)), 1e-7)




	def test_affine_preconditioner(self):
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

		# check that preconditioner does what it's expected to do
		self.assertEqual(pre.dim_in, X.shape[0])
		self.assertEqual(pre.dim_out, Y.shape[0])
		self.assertLess(max(abs(Xp - dot(preIn, X - meanIn))), 1e-10)
		self.assertLess(max(abs(Yp - dot(preOut, Y - meanOut - dot(predictor, Xp)))), 1e-10)

		# check that inverse works
		self.assertLess(max(abs(Xr - X)), 1e-10)
		self.assertLess(max(abs(Yr - Y)), 1e-10)

		# reference counts should not change
		Xrc = sys.getrefcount(X)
		Yrc = sys.getrefcount(Y)

		for i in range(10):
			pre(X, Y)

		self.assertEqual(sys.getrefcount(X), Xrc)
		self.assertEqual(sys.getrefcount(Y), Yrc)



	def test_affine_transform(self):
		X = dot(randn(5, 5), randn(5, 1000)) + randn(5, 1)
		Y = dot(randn(2, 2), randn(2, 1000)) + dot(randn(2, 5), X)

		meanIn = randn(5, 1)
		preIn = randn(5, 5)

		pre = AffineTransform(meanIn, preIn, Y.shape[0])
		self.assertLess(max(abs(pre(X) - dot(preIn, X - meanIn))), 1e-10)

		# test inverse
		Xp, Yp = pre(X, Y)
		Xr, Yr = pre.inverse(Xp, Yp)

		self.assertLess(max(abs(Xr - X)), 1e-10)
		self.assertLess(max(abs(Yr - Y)), 1e-10)

		# reference counts should not change
		Xrc = sys.getrefcount(X)
		Yrc = sys.getrefcount(Y)

		for i in range(10):
			pre(X, Y)

		self.assertEqual(sys.getrefcount(X), Xrc)
		self.assertEqual(sys.getrefcount(Y), Yrc)



	def test_affine_preconditioner_pickle(self):
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

		# test inverse after pickling
		Xp, Yp = pre1(X, Y)
		Xr, Yr = pre1.inverse(Xp, Yp)

		self.assertLess(max(abs(Xr - X)), 1e-10)
		self.assertLess(max(abs(Yr - Y)), 1e-10)



	def test_affine_transform_pickle(self):
		meanIn = randn(5, 1)
		preIn = randn(5, 5)
		dim_out = 3

		pre0 = AffineTransform(meanIn, preIn, dim_out)

		tmp_file = mkstemp()[1]

		# store transformation
		with open(tmp_file, 'w') as handle:
			dump({'pre': pre0}, handle)

		# load transformation
		with open(tmp_file) as handle:
			pre1 = load(handle)['pre']

		X, Y = randn(5, 100), randn(3, 100)

		X0, Y0 = pre0(X, Y)
		X1, Y1 = pre1(X, Y)

		# make sure linear transformation hasn't changed
		self.assertLess(max(abs(X0 - X1)), 1e-20)
		self.assertLess(max(abs(Y0 - Y1)), 1e-20)



	def test_affine_preconditioner_logjacobian(self):
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



	def test_whitening_preconditioner(self):
		X = dot(randn(5, 5), randn(5, 1000)) + randn(5, 1)
		Y = dot(randn(2, 2), randn(2, 1000)) + dot(randn(2, 5), X)

		wt = WhiteningPreconditioner(X, Y)

		# joint covariance
		C = cov(vstack(wt(X, Y)), bias=True)

		self.assertLess(max(abs(C - eye(7))), 1e-8)

		# test inverse
		Xw, Yw = wt(X, Y)
		Xr, Yr = wt.inverse(Xw, Yw)

		self.assertLess(max(abs(Xr - X)), 1e-10)
		self.assertLess(max(abs(Yr - Y)), 1e-10)



	def test_whitening_preconditioner_pickle(self):
		wt0 = WhiteningPreconditioner(randn(5, 1000), randn(2, 1000))

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



	def test_whitening_transform(self):
		X = dot(randn(5, 5), randn(5, 1000)) + randn(5, 1)
		Y = dot(randn(2, 2), randn(2, 1000)) + dot(randn(2, 5), X)

		wt = WhiteningTransform(X, Y)
		C = cov(wt(X), bias=True)
		self.assertLess(max(abs(C - eye(5))), 1e-8)

		wt = WhiteningTransform(X, dim_out=Y.shape[0])
		C = cov(wt(X), bias=True)
		self.assertLess(max(abs(C - eye(5))), 1e-8)

		# test inverse
		Xw, Yw = wt(X, Y)
		Xr, Yr = wt.inverse(Xw, Yw)

		self.assertLess(max(abs(Xr - X)), 1e-10)
		self.assertLess(max(abs(Yw - Y)), 1e-10)
		self.assertLess(max(abs(Yr - Y)), 1e-10)



	def test_whitening_transform_pickle(self):
		wt0 = WhiteningTransform(randn(5, 1000), randn(2, 1000))

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



	def test_pca_preconditioner(self):
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

		# test inverse
		Xp, Yp = pca(X, Y)
		Xr, Yr = pca.inverse(Xp, Yp)

		self.assertLess(max(abs(Yr - Y)), 1e-10)



	def test_pca_preconditioner_pickle(self):
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



	def test_pca_preconditioner_logjacobian(self):
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



	def test_pca_transform(self):
		X = dot(randn(5, 5), randn(5, 1000)) + randn(5, 1)
		Y = dot(randn(2, 2), randn(2, 1000)) + dot(randn(2, 5), X)

		wt = PCATransform(X, Y, num_pcs=X.shape[0])

		# joint covariance
		C = cov(wt(X), bias=True)

		self.assertLess(max(abs(C - eye(5))), 1e-8)

		wt = PCATransform(X, Y, var_explained=100.)

		# joint covariance
		C = cov(wt(X), bias=True)

		self.assertLess(max(abs(C - eye(5))), 1e-8)

		# test inverse
		Xw, Yw = wt(X, Y)
		Xr, Yr = wt.inverse(Xw, Yw)

		self.assertLess(max(abs(Xr - X)), 1e-10)
		self.assertLess(max(abs(Yr - Y)), 1e-10)

		wt = PCATransform(X, dim_out=Y.shape[0], num_pcs=X.shape[0] - 1)

		# joint covariance
		C = cov(wt(X), bias=True)

		self.assertLess(max(abs(C - eye(4))), 1e-8)

		# test inverse
		Xw, Yw = wt(X, Y)
		Xr, Yr = wt.inverse(Xw, Yw)

		self.assertLess(max(abs(Yr - Y)), 1e-10)




	def test_pca_transform_pickle(self):
		wt0 = PCATransform(randn(5, 1000), randn(2, 1000), num_pcs=3)

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



	def test_binning_transform(self):
		pre = BinningTransform(4, 12, 2)
		
		self.assertEqual(pre.dim_in, 12)
		self.assertEqual(pre.dim_in_pre, 12 / pre.binning)

		x = array([[1, 1, 1, 1, 0, 1, 2, 3, -1, 1, -1, 1]]).T
		y = pre(x)

		self.assertEqual(y[0], 4.)
		self.assertEqual(y[1], 6.)
		self.assertEqual(y[2], 0.)



	def test_binning_transform_pickle(self):
		pre0 = BinningTransform(3, 9, 5)

		tmp_file = mkstemp()[1]

		# store transformation
		with open(tmp_file, 'w') as handle:
			dump({'pre': pre0}, handle)

		# load transformation
		with open(tmp_file) as handle:
			pre1 = load(handle)['pre']

		# make sure linear transformation hasn't changed
		self.assertEqual(pre1.dim_in, pre0.dim_in)
		self.assertEqual(pre1.dim_in_pre, pre0.dim_in_pre)
		self.assertEqual(pre1.dim_out, pre0.dim_out)
		self.assertEqual(pre1.dim_out_pre, pre0.dim_out_pre)
		self.assertEqual(pre1.binning, pre0.binning)



if __name__ == '__main__':
	unittest.main()
