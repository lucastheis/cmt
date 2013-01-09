import sys
import unittest

sys.path.append('./code')

from mcm import PCATransform
from numpy import *
from numpy.random import *
from numpy import max, round
from pickle import dump, load
from tempfile import mkstemp

class Tests(unittest.TestCase):
	def test_transform(self):
		A = randn(5, 5)
		X = dot(A, randn(5, 1000))

		pca = PCATransform(X)

		self.assertLess(max(abs(pca.b.ravel() + mean(X, 1))), 1e-8)
		self.assertLess(max(abs(cov(pca(X), bias=1) - eye(5))), 1e-10)
		self.assertLess(max(abs(pca.inverse(pca.A + pca.b) - eye(5))), 1e-10)



	def test_pickle(self):
		pca0 = PCATransform(randn(5, 1000))

		tmp_file = mkstemp()[1]

		# store transformation
		with open(tmp_file, 'w') as handle:
			dump({'pca': pca0}, handle)

		# load transformation
		with open(tmp_file) as handle:
			pca1 = load(handle)['pca']

		# make sure linear transformation hasn't changed
		self.assertLess(max(abs(pca0.A - pca1.A)), 1e-20)



if __name__ == '__main__':
	unittest.main()
