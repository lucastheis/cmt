import sys
import unittest

sys.path.append('./code')
sys.path.append('./build/lib.macosx-10.6-intel-2.7')
sys.path.append('./build/lib.linux-x86_64-2.7')

from mcm import WhiteningTransform
from numpy import *
from numpy.random import *
from numpy import max, round
from pickle import dump, load
from tempfile import mkstemp

class Tests(unittest.TestCase):
	def test_transform(self):
		A = randn(5, 5)
		X = dot(A, randn(5, 1000))

		wt = WhiteningTransform(X)

		self.assertLess(max(abs(cov(wt(X), bias=1) - eye(5))), 1e-10)
		self.assertLess(max(abs(wt.inverse(wt.A) - eye(5))), 1e-10)

		# whitening transform should be symmetric
		self.assertLess(max(abs(wt.A - wt.A.T)), 1e-10)



	def test_pickle(self):
		wt0 = WhiteningTransform(randn(5, 1000))

		tmp_file = mkstemp()[1]

		# store transformation
		with open(tmp_file, 'w') as handle:
			dump({'wt': wt0}, handle)

		# load transformation
		with open(tmp_file) as handle:
			wt1 = load(handle)['wt']

		# make sure linear transformation hasn't changed
		self.assertLess(max(abs(wt0.A - wt1.A)), 1e-20)



if __name__ == '__main__':
	unittest.main()
