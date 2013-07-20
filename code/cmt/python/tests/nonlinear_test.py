import sys
import unittest

from pickle import dump, load
from tempfile import mkstemp
from numpy import exp, max, abs
from numpy.random import randn
from cmt.nonlinear import LogisticFunction

class Tests(unittest.TestCase):
	def test_logistic_function(self):
		f = LogisticFunction()
		x = randn(1000)
		y = f(x).ravel()

		for i in range(x.size):
			self.assertAlmostEqual(y[i], 1. / (1. + exp(-x[i])))

	
	def test_logistic_function_pickle(self):
		tmp_file = mkstemp()[1]

		f0 = LogisticFunction()

		with open(tmp_file, 'w') as handle:
			dump({'f': f0}, handle)

		with open(tmp_file) as handle:
			f1 = load(handle)['f']

		x = randn(100)
		self.assertLess(max(abs(f0(x) - f1(x))), 1e-6)



if __name__ == '__main__':
	unittest.main()
