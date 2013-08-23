import sys
import unittest

from pickle import dump, load
from tempfile import mkstemp
from numpy import exp, max, abs
from numpy.random import randn
from cmt.nonlinear import LogisticFunction, ExponentialFunction, HistogramNonlinearity

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



	def test_exponential_function(self):
		f = ExponentialFunction()
		x = randn(1000)
		y = f(x).ravel()

		for i in range(x.size):
			self.assertAlmostEqual(y[i], exp(x[i]))



	def test_exponential_function_pickle(self):
		tmp_file = mkstemp()[1]

		f0 = ExponentialFunction()

		with open(tmp_file, 'w') as handle:
			dump({'f': f0}, handle)

		with open(tmp_file) as handle:
			f1 = load(handle)['f']

		x = randn(100)
		self.assertLess(max(abs(f0(x) - f1(x))), 1e-6)



	def test_histogram_nonlinearity(self):
		inputs = [1, 1, 2, 3]
		outputs = [4, 3, 2, 1]

		f = HistogramNonlinearity(inputs, outputs, 3)

		# output should be average
		self.assertLess(max(abs(f([-1, 1, 2, 3, 10]).ravel() - [3.5, 3.5, 2, 1, 1])), 1e-8)



if __name__ == '__main__':
	unittest.main()
