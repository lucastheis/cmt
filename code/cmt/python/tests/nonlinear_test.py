import sys
import unittest

from pickle import dump, load
from tempfile import mkstemp
from numpy import exp, max, abs
from numpy.random import randn, rand
from cmt.nonlinear import LogisticFunction, ExponentialFunction, HistogramNonlinearity
from cmt.nonlinear import BlobNonlinearity
from cmt.models import GLM

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



	def test_histogram_nonlinearity_pickle(self):
		tmp_file = mkstemp()[1]

		f0 = HistogramNonlinearity(randn(1, 100), rand(1, 100), 10)

		# store model
		with open(tmp_file, 'w') as handle:
			dump({'f': f0}, handle)

		# load model
		with open(tmp_file) as handle:
			f1 = load(handle)['f']

		x = randn(1, 100)

		self.assertLess(max(abs(f0(x) - f1(x))), 1e-6)



	def test_blob_nonlinearity(self):
		# generate test data
		x = randn(1, 10000) * 4.
		y = exp(-(x - 2.)**2) / 2. + exp(-(x + 5.)**2 / 4.) / 4.
		z = (rand(*y.shape) < y) * 1.

		glm = GLM(1, BlobNonlinearity(3))
		glm.weights = [[.5 + rand()]]

		err = glm._check_gradient(x, z, 
			parameters={'train_weights': False, 'train_bias': False, 'train_nonlinearity': True})

		self.assertLess(err, 1e-6)

		err = glm._check_gradient(x, z, 
			parameters={'train_weights': True, 'train_bias': False, 'train_nonlinearity': False})

		self.assertLess(err, 1e-6)



	def test_blob_nonlinearity_pickle(self):
		tmp_file = mkstemp()[1]

		f0 = BlobNonlinearity(4, 0.1)

		# store model
		with open(tmp_file, 'w') as handle:
			dump({'f': f0}, handle)

		# load model
		with open(tmp_file) as handle:
			f1 = load(handle)['f']

		x = randn(1, 100)

		self.assertLess(max(abs(f0(x) - f1(x))), 1e-6)



if __name__ == '__main__':
	unittest.main()
