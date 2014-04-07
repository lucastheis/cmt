import sys
import unittest

from pickle import dump, load
from tempfile import mkstemp
from numpy import *
from numpy import max
from numpy.linalg import inv
from numpy.random import randn, rand
from cmt.models import Bernoulli, GLM
from cmt.nonlinear import LogisticFunction, BlobNonlinearity

class Tests(unittest.TestCase):
	def test_glm_basics(self):
		glm = GLM(4, LogisticFunction, Bernoulli)

		x = randn(1000)
		f = glm.nonlinearity
		y = f(x).ravel()

		for i in range(x.size):
			self.assertAlmostEqual(y[i], 1. / (1. + exp(-x[i])))

		glm.nonlinearity = f
		y = glm.nonlinearity(x).ravel()

		for i in range(x.size):
			self.assertAlmostEqual(y[i], 1. / (1. + exp(-x[i])))

		b = Bernoulli()

		glm = GLM(4, f, b)

		glm.nonlinearity = f
		y = glm.nonlinearity(x).ravel()

		for i in range(x.size):
			self.assertAlmostEqual(y[i], 1. / (1. + exp(-x[i])))

		self.assertTrue(isinstance(glm.distribution, Bernoulli))

		# test wrong order of arguments
		self.assertRaises(TypeError, lambda: GLM(5, Bernoulli, LogisticFunction))



	def test_glm_train(self):
		w = asarray([[-1., 0., 1., 2.]]).T
		b = 1.

		x = randn(4, 100000)
		p = 1. / (1. + exp(-dot(w.T, x) - b))
		y = rand(*p.shape) < p

		glm = GLM(4, LogisticFunction, Bernoulli)

		# test gradient
		err = glm._check_gradient(x, y, 1e-5, parameters={
			'train_weights': False,
			'train_bias': True})
		self.assertLess(err, 1e-8)

		err = glm._check_gradient(x, y, 1e-5, parameters={
			'train_weights': True,
			'train_bias': False})
		self.assertLess(err, 1e-8)

		err = glm._check_gradient(x, y, 1e-5)
		self.assertLess(err, 1e-8)

		err = glm._check_gradient(x, y, 1e-5, parameters={
			'regularize_weights': 10.,
			'regularize_bias': 10.})
		self.assertLess(err, 1e-8)

		# test training
		glm.train(x, y, parameters={'verbosity': 0})

		self.assertLess(max(abs(glm.weights - w)), 0.1)
		self.assertLess(max(abs(glm.bias - b)), 0.1)

		glm.weights = w
		glm.bias = -1.

		glm.train(x, y, parameters={'verbosity': 0, 'train_weights': False})

		self.assertLess(max(abs(glm.weights - w)), 1e-12)
		self.assertLess(max(abs(glm.bias - b)), 0.1)

		glm.weights = randn(*glm.weights.shape)
		glm.bias = b

		glm.train(x, y, parameters={'verbosity': 0, 'train_bias': False})

		self.assertLess(max(abs(glm.weights - w)), 0.1)
		self.assertLess(max(abs(glm.bias - b)), 1e-12)



	def test_glm_fisher_information(self):
		N = 2000
		T = 1000

		glm = GLM(4)
		glm.weights = randn(glm.dim_in, 1)
		glm.bias = -2.

		inputs = randn(glm.dim_in, N)
		outputs = glm.sample(inputs)

		x = glm._parameters()
		I = glm._fisher_information(inputs, outputs)

		x_mle = []

		# repeated maximum likelihood estimation
		for t in range(T):
			inputs = randn(glm.dim_in, N)
			outputs = glm.sample(inputs)

			# initialize at true parameters for fast convergence
			glm_ = GLM(glm.dim_in)
			glm_.weights = glm.weights
			glm_.bias = glm.bias
			glm_.train(inputs, outputs)

			x_mle.append(glm_._parameters())

		C = cov(hstack(x_mle), ddof=1)

		# inv(I) should be sufficiently close to C
		self.assertLess(max(abs(inv(I) - C) / (abs(C) + .1)), max(abs(C) / (abs(C) + .1)) / 2.)



	def test_glm_data_gradient(self):
		glm = GLM(7, LogisticFunction, Bernoulli)

		x = randn(glm.dim_in, 100)
		y = glm.sample(x)

		dx, _, ll = glm._data_gradient(x, y)

		h = 1e-7

		# compute numerical gradient
		dx_ = zeros_like(dx)

		for i in range(glm.dim_in):
			x_p = x.copy()
			x_m = x.copy()
			x_p[i] += h
			x_m[i] -= h
			dx_[i] = (
				glm.loglikelihood(x_p, y) -
				glm.loglikelihood(x_m, y)) / (2. * h)

		self.assertLess(max(abs(ll - glm.loglikelihood(x, y))), 1e-8)
		self.assertLess(max(abs(dx_ - dx)), 1e-7)



	def test_glm_pickle(self):
		tmp_file = mkstemp()[1]

		model0 = GLM(5, BlobNonlinearity, Bernoulli)
		model0.weights = randn(*model0.weights.shape)
		model0.bias = randn()

		# store model
		with open(tmp_file, 'w') as handle:
			dump({'model': model0}, handle)

		# load model
		with open(tmp_file) as handle:
			model1 = load(handle)['model']

		# make sure parameters haven't changed
		self.assertLess(max(abs(model0.bias - model1.bias)), 1e-20)
		self.assertLess(max(abs(model0.weights - model1.weights)), 1e-20)

		x = randn(model0.dim_in, 100)
		y = model0.sample(x)
		self.assertEqual(
			model0.evaluate(x, y),
			model1.evaluate(x, y))



if __name__ == '__main__':
	unittest.main()
