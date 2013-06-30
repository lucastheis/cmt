import sys
import unittest

sys.path.append('./code')

from pickle import dump, load
from tempfile import mkstemp
from numpy import *
from numpy import max
from numpy.random import randn, rand
from cmt import Bernoulli
from cmt import LogisticFunction
from cmt import GLM

class Tests(unittest.TestCase):
	def test_bernoulli(self):
		bernoulli = Bernoulli(.725)
		self.assertAlmostEqual(mean(bernoulli.sample(1000000)), .725, 2)

		samples = bernoulli.sample(10)

		loglik = log(.725) * samples + log(0.275) * (1. - samples)
		self.assertLess(max(abs(loglik - bernoulli.loglikelihood(samples))), 1e-8)



	def test_logistic_function(self):
		f = LogisticFunction()
		x = randn(1000)
		y = f(x).ravel()

		for i in range(x.size):
			self.assertAlmostEqual(y[i], 1. / (1. + exp(-x[i])))



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



	def test_glm_train(self):
		w = asarray([[-1., 0., 1., 2.]]).T
		b = 1.

		x = randn(4, 100000)
		p = 1. / (1. + exp(-dot(w.T, x) - b))
		y = rand(*p.shape) < p

		glm = GLM(4, LogisticFunction, Bernoulli)

		err = glm._check_gradient(x, y, 1e-5)
		self.assertLess(err, 1e-8)

		glm.train(x, y, parameters={'verbosity': 0})

		self.assertLess(max(abs(glm.weights - w)), 0.1)
		self.assertLess(max(abs(glm.bias - b)), 0.1)



	def test_glm_pickle(self):
		f0 = LogisticFunction()

		tmp_file = mkstemp()[1]

		with open(tmp_file, 'w') as handle:
			dump({'f': f0}, handle)

		with open(tmp_file) as handle:
			f1 = load(handle)['f']

		x = randn(100)
		self.assertLess(max(abs(f0(x) - f1(x))), 1e-6)

		p0 = Bernoulli(.3)

		with open(tmp_file, 'w') as handle:
			dump({'p': p0}, handle)

		with open(tmp_file) as handle:
			p1 = load(handle)['p']

		x = p0.sample(100)
		self.assertLess(max(abs(p0.loglikelihood(x) - p1.loglikelihood(x))), 1e-6)

		model0 = GLM(5, LogisticFunction, Bernoulli)
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
		model1.evaluate(x, model1.sample(x))



if __name__ == '__main__':
	unittest.main()
