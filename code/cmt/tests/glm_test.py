import sys
import unittest

sys.path.append('./code')

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
		x = x

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

		x = randn(4, 100000)
		p = 1. / (1. + exp(-dot(w.T, x)))
		y = rand(*p.shape) < p

		glm = GLM(4, LogisticFunction, Bernoulli)

		err = glm._check_gradient(x, y, 1e-5)
		self.assertLess(err, 1e-8)

		glm.train(x, y, parameters={'verbosity': 0})

		self.assertLess(max(abs(glm.weights - w)), 0.1)



if __name__ == '__main__':
	unittest.main()
