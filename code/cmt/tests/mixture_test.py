import unittest

from cmt import Mixture, GSM
from numpy import *
from numpy import max
from numpy.linalg import cholesky
from numpy.random import *

from matplotlib.pyplot import *

class Test(unittest.TestCase):
	def test_train(self):
		model = Mixture(3)

		model.add_component(GSM(3, 1))
		model.add_component(GSM(3, 1))

		self.assertRaises(Exception, model.add_component, GSM(5))
		self.assertRaises(Exception, model.__getitem__, 2)

		self.assertIsInstance(model[1], GSM)

		p0 = 0.3
		p1 = 0.7
		N = 10000
		m0 = array([[2], [0], [0]])
		m1 = array([[0], [2], [1]])
		C0 = cov(randn(model.dim, model.dim**2))
		C1 = cov(randn(model.dim, model.dim**2))
		data = hstack([
			dot(cholesky(C0), randn(model.dim, int(p0 * N))) + m0,
			dot(cholesky(C1), randn(model.dim, int(p1 * N))) + m1])

		model[0].mean = m0
		model[1].mean = m1
		model[0].covariance = C0
		model[1].covariance = C1
		model[0].scales = [1.]
		model[1].scales = [1.]

		# training shouldn't change the parameters too much
		model.train(data, parameters={'max_iter': 20})

		self.assertLess(abs(1. - model.priors[0] / p0), 0.1)
		self.assertLess(abs(1. - model.priors[1] / p1), 0.1)
		self.assertLess(max(abs(model[0].mean - m0)), 0.1)
		self.assertLess(max(abs(model[1].mean - m1)), 0.1)
		self.assertLess(max(abs(model[0].covariance / model[0].scales - C0)), 0.2)
		self.assertLess(max(abs(model[1].covariance / model[1].scales - C1)), 0.2)



if __name__ == '__main__':
	unittest.main()
