import unittest

from numpy import *
from numpy import max
from numpy.linalg import cholesky
from numpy.random import *
from tempfile import mkstemp
from pickle import dump
from cmt.models import Mixture, MoGSM, GSM

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
		N = 20000
		m0 = array([[2], [0], [0]])
		m1 = array([[0], [2], [1]])
		C0 = cov(randn(model.dim, model.dim**2))
		C1 = cov(randn(model.dim, model.dim**2))
		data = hstack([
			dot(cholesky(C0), randn(model.dim, int(p0 * N))) + m0,
			dot(cholesky(C1), randn(model.dim, int(p1 * N))) + m1])

		# if this is not call train() will initialize the parameters
		model.initialize(data)

		model[0].mean = m0
		model[1].mean = m1
		model[0].covariance = C0
		model[1].covariance = C1
		model[0].scales = [1.]
		model[1].scales = [1.]

		# training shouldn't change the parameters too much
		model.train(data, parameters={'verbosity': 0, 'max_iter': 20, 'threshold': 1e-7})

		self.assertLess(abs(1. - model.priors[0] / p0), 0.1)
		self.assertLess(abs(1. - model.priors[1] / p1), 0.1)
		self.assertLess(max(abs(model[0].mean - m0)), 0.2)
		self.assertLess(max(abs(model[1].mean - m1)), 0.2)
		self.assertLess(max(abs(model[0].covariance / model[0].scales - C0)), 0.2)
		self.assertLess(max(abs(model[1].covariance / model[1].scales - C1)), 0.2)



	def test_pickle(self):
		models = [
			Mixture(dim=5),
			MoGSM(dim=3, num_components=4, num_scales=7)]

		for _ in range(3):
			models[0].add_component(GSM(models[0].dim, 7))

		for model0 in models:
			tmp_file = mkstemp()[1]

			# store model
			with open(tmp_file, 'wb') as handle:
				dump({'model': model0}, handle)

			# load model
			with open(tmp_file, 'rb') as handle:
				model1 = load(handle)['model']

			# make sure parameters haven't changed
			self.assertEqual(model0.dim, model1.dim)
			self.assertEqual(model0.num_components, model1.num_components)

			for k in range(model0.num_components):
				self.assertLess(max(abs(model0[k].scales - model0[k].scales)), 1e-10)
				self.assertLess(max(abs(model0[k].priors - model1[k].priors)), 1e-10)
				self.assertLess(max(abs(model0[k].mean - model1[k].mean)), 1e-10)
				self.assertLess(max(abs(model0[k].covariance - model1[k].covariance)), 1e-10)



if __name__ == '__main__':
	unittest.main()
