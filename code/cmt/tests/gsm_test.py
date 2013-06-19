import unittest
from numpy import *
from numpy import max
from numpy.random import randn, rand
from numpy.linalg import inv, det, slogdet
from tempfile import mkstemp
from pickle import dump
from cmt import GSM

class Tests(unittest.TestCase):
	def test_basics(self):
		gsm = GSM(3, 5)

		self.assertTrue(gsm.scales.size, 5)
		self.assertTrue(gsm.dim, 3)

		covariance = cov(randn(gsm.dim, 10))
		gsm.covariance = covariance

		self.assertLess(max(abs(gsm.covariance - covariance)), 1e-8)



	def test_train(self):
		gsm0 = GSM(3, 2)
		gsm0.mean = [1, 1, 1]
		gsm0.scales = [1, 5]
		gsm0.priors = [0.7, 0.3]
		gsm0.covariance = gsm0.covariance / power(det(gsm0.covariance), 1. / gsm0.dim)

		samples = gsm0.sample(20000)

		# try to recover parameters
		gsm1 = GSM(3, 2)
		gsm1.train(samples, parameters={'max_iter': 50})

		# normalize
		f = power(det(gsm1.covariance), 1. / gsm1.dim)
		gsm1.covariance = gsm1.covariance / f
		gsm1.scales = gsm1.scales / f

		self.assertLess(max(abs(gsm1.mean - gsm0.mean)), 0.2)
		self.assertLess(max(abs(1. - sort(gsm1.priors.ravel()) / sort(gsm0.priors.ravel()))), 0.2)
		self.assertLess(max(abs(1. - sort(gsm1.scales.ravel()) / sort(gsm0.scales.ravel()))), 0.2)
		self.assertLess(max(abs(gsm1.covariance - gsm0.covariance)), 0.2)

		weights = rand(1, samples.shape[1])
		weights /= sum(weights)

		gsm1.train(samples, weights, parameters={'max_iter': 50})

		# normalize
		f = power(det(gsm1.covariance), 1. / gsm1.dim)
		gsm1.covariance = gsm1.covariance / f
		gsm1.scales = gsm1.scales / f

		self.assertLess(max(abs(gsm1.mean - gsm0.mean)), 0.2)
		self.assertLess(max(abs(1. - sort(gsm1.priors.ravel()) / sort(gsm0.priors.ravel()))), 0.2)
		self.assertLess(max(abs(1. - sort(gsm1.scales.ravel()) / sort(gsm0.scales.ravel()))), 0.2)
		self.assertLess(max(abs(gsm1.covariance - gsm0.covariance)), 0.2)



	def test_loglikelihood(self):
		gsm = GSM(3, 1)

		samples = gsm.sample(100000)

		# compute entropy analytically
		entropy = 0.5 * slogdet(2. * pi * e * gsm.covariance / gsm.scales)[1]

		# compare with estimated entropy
		self.assertAlmostEqual(entropy, -mean(gsm.loglikelihood(samples)), 1)



	def test_pickle(self):
		model0 = GSM(7, 9)

		model0.mean = randn(7, 1)
		model0.covariance = cov(randn(7, 20))

		tmp_file = mkstemp()[1]

		# store model
		with open(tmp_file, 'w') as handle:
			dump({'model': model0}, handle)

		# load model
		with open(tmp_file) as handle:
			model1 = load(handle)['model']

		# make sure parameters haven't changed
		self.assertEqual(model0.dim, model1.dim)
		self.assertEqual(model0.num_scales, model1.num_scales)

		self.assertLess(max(abs(model0.scales - model0.scales)), 1e-10)
		self.assertLess(max(abs(model0.priors - model1.priors)), 1e-10)
		self.assertLess(max(abs(model0.mean - model1.mean)), 1e-10)
		self.assertLess(max(abs(model0.covariance - model1.covariance)), 1e-10)



if __name__ == '__main__':
	unittest.main()
