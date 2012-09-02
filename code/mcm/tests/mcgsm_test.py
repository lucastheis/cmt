import sys
import unittest

sys.path.append('./code')
sys.path.append('./build/lib.macosx-10.6-intel-2.7')
sys.path.append('./build/lib.linux-x86_64-2.7')

from mcm import MCGSM
from numpy import *
from numpy.random import *
from numpy import max
from scipy.stats import kstest
from scipy.optimize import check_grad
from pickle import dump, load
from tempfile import mkstemp

class Tests(unittest.TestCase):
	def test_basics(self):
		dim_in = 10
		dim_out = 3
		num_components = 7
		num_scales = 5
		num_features = 50
		num_samples = 100

		# create model
		mcgsm = MCGSM(dim_in, dim_out, num_components, num_scales, num_features)

		# generate output
		input = randn(dim_in, num_samples)
		output = mcgsm.sample(input)
		loglik = mcgsm.loglikelihood(input, output)
		post = mcgsm.posterior(input, output)
		samples = mcgsm.sample_posterior(input, output)

		# check hyperparameters
		self.assertEqual(mcgsm.dim_in, dim_in)
		self.assertEqual(mcgsm.dim_out, dim_out)
		self.assertEqual(mcgsm.num_components, num_components)
		self.assertEqual(mcgsm.num_scales, num_scales)
		self.assertEqual(mcgsm.num_features, num_features)
	
		# check parameters
		self.assertEqual(mcgsm.priors.shape[0], num_components)
		self.assertEqual(mcgsm.priors.shape[1], num_scales)
		self.assertEqual(mcgsm.scales.shape[0], num_components)
		self.assertEqual(mcgsm.scales.shape[1], num_scales)
		self.assertEqual(mcgsm.weights.shape[0], num_components)
		self.assertEqual(mcgsm.weights.shape[1], num_features)
		self.assertEqual(mcgsm.features.shape[0], dim_in)
		self.assertEqual(mcgsm.features.shape[1], num_features)
		self.assertEqual(len(mcgsm.cholesky_factors), num_components)
		self.assertEqual(len(mcgsm.predictors), num_components)
		self.assertEqual(mcgsm.cholesky_factors[0].shape[0], dim_out)
		self.assertEqual(mcgsm.cholesky_factors[0].shape[1], dim_out)
		self.assertEqual(mcgsm.predictors[0].shape[0], dim_out)
		self.assertEqual(mcgsm.predictors[0].shape[1], dim_in)

		# check dimensionality of output
		self.assertEqual(output.shape[0], dim_out)
		self.assertEqual(output.shape[1], num_samples)
		self.assertEqual(loglik.shape[0], 1)
		self.assertEqual(loglik.shape[1], num_samples)
		self.assertEqual(loglik.shape[1], num_samples)
		self.assertEqual(post.shape[0], num_components)
		self.assertEqual(post.shape[1], num_samples)
		self.assertEqual(samples.shape[0], 1)
		self.assertEqual(samples.shape[1], num_samples)



	def test_train(self):
		mcgsm = MCGSM(8, 3, 4, 2, 20)

		priors = mcgsm.priors
		scales = mcgsm.scales
		weights = mcgsm.weights
		features = mcgsm.features
		predictor = mcgsm.predictors[0]

		mcgsm.train(
			randn(mcgsm.dim_in, 20000),
			randn(mcgsm.dim_out, 20000),
			parameters={
				'verbosity': 0,
				'max_iter': 0,
				'batch_size': 1999,
				})

		# parameters should not have changed
		self.assertLess(max(abs(mcgsm.priors - priors)), 1e-20)
		self.assertLess(max(abs(mcgsm.scales - scales)), 1e-20)
		self.assertLess(max(abs(mcgsm.weights - weights)), 1e-20)
		self.assertLess(max(abs(mcgsm.features - features)), 1e-20)
		self.assertLess(max(abs(mcgsm.predictors[0] - predictor)), 1e-20)

		# make sure training doesn't throw any errors
		mcgsm.train(
			randn(mcgsm.dim_in, 20000),
			randn(mcgsm.dim_out, 20000),
			parameters={
				'verbosity': 0,
				'max_iter': 1,
				'batch_size': 1999,
				})



	def test_gradient(self):
		mcgsm = MCGSM(5, 2, 2, 4, 10)

		err = mcgsm.check_gradient(
			randn(mcgsm.dim_in, 1000),
			randn(mcgsm.dim_out, 1000), 1e-5)

		self.assertLess(err, 1e-8)



	def test_pickle(self):
		mcgsm0 = MCGSM(11, 2, 4, 7, 21)

		tmp_file = mkstemp()[1]

		# store model
		with open(tmp_file, 'w') as handle:
			dump({'mcgsm': mcgsm0}, handle)

		# load model
		with open(tmp_file) as handle:
			mcgsm1 = load(handle)['mcgsm']

		# make sure parameters haven't changed
		self.assertEqual(mcgsm0.dim_in, mcgsm1.dim_in)
		self.assertEqual(mcgsm0.dim_out, mcgsm1.dim_out)
		self.assertEqual(mcgsm0.num_components, mcgsm1.num_components)
		self.assertEqual(mcgsm0.num_scales, mcgsm1.num_scales)
		self.assertEqual(mcgsm0.num_features, mcgsm1.num_features)

		self.assertLess(max(abs(mcgsm0.scales - mcgsm1.scales)), 1e-20)
		self.assertLess(max(abs(mcgsm0.weights - mcgsm1.weights)), 1e-20)
		self.assertLess(max(abs(mcgsm0.features - mcgsm1.features)), 1e-20)

		for chol0, chol1 in zip(mcgsm0.cholesky_factors, mcgsm1.cholesky_factors):
			self.assertLess(max(abs(chol0 - chol1)), 1e-20)

		for pred0, pred1 in zip(mcgsm0.predictors, mcgsm1.predictors):
			self.assertLess(max(abs(pred0 - pred1)), 1e-20)



if __name__ == '__main__':
	unittest.main()
