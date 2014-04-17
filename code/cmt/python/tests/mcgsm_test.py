import sys
import unittest

from numpy import *
from numpy import min, max
from numpy.linalg import cholesky, inv
from numpy.random import *
from scipy.stats import kstest, ks_2samp, norm
from pickle import dump, load
from tempfile import mkstemp
from cmt.models import MCGSM, MoGSM, PatchMCGSM, GSM
from cmt.transforms import WhiteningPreconditioner

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
		self.assertEqual(mcgsm.linear_features.shape[0], num_components)
		self.assertEqual(mcgsm.linear_features.shape[1], dim_in)
		self.assertEqual(mcgsm.means.shape[0], dim_out)
		self.assertEqual(mcgsm.means.shape[1], num_components)

		# check dimensionality of output
		self.assertEqual(output.shape[0], dim_out)
		self.assertEqual(output.shape[1], num_samples)
		self.assertEqual(loglik.shape[0], 1)
		self.assertEqual(loglik.shape[1], num_samples)
		self.assertEqual(post.shape[0], num_components)
		self.assertEqual(post.shape[1], num_samples)
		self.assertLess(max(samples), mcgsm.num_components)
		self.assertGreaterEqual(min(samples), 0)
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
				})

		# this should raise errors
		self.assertRaises(RuntimeError, mcgsm.train, randn(mcgsm.dim_in - 1, 2000), randn(1, 2000))
		self.assertRaises(RuntimeError, mcgsm.train, randn(mcgsm.dim_in - 1, 2000), randn(2000))
		self.assertRaises(RuntimeError, mcgsm.train,
			randn(mcgsm.dim_in - 1, 2000),
			randn(mcgsm.dim_out, 2000),
			randn(mcgsm.dim_in - 1, 1000),
			randn(mcgsm.dim_out, 1000))

		# parameters should not have changed
		self.assertLess(max(abs(mcgsm.priors - priors)), 1e-20)
		self.assertLess(max(abs(mcgsm.scales - scales)), 1e-20)
		self.assertLess(max(abs(mcgsm.weights - weights)), 1e-20)
		self.assertLess(max(abs(mcgsm.features - features)), 1e-20)
		self.assertLess(max(abs(mcgsm.predictors[0] - predictor)), 1e-20)

		count = []
		def callback(i, mcgsm):
			count.append(i)
			return

		max_iter = 10
		cb_iter = 2

		# make sure training doesn't throw any errors
		mcgsm.train(
			randn(mcgsm.dim_in, 10000),
			randn(mcgsm.dim_out, 10000),
			parameters={
				'verbosity': 0,
				'max_iter': max_iter,
				'threshold': 0.,
				'batch_size': 1999,
				'callback': callback,
				'cb_iter': cb_iter,
				})

		# test callback
		self.assertTrue(range(cb_iter, max_iter + 1, cb_iter) == count)



	def test_mogsm(self):
		mcgsm = MCGSM(
			dim_in=0,
			dim_out=3,
			num_components=2,
			num_scales=2,
			num_features=0)

		p0 = 0.3
		p1 = 0.7
		N = 20000
		m0 = array([[2], [0], [0]])
		m1 = array([[0], [2], [1]])
		C0 = cov(randn(mcgsm.dim_out, mcgsm.dim_out**2))
		C1 = cov(randn(mcgsm.dim_out, mcgsm.dim_out**2))
		input = zeros([0, N])
		output = hstack([
			dot(cholesky(C0), randn(mcgsm.dim_out, round(p0 * N))) + m0,
			dot(cholesky(C1), randn(mcgsm.dim_out, round(p1 * N))) + m1]) * (rand(1, N) + 0.5)

		mcgsm.train(input, output, parameters={
			'verbosity': 0,
			'max_iter': 10,
			'train_means': True})

		mogsm = MoGSM(3, 2, 2)

		# translate parameters from MCGSM to MoGSM
		mogsm.priors = sum(exp(mcgsm.priors), 1) / sum(exp(mcgsm.priors))

		for k in range(mogsm.num_components):
			mogsm[k].mean = mcgsm.means[:, k]
			mogsm[k].covariance = inv(dot(mcgsm.cholesky_factors[k], mcgsm.cholesky_factors[k].T))
			mogsm[k].scales = exp(mcgsm.scales[k, :])
			mogsm[k].priors = exp(mcgsm.priors[k, :]) / sum(exp(mcgsm.priors[k, :]))

		self.assertAlmostEqual(mcgsm.evaluate(input, output), mogsm.evaluate(output), 5)

		mogsm_samples = mogsm.sample(N)
		mcgsm_samples = mcgsm.sample(input)

		# generated samples should have the same distribution
		for i in range(mogsm.dim):
			self.assertTrue(ks_2samp(mogsm_samples[i], mcgsm_samples[0]) > 0.0001)
			self.assertTrue(ks_2samp(mogsm_samples[i], mcgsm_samples[1]) > 0.0001)
			self.assertTrue(ks_2samp(mogsm_samples[i], mcgsm_samples[2]) > 0.0001)

		posterior = mcgsm.posterior(input, mcgsm_samples)

		# average posterior should correspond to prior
		for k in range(mogsm.num_components):
			self.assertLess(abs(1 - mean(posterior[k]) / mogsm.priors[k]), 0.1)



	def test_sample(self):
		mcgsm = MCGSM(1, 1, 1, 1, 1)
		mcgsm.scales = [[0.]]
		mcgsm.predictors = [[0.]]

		samples = mcgsm.sample(zeros([1, 10000])).flatten()

		p = kstest(samples, lambda x: norm.cdf(x, scale=1.))[1]

		# make sure Gaussian random number generation works
		self.assertTrue(p > 0.0001)



	def test_sample_conditionally(self):
		mcgsm = MCGSM(3, 2, 2, 2, 4)

		# make sure there are differences between components
		mcgsm.weights = -log(rand(*mcgsm.weights.shape)) * 10.
		mcgsm.scales = square(mcgsm.scales * 3.)

		inputs = randn(mcgsm.dim_in, 100000)

		# sample directly
		outputs0 = mcgsm.sample(inputs)

		# sample indirectly
		labels = mcgsm.sample_prior(inputs)
		outputs1 = mcgsm.sample(inputs, labels)

		p = ks_2samp(outputs0.ravel(), outputs1.ravel())[1]

		self.assertGreater(p, 1e-5)



	def test_conditional_loglikelihood(self):
		mcgsm = MCGSM(3, 1, 2, 1, 4)

		mcgsm.linear_features = randn(mcgsm.num_components, mcgsm.dim_in) / 5.
		mcgsm.means = randn(mcgsm.dim_out, mcgsm.num_components) / 5.

		M = 100

		inputs = randn(mcgsm.dim_in, M)
		outputs = mcgsm.sample(inputs)

		loglik0 = mcgsm.loglikelihood(inputs, outputs)
		loglik1 = []

		N = 1000

		# estimate log-likelihood via sampling
		for _ in range(N):
			labels = mcgsm.sample_prior(inputs)
			loglik1.append(mcgsm.loglikelihood(inputs, outputs, labels))

		loglik1 = vstack(loglik1)

		d = abs(logmeanexp(loglik1, 0) - loglik0).ravel()
		s = std(loglik1, 0, ddof=1).ravel()

		for i in range(M):
			self.assertLess(d[i], 6. * s[i] / sqrt(N))




	def test_gradient(self):
		mcgsm = MCGSM(5, 2, 2, 4, 10)

		cholesky_factors = []
		for k in range(mcgsm.num_components):
			cholesky_factors.append(cholesky(cov(randn(mcgsm.dim_out, mcgsm.dim_out**2))))
		mcgsm.cholesky_factors = cholesky_factors

		mcgsm.linear_features = randn(mcgsm.num_components, mcgsm.dim_in) / 5.
		mcgsm.means = randn(mcgsm.dim_out, mcgsm.num_components) / 5.

		err = mcgsm._check_gradient(
			randn(mcgsm.dim_in, 1000),
			randn(mcgsm.dim_out, 1000), 1e-5)
		self.assertLess(err, 1e-8)

		# without regularization
		for param in ['priors', 'scales', 'weights', 'features', 'chol', 'pred', 'linear_features', 'means']:
			err = mcgsm._check_gradient(
				randn(mcgsm.dim_in, 1000),
				randn(mcgsm.dim_out, 1000),
				1e-5,
				parameters={
					'train_prior': param == 'priors',
					'train_scales': param == 'scales',
					'train_weights': param == 'weights',
					'train_features': param == 'features',
					'train_cholesky_factors': param == 'chol',
					'train_predictors': param == 'pred',
					'train_linear_features': param == 'linear_features',
					'train_means': param == 'means',
				})
			self.assertLess(err, 1e-8)

		# with regularization
		for norm in ['L1', 'L2']:
			for param in ['priors', 'scales', 'weights', 'features', 'chol', 'pred', 'linear_features', 'means']:
				err = mcgsm._check_gradient(
					randn(mcgsm.dim_in, 1000),
					randn(mcgsm.dim_out, 1000),
					1e-7,
					parameters={
						'train_prior': param == 'priors',
						'train_scales': param == 'scales',
						'train_weights': param == 'weights',
						'train_features': param == 'features',
						'train_cholesky_factors': param == 'chol',
						'train_predictors': param == 'pred',
						'train_linear_features': param == 'linear_features',
						'train_means': param == 'means',
						'regularize_features': {'strength': 0.4, 'norm': norm},
						'regularize_predictors': {'strength': 0.5, 'norm': norm},
						'regularize_weights': {'strength': 0.7, 'norm': norm},
						'regularize_linear_features': {'strength': 0.3, 'norm': norm},
						'regularize_means': {'strength': 0.6, 'norm': norm},
					})
				self.assertLess(err, 1e-6)



	def test_evaluate(self):
		mcgsm = MCGSM(5, 3, 4, 2, 10)

		inputs = randn(mcgsm.dim_in, 100)
		outputs = mcgsm.sample(inputs)

		pre = WhiteningPreconditioner(inputs, outputs)

		loglik1 = -mcgsm.evaluate(inputs, outputs, pre)
		loglik2 = (mcgsm.loglikelihood(*pre(inputs, outputs)).mean() 
			+ pre.logjacobian(inputs, outputs).mean()) / log(2.) / mcgsm.dim_out

		self.assertAlmostEqual(loglik1, loglik2, 8)



	def test_data_gradient(self):
		mcgsm = MCGSM(5, 3, 4, 5, 10)

		cholesky_factors = []
		for k in range(mcgsm.num_components):
			cholesky_factors.append(cholesky(cov(randn(mcgsm.dim_out, mcgsm.dim_out**2))))
		mcgsm.cholesky_factors = cholesky_factors

		inputs = randn(mcgsm.dim_in, 100)
		outputs = ones_like(mcgsm.sample(inputs))

		# compute density gradient and loglikelihood
		dx, dy, ll = mcgsm._data_gradient(inputs, outputs)

		self.assertLess(max(abs(ll - mcgsm.loglikelihood(inputs, outputs))), 1e-8)

		h = 1e-5

		dx_ = zeros_like(dx)
		dy_ = zeros_like(dy)

		for i in range(mcgsm.dim_in):
			inputs_p = inputs.copy()
			inputs_m = inputs.copy()
			inputs_p[i] += h
			inputs_m[i] -= h
			dx_[i] = (
				mcgsm.loglikelihood(inputs_p, outputs) -
				mcgsm.loglikelihood(inputs_m, outputs)) / (2. * h)

		for i in range(mcgsm.dim_out):
			outputs_p = outputs.copy()
			outputs_m = outputs.copy()
			outputs_p[i] += h
			outputs_m[i] -= h
			dy_[i] = (
				mcgsm.loglikelihood(inputs, outputs_p) -
				mcgsm.loglikelihood(inputs, outputs_m)) / (2. * h)

		self.assertLess(max(abs(dy_ - dy)), 1e-8)
		self.assertLess(max(abs(dx_ - dx)), 1e-8)



	def test_pickle(self):
		mcgsm0 = MCGSM(11, 2, 4, 7, 21)

		mcgsm0.linear_features = randn(mcgsm0.num_components, mcgsm0.dim_in)
		mcgsm0.means = randn(mcgsm0.dim_out, mcgsm0.num_components)

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
		self.assertLess(max(abs(mcgsm0.linear_features - mcgsm1.linear_features)), 1e-20)
		self.assertLess(max(abs(mcgsm0.means - mcgsm1.means)), 1e-20)

		for chol0, chol1 in zip(mcgsm0.cholesky_factors, mcgsm1.cholesky_factors):
			self.assertLess(max(abs(chol0 - chol1)), 1e-20)

		for pred0, pred1 in zip(mcgsm0.predictors, mcgsm1.predictors):
			self.assertLess(max(abs(pred0 - pred1)), 1e-20)



	def test_patchmcgsm(self):
		xmask = ones([8, 8], dtype='bool')
		ymask = zeros([8, 8], dtype='bool')
		xmask[-1, -1] = False
		ymask[-1, -1] = True

		model = PatchMCGSM(8, 8, xmask, ymask, model=MCGSM(sum(xmask), 1))

		self.assertLess(max(abs(model.input_mask() - xmask)), 1e-8)
		self.assertLess(max(abs(model.output_mask() - ymask)), 1e-8)

		for i in range(8):
			for j in range(8):
				self.assertEqual(model[i, j].dim_in, (i + 1) * (j + 1) - 1)
				self.assertTrue(isinstance(model[i, j], MCGSM))

		# random pixel ordering
		rows, cols = 7, 5
		order = [(i // cols, i % cols) for i in permutation(rows * cols)]

		model = PatchMCGSM(rows, cols, xmask, ymask, order, MCGSM(sum(xmask), 1))

		self.assertLess(max(abs(model.input_mask() - xmask)), 1e-8)
		self.assertLess(max(abs(model.output_mask() - ymask)), 1e-8)

		for i in range(rows):
			for j in range(cols):
				self.assertEqual(model.input_mask(i, j).sum(), model[i, j].dim_in)

		# test constructors
		model0 = PatchMCGSM(rows, cols, max_pcs=3)
		model1 = PatchMCGSM(rows, cols, model0.input_mask(), model0.output_mask(), model0.order)

		self.assertLess(max(abs(model0.input_mask() - model1.input_mask())), 1e-8)
		self.assertLess(max(abs(model0.output_mask() - model1.output_mask())), 1e-8)
		self.assertLess(max(abs(asarray(model0.order) - asarray(model1.order))), 1e-8)

		# test computation of input masks
		model = PatchMCGSM(rows, cols, order, max_pcs=3)

		i, j = model0.order[0]
		input_mask = model.input_mask(i, j)
		for i, j in model.order[1:]:
			self.assertEqual(sum(model.input_mask(i, j) - input_mask), 1)
			input_mask = model.input_mask(i, j)



	def test_patchmcgsm_train(self):
		xmask = ones([2, 2], dtype='bool')
		ymask = zeros([2, 2], dtype='bool')
		xmask[-1, -1] = False
		ymask[-1, -1] = True

		model = PatchMCGSM(2, 2, xmask, ymask, model=MCGSM(sum(xmask), 1, 1, 1))

		data = randn(4, 10000)

		model.initialize(data)
		converged = model.train(data, parameters={'verbosity': 0, 'max_iter': 200, 'treshold': 1e-4})

		self.assertTrue(converged)



def logsumexp(x, ax=None):
	"""
	Computes the log of the sum of the exp of the entries in x in a numerically
	stable way.

	@type  x: array_like
	@param x: a list, array or matrix of numbers

	@type  ax: integer
	@param ax: axis along which the sum is applied

	@rtype: array
	@return: an array containing the results
	"""

	if ax is None:
		x_max = max(x, ax)
		return x_max + log(sum(exp(x - x_max)))

	else:
		x_max_shape = list(x.shape)
		x_max_shape[ax] = 1

		x_max = asarray(max(x, ax))
		return x_max + log(sum(exp(x - x_max.reshape(x_max_shape)), ax))



def logmeanexp(x, ax=None):
	"""
	Computes the log of the mean of the exp of the entries in x in a numerically
	stable way. Uses logsumexp.

	@type  x: array_like
	@param x: a list, array or matrix of numbers

	@type  ax: integer
	@param ax: axis along which the values are averaged

	@rtype: array
	@return: an array containing the results
	"""

	x = asarray(x)
	n = x.size if ax is None else x.shape[ax]

	return logsumexp(x, ax) - log(n)



if __name__ == '__main__':
	unittest.main()
