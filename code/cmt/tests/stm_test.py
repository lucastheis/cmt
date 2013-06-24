import sys
import unittest

sys.path.append('./code')

from cmt import STM
from numpy import *
from numpy import max, min
from numpy.random import *
from pickle import dump, load
from tempfile import mkstemp

class Tests(unittest.TestCase):
	def test_basics(self):
		dim_in_nonlinear = 10
		dim_in_linear = 8
		num_components = 7
		num_features = 50
		num_samples = 100

		# create model
		stm = STM(dim_in_nonlinear, dim_in_linear, num_components, num_features)

		# generate output
		input_nonlinear = randint(2, size=[dim_in_nonlinear, num_samples])
		input_linear = randint(2, size=[dim_in_linear, num_samples])
		input = vstack([input_nonlinear, input_linear])

		output = stm.sample(input)
		loglik = stm.loglikelihood(input, output)

		# check hyperparameters
		self.assertEqual(stm.dim_in, dim_in_linear + dim_in_nonlinear)
		self.assertEqual(stm.num_components, num_components)
		self.assertEqual(stm.num_features, num_features)
	
		# check parameters
		self.assertEqual(stm.biases.shape[0], num_components)
		self.assertEqual(stm.biases.shape[1], 1)
		self.assertEqual(stm.weights.shape[0], num_components)
		self.assertEqual(stm.weights.shape[1], num_features)
		self.assertEqual(stm.features.shape[0], dim_in_nonlinear)
		self.assertEqual(stm.features.shape[1], num_features)
		self.assertEqual(stm.predictors.shape[0], num_components)
		self.assertEqual(stm.predictors.shape[1], dim_in_nonlinear)
		self.assertEqual(stm.linear_predictor.shape[0], dim_in_linear)
		self.assertEqual(stm.linear_predictor.shape[1], 1)

		# check dimensionality of output
		self.assertEqual(output.shape[0], 1)
		self.assertEqual(output.shape[1], num_samples)
		self.assertEqual(loglik.shape[0], 1)
		self.assertEqual(loglik.shape[1], num_samples)



#	def test_train(self):
#		stm = STM(8, 4, 20)
#
#		parameters = stm._parameters()
#
#		stm.train(
#			randint(2, size=[stm.dim_in, 2000]),
#			randint(2, size=[stm.dim_out, 2000]),
#			parameters={
#				'verbosity': 0,
#				'max_iter': 0,
#				})
#
#		# parameters should not have changed
#		self.assertLess(max(abs(stm._parameters() - parameters)), 1e-20)
#
#		def callback(i, stm):
#			return
#
#		stm.train(
#			randint(2, size=[stm.dim_in, 10000]),
#			randint(2, size=[stm.dim_out, 10000]),
#			parameters={
#				'verbosity': 0,
#				'max_iter': 10,
#				'threshold': 0.,
#				'batch_size': 1999,
#				'callback': callback,
#				'cb_iter': 1,
#				})



	def test_gradient(self):
		stm = STM(5, 2, 10)

		# choose random parameters
		stm._set_parameters(randn(*stm._parameters().shape) / 100.)

		err = stm._check_gradient(
			randn(stm.dim_in, 1000),
			randint(2, size=[stm.dim_out, 1000]), 1e-5)
		self.assertLess(err, 1e-8)

		# test with regularization turned off
		for param in ['biases', 'weights', 'features', 'pred', 'linear_predictor']:
			err = stm._check_gradient(
				randn(stm.dim_in, 1000),
				randint(2, size=[stm.dim_out, 1000]),
				1e-6,
				parameters={
					'train_biases': param == 'biases',
					'train_weights': param == 'weights',
					'train_features': param == 'features',
					'train_predictors': param == 'pred',
					'train_linear_predictor': param == 'linear_predictor',
				})
			self.assertLess(err, 1e-7)

		# test with regularization turned on
		for regularizer in ['L1', 'L2']:
			for param in ['priors', 'weights', 'features', 'pred', 'input_bias', 'output_bias']:
				err = stm._check_gradient(
					randint(2, size=[stm.dim_in, 1000]),
					randint(2, size=[stm.dim_out, 1000]),
					1e-7,
					parameters={
						'train_prior': param == 'priors',
						'train_weights': param == 'weights',
						'train_features': param == 'features',
						'train_predictors': param == 'pred',
						'train_input_bias': param == 'input_bias',
						'train_output_bias': param == 'output_bias',
						'regularizer': regularizer,
						'regularize_features': 0.5,
						'regularize_predictors': 0.4,
						'regularize_weights': 0.7,
					})
				self.assertLess(err, 1e-6)

		self.assertFalse(any(isnan(
			stm._parameter_gradient(
				randint(2, size=[stm.dim_in, 1000]),
				randint(2, size=[stm.dim_out, 1000]),
				stm._parameters()))))



	def test_pickle(self):
		stm0 = STM(5, 10, 4, 21)

		tmp_file = mkstemp()[1]

		# store model
		with open(tmp_file, 'w') as handle:
			dump({'stm': stm0}, handle)

		# load model
		with open(tmp_file) as handle:
			stm1 = load(handle)['stm']

		# make sure parameters haven't changed
		self.assertEqual(stm0.dim_in, stm1.dim_in)
		self.assertEqual(stm0.dim_in_nonlinear, stm1.dim_in_nonlinear)
		self.assertEqual(stm0.dim_in_linear, stm1.dim_in_linear)
		self.assertEqual(stm0.num_components, stm1.num_components)
		self.assertEqual(stm0.num_features, stm1.num_features)

		self.assertLess(max(abs(stm0.biases - stm1.biases)), 1e-20)
		self.assertLess(max(abs(stm0.weights - stm1.weights)), 1e-20)
		self.assertLess(max(abs(stm0.features - stm1.features)), 1e-20)
		self.assertLess(max(abs(stm0.predictors - stm1.predictors)), 1e-20)
		self.assertLess(max(abs(stm0.linear_predictor - stm1.linear_predictor)), 1e-20)



if __name__ == '__main__':
	unittest.main()
