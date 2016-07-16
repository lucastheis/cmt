import unittest

from tempfile import mkstemp
from numpy import *
from numpy import max, abs
from numpy.random import randn, randint
from cmt.models import MLR
from pickle import load, dump

class Tests(unittest.TestCase):
	def test_mlr_train(self):
		mlr = MLR(3, 3)
		
		N = 1000
		inputs = zeros([3, N])
		inputs[randint(3, size=N), range(N)] = 1.

		self.assertLess(mlr._check_gradient(inputs, inputs, 1e-4), 1e-6)

		mlr.train(inputs, inputs)

		# prediction should be perfect (almost always)
		self.assertLess(sum(mlr.sample(inputs) - inputs), 2)



	def test_gradient(self):
		mlr = MLR(10, 5)

		N = 1000

		inputs = randn(mlr.dim_in, N)
		outputs = zeros([mlr.dim_out, N])
		outputs[randint(mlr.dim_out, size=N), range(N)] = 1.

		err = mlr._check_gradient( inputs, outputs, 1e-5)
		self.assertLess(err, 1e-8)

		# without regularization
		for param in ['weights', 'biases']:
			err = mlr._check_gradient(
				inputs,
				outputs,
				1e-5,
				parameters={
					'train_weights': param == 'weights',
					'train_biases': param == 'biases',
				})
			self.assertLess(err, 1e-8)

		# with regularization
		for norm in ['L1', 'L2']:
			for param in ['weights', 'biases']:
				err = mlr._check_gradient(
					inputs,
					outputs,
					1e-7,
					parameters={
						'train_weights': param == 'weights',
						'train_biases': param == 'biases',
						'regularize_weights': {'strength': 0.7, 'norm': norm},
						'regularize_biases': {'strength': 0.3, 'norm': norm},
					})
				self.assertLess(err, 1e-6)



	def test_mlr_pickle(self):
		tmp_file = mkstemp()[1]

		model0 = MLR(10, 3)
		model0.weights = randn(*model0.weights.shape)
		model0.biases = randn(*model0.biases.shape)

		# store model
		with open(tmp_file, 'w') as handle:
			dump({'model': model0}, handle)

		# load model
		with open(tmp_file) as handle:
			model1 = load(handle)['model']

		# make sure parameters haven't changed
		self.assertEqual(model0.dim_in, model1.dim_in)
		self.assertEqual(model0.dim_out, model1.dim_out)
		self.assertLess(max(abs(model0.biases - model1.biases)), 1e-20)
		self.assertLess(max(abs(model0.weights - model1.weights)), 1e-20)



if __name__ == '__main__':
	unittest.main()
