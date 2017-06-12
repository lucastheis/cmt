import sys
import unittest

from numpy import *
from numpy.random import randn, permutation
from numpy import max
from pickle import dump, load
from tempfile import mkstemp
from cmt.models import FVBN, GLM, Bernoulli
from cmt.nonlinear import LogisticFunction

class Tests(unittest.TestCase):
	def test_fvbn(self):
		xmask = ones([8, 8], dtype='bool')
		ymask = zeros([8, 8], dtype='bool')
		xmask[-1, -1] = False
		ymask[-1, -1] = True

		model = FVBN(8, 8, xmask, ymask)

		self.assertLess(max(abs(model.input_mask() - xmask)), 1e-8)
		self.assertLess(max(abs(model.output_mask() - ymask)), 1e-8)

		for i in range(8):
			for j in range(8):
				self.assertEqual(model[i, j].dim_in, (i + 1) * (j + 1) - 1)
				self.assertTrue(isinstance(model[i, j], GLM))



	def test_fvbn_train(self):
		xmask = ones([2, 2], dtype='bool')
		ymask = zeros([2, 2], dtype='bool')
		xmask[-1, -1] = False
		ymask[-1, -1] = True

		model = FVBN(2, 2, xmask, ymask, model=GLM(sum(xmask), LogisticFunction, Bernoulli))

		# checkerboard
		data = array([[0, 1], [1, 0]], dtype='bool').reshape(-1, 1)
		data = tile(data, (1, 1000))

		logloss = model.evaluate(data)

		model.initialize(data, parameters={'max_iter': 100})

		# training should converge in much less than 2000 iterations
		self.assertTrue(model.train(data, parameters={'max_iter': 2000}))

		# negative log-likelihood should have decreased
		self.assertLess(model.evaluate(data), logloss)



	def test_pickle(self):
		xmask = ones([2, 2], dtype='bool')
		ymask = zeros([2, 2], dtype='bool')
		xmask[-1, -1] = False
		ymask[-1, -1] = True

		order = [(i // 2, i % 2) for i in permutation(4)]

		model0 = FVBN(2, 2, xmask, ymask, order, GLM(sum(xmask), LogisticFunction, Bernoulli))

		samples = model0.sample(1000)

		tmp_file = mkstemp()[1]

		# store model
		with open(tmp_file, 'wb') as handle:
			dump({'model': model0}, handle)

		# load model
		with open(tmp_file, 'rb') as handle:
			model1 = load(handle)['model']

		# make sure parameters haven't changed
		self.assertEqual(model0.rows, model1.rows)
		self.assertEqual(model0.cols, model1.cols)

		for i in range(model0.rows):
			for j in range(model0.cols):
				if model0[i, j].dim_in > 0:
					self.assertLess(max(abs(model0[i, j].weights - model1[i, j].weights)), 1e-8)
					self.assertLess(max(abs(model0[i, j].bias - model1[i, j].bias)), 1e-8)

		self.assertAlmostEqual(
			mean(model0.loglikelihood(samples)),
			mean(model1.loglikelihood(samples)))



if __name__ == '__main__':
	unittest.main()
