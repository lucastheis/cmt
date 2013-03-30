import sys
import unittest

sys.path.append('./code')

from cmt import MCBM
from numpy import *
from numpy import max
from numpy.random import *
from pickle import dump, load
from tempfile import mkstemp

class Tests(unittest.TestCase):
	def test_gradient(self):
		mcbm = MCBM(5, 2, 10)

		# choose random parameters
		mcbm._set_parameters(randn(*mcbm._parameters().shape))

		err = mcbm._check_gradient(
			randint(2, size=[mcbm.dim_in, 1000]),
			randint(2, size=[mcbm.dim_out, 1000]), 1e-5)
		self.assertLess(err, 1e-8)

		for param in ['priors', 'weights', 'features', 'pred', 'input_bias', 'output_bias']:
			err = mcbm._check_gradient(
				randint(2, size=[mcbm.dim_in, 1000]),
				randint(2, size=[mcbm.dim_out, 1000]),
				1e-5,
				parameters={
					'train_prior': param == 'priors',
					'train_weights': param == 'weights',
					'train_features': param == 'features',
					'train_predictors': param == 'pred',
				})
			self.assertLess(err, 1e-8)



if __name__ == '__main__':
	unittest.main()
