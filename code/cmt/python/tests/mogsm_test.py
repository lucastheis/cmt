import sys
import unittest

from numpy import asarray, arange, sum
from scipy.stats import binom
from cmt.models import MoGSM

class Tests(unittest.TestCase):
	def test_basics(self):
		model = MoGSM(1, 4, 1)

		model.priors = arange(model.num_components) + 1.
		model.priors = model.priors / sum(model.priors)

		for k in range(model.num_components):
			model[k].mean = [[k]]
			model[k].scales = [[1000.]]

		n = 1000
		samples = asarray(model.sample(n) + .5, dtype=int)

		for k in range(model.num_components):
			p = model.priors.ravel()[k]
			x = sum(samples == k)
			c = binom.cdf(x, n, p)
			self.assertGreater(c, 1e-5)
			self.assertGreater(1. - c, 1e-5)



if __name__ == '__main__':
	unittest.main()
