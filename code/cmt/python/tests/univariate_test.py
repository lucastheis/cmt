import sys
import unittest

from pickle import dump, load
from tempfile import mkstemp
from numpy import log, mean, max, abs
from numpy.random import randn
from cmt.models import Bernoulli

class Tests(unittest.TestCase):
	def test_bernoulli(self):
		bernoulli = Bernoulli(.725)
		self.assertAlmostEqual(mean(bernoulli.sample(1000000)), .725, 2)

		samples = bernoulli.sample(10)

		loglik = log(.725) * samples + log(0.275) * (1. - samples)
		self.assertLess(max(abs(loglik - bernoulli.loglikelihood(samples))), 1e-8)


	
	def test_bernoulli_pickle(self):
		tmp_file = mkstemp()[1]

		p0 = Bernoulli(.3)

		with open(tmp_file, 'w') as handle:
			dump({'p': p0}, handle)

		with open(tmp_file) as handle:
			p1 = load(handle)['p']

		x = p0.sample(100)
		self.assertLess(max(abs(p0.loglikelihood(x) - p1.loglikelihood(x))), 1e-6)



if __name__ == '__main__':
	unittest.main()
