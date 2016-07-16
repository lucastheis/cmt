import sys
import unittest

from pickle import dump, load
from tempfile import mkstemp
from numpy import log, mean, max, abs, var, histogram, empty, arange
from numpy.random import randn, rand, randint
from cmt.models import Bernoulli, Poisson, Binomial
from scipy import stats

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



	def test_poisson_sample(self):
		M = 10
		N = 1000
		l = 3.5

		poisson = Poisson(l)

		# observed and expected frequencies
		f_obs = histogram(poisson.sample(N).ravel(), arange(M + 1) - 0.5)[0]
		f_exp = empty(M)

		for m in range(M - 1):
			f_exp[m] = stats.poisson.pmf(m, l) * N
		f_exp[M - 1] = (1. - stats.poisson.cdf(M - 2, l)) * N

		# Pearson's chi-squared test
		p = stats.chisquare(f_obs, f_exp)[1]

		# should fail once in 10000 tests, but not more often
		self.assertGreater(p, 0.0001)



	def test_poisson_loglikelihood(self):
		l = 4.2

		poisson = Poisson(l)

		samples = poisson.sample(100)

		loglik0 = stats.poisson.logpmf(samples, l)
		loglik1 = poisson.loglikelihood(samples)

		self.assertLess(max(abs(loglik0 - loglik1)), 1e-8)



	def test_poisson_pickle(self):
		tmp_file = mkstemp()[1]

		p0 = Poisson(2.5)

		with open(tmp_file, 'w') as handle:
			dump({'p': p0}, handle)

		with open(tmp_file) as handle:
			p1 = load(handle)['p']

		x = p0.sample(100)
		self.assertLess(max(abs(p0.loglikelihood(x) - p1.loglikelihood(x))), 1e-6)



	def test_binomial_sample(self):
		T = 1000

		for p, N in zip(rand(10), randint(100, size=10)):
			binomial = Binomial(N, p)

			# observed and expected frequencies
			f_obs = histogram(binomial.sample(T).ravel(), arange(N + 2) - 0.5)[0]
			f_exp = stats.binom.pmf(arange(N + 1), N, p) * T

			# Pearson's chi-squared test
			p = stats.chisquare(f_obs, f_exp)[1]

			self.assertGreater(p, 0.00001)



	def test_binomial_loglikelihood(self):
		n = 17
		p = .9

		binomial = Binomial(n, p)

		samples = binomial.sample(100)

		loglik0 = stats.binom.logpmf(samples, n, p)
		loglik1 = binomial.loglikelihood(samples)

		self.assertLess(max(abs(loglik0 - loglik1)), 1e-8)



	def test_binomial_pickle(self):
		tmp_file = mkstemp()[1]

		p0 = Binomial(13, .76)

		with open(tmp_file, 'w') as handle:
			dump({'p': p0}, handle)

		with open(tmp_file) as handle:
			p1 = load(handle)['p']

		x = p0.sample(100)
		self.assertLess(max(abs(p0.loglikelihood(x) - p1.loglikelihood(x))), 1e-6)



if __name__ == '__main__':
	unittest.main()
