import unittest
from numpy import *
from numpy import max
from numpy.random import randn
from numpy.linalg import inv

import sys
sys.path.append('/Users/lucas/Code/Python/mixtures/code/')
from models import GSM as GSM_
from time import time

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
		gsm0.scales = [1, 5]
		gsm0.mean = [1, 1, 1]
		gsm0.priors = [0.7, 0.3]

		samples = gsm0.sample(10000)

		# try to recover parameters
		gsm1 = GSM(3, 2)
		gsm1.train(samples, parameters={'max_iter': 50})

		self.assertLess(max(abs(gsm1.mean - gsm0.mean)), 0.1)
		self.assertLess(max(abs(sort(gsm1.priors.ravel()) - sort(gsm0.priors.ravel()))), 0.1)
		self.assertLess(max(abs(sort(gsm1.scales.ravel()) - sort(gsm0.scales.ravel()))), 0.5)
		self.assertLess(max(abs(gsm1.covariance - gsm0.covariance)), 0.2)



if __name__ == '__main__':
	unittest.main()
