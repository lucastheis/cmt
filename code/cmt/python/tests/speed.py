import sys
import socket

from argparse import ArgumentParser
from time import clock, time
from datetime import datetime
from numpy.random import randn
from cmt.models import MCGSM
 
parser = ArgumentParser(sys.argv[0], description=__doc__)
parser.add_argument('--num_data',    '-d', type=int, default=200000)
parser.add_argument('--dim_in',      '-i', type=int, default=40)
parser.add_argument('--dim_out',     '-o', type=int, default=2)
parser.add_argument('--repetitions', '-r', type=int, default=2)

args = parser.parse_args(sys.argv[1:])

###
print socket.gethostname()
print datetime.now()
print args
print

###
data = randn(args.dim_in, args.num_data), randn(args.dim_out, args.num_data)

model = MCGSM(
	dim_in=args.dim_in,
	dim_out=args.dim_out,
	num_components=12,
	num_features=40,
	num_scales=6)

###
print 'model.loglikelihood'
t = time()
for r in range(args.repetitions):
	model.loglikelihood(*data)
print '{0:12.8f} seconds'.format((time() - t) / float(args.repetitions))
print

###
print 'model._check_performance'
for batch_size in [1000, 2000, 5000]:
	t = model._check_performance(*data, repetitions=args.repetitions, parameters={'batch_size': batch_size})
	print '{0:12.8f} seconds ({1})'.format(t, batch_size)
print

###
print 'model.posterior'
t = time()
for r in range(args.repetitions):
	model.posterior(*data)
print '{0:12.8f} seconds'.format((time() - t) / float(args.repetitions))
print
