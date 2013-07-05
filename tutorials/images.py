"""
Trains an MCGSM on a single grayscale images.
"""

import sys

from itertools import chain
from numpy import mean, split, min, max, log
from matplotlib.pyplot import imread, imsave
from pickle import dump
from cmt.models import MCGSM
from cmt.transforms import WhiteningPreconditioner
from cmt.tools import generate_data_from_image, sample_image, rgb2gray

# causal neighborhood and target pixel
input_mask = [
	[1, 1, 1, 1, 1, 1, 1],
	[1, 1, 1, 1, 1, 1, 1],
	[1, 1, 1, 1, 1, 1, 1],
	[1, 1, 1, 0, 0, 0, 0]]
output_mask = [
	[0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 1, 0, 0, 0]]

def main(argv):
	# load image and turn into grayscale
	img = rgb2gray(imread('media/newyork.png'))

	# generate data
	inputs, outputs = generate_data_from_image(
		img, input_mask, output_mask, 220000)

	# split data into training, test, and validation sets
	inputs  = split(inputs,  [100000, 200000], 1)
	outputs = split(outputs, [100000, 200000], 1)

	data_train = inputs[0], outputs[0]
	data_test  = inputs[1], outputs[1]
	data_valid = inputs[2], outputs[2]

	# compute normalizing transformation
	pre = WhiteningPreconditioner(*data_train)

	# intialize model
	model = MCGSM(
		dim_in=data_train[0].shape[0],
		dim_out=data_train[1].shape[0],
		num_components=12,
		num_scales=6,
		num_features=36)

	# fit parameters
	model.initialize(*pre(*data_train))
	model.train(*chain(pre(*data_train), pre(*data_valid)),
		parameters={
			'verbosity': 1,
			'max_iter': 1000,
			'threshold': 1e-7,
			'val_iter': 5,
			'val_look_ahead': 10,
			'num_grad': 20,
		})

	# evaluate model
	print 'Average log-likelihood: {0:.4f} [bit/px]'.format(
			-model.evaluate(data_test[0], data_test[1], pre))

	# synthesize a new image
	img_sample = sample_image(img, model, input_mask, output_mask, pre)

	imsave('newyork_sample.png', img_sample,
		cmap='gray',
		vmin=min(img),
		vmax=max(img))

	# save model
	with open('model.pck', 'wb') as handle:
		dump({'model': model}, handle, 1)

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
