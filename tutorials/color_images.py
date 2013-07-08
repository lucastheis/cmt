"""
Trains an MCGSM on color images.
"""

import sys

from itertools import chain
from numpy import min, max, array
from pickle import dump
from cmt.models import MCGSM
from cmt.transforms import WhiteningPreconditioner
from cmt.tools import imread, imwrite
from cmt.tools import generate_masks, generate_data_from_image
from cmt.tools import sample_image, rgb2ycc, ycc2rgb

def train_model(img, input_mask, output_mask):
	# generate data
	inputs, outputs = generate_data_from_image(
		img, input_mask, output_mask, 120000)

	# split data into training and validation sets
	data_train = inputs[:, :100000], outputs[:, :100000]
	data_valid = inputs[:, 100000:], outputs[:, 100000:]

	# compute normalizing transformation
	pre = WhiteningPreconditioner(*data_train)

	# intialize model
	model = MCGSM(
		dim_in=data_train[0].shape[0],
		dim_out=data_train[1].shape[0],
		num_components=8,
		num_scales=4,
		num_features=30)

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

	return model, pre



def main(argv):
	# load image and turn into grayscale
	img = rgb2ycc(imread('media/newyork.png'))

	# generate masks for grayscale and color model, respectively
	input_mask0, output_mask0 = generate_masks(7, 1)
	input_mask1, output_mask1 = generate_masks([5, 7, 7], 1, [1, 0, 0])

	# train model
	model0, pre0 = train_model(img[:, :, 0], input_mask0, output_mask0)
	model1, pre1 = train_model(img, input_mask1, output_mask1)

	# synthesize a new image
	img_sample = img.copy()

	# sample intensities
	img_sample[:, :, 0] = sample_image(
		img_sample[:, :, 0], model0, input_mask0, output_mask0, pre0)

	# sample color
	img_sample = sample_image(
		img_sample, model1, input_mask1, output_mask1, pre1)

	# convert back to RGB and enforce constraints
	img_sample = ycc2rgb(img_sample)

	imwrite('newyork_sample.png', img_sample, vmin=0, vmax=255)

	# save model
	with open('image_model.pck', 'wb') as handle:
		dump({
			'model0': model0,
			'model1': model1,
			'input_mask0': input_mask0,
			'input_mask1': input_mask1,
			'output_mask0': output_mask0,
			'output_mask1': output_mask1}, handle, 1)

	return 0



if __name__ == '__main__':
	sys.exit(main(sys.argv))
