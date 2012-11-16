#include <iostream>
#include <algorithm>
#include "tools.h"
#include "exception.h"

using std::max;
using std::min;

VectorXd extractFromImage(ArrayXXd img, Tuples indices) {
	VectorXd pixels(indices.size());

	for(int i = 0; i < indices.size(); ++i)
		pixels[i] = img(indices[i].first, indices[i].second);

	return pixels;
}



ArrayXXd sampleImage(
	ArrayXXd img,
	const ConditionalDistribution& model,
	ArrayXXb inputMask,
	ArrayXXb outputMask)
{
	if(inputMask.cols() != outputMask.cols() || inputMask.rows() != outputMask.rows())
		throw Exception("Input and output masks should be of the same size.");

	Tuples inputIndices;
	Tuples outputIndices;

	// boundaries of output region
	int iMin = outputMask.rows();
	int iMax = 0;
	int jMin = outputMask.cols();
	int jMax = 0;

	// precompute indices of active pixels in masks
	for(int i = 0; i < inputMask.rows(); ++i)
		for(int j = 0; j < inputMask.cols(); ++j) {
			if(inputMask(i, j))
				inputIndices.push_back(make_pair(i, j));
			if(outputMask(i, j)) {
				outputIndices.push_back(make_pair(i, j));

				// update boundaries
				iMax = max(iMax, i);
				iMin = min(iMin, i);
				jMax = max(jMax, j);
				jMin = min(jMin, j);
			}
			if(inputMask(i, j) && outputMask(i, j))
				throw Exception("Input and output mask should not overlap.");
		}

	// width and height of output region
	int h = iMax - iMin + 1;
	int w = jMax - jMin + 1;

	if(w < 1)
		throw Exception("There needs to be at least one active pixel in the output mask.");

	if(outputIndices.size() != w * h)
		throw Exception("Unsupported output mask.");

	if(inputIndices.size() != model.dimIn() || outputIndices.size() != model.dimOut())
		throw Exception("Model and masks are incompatible.");

	for(int i = 0; i + inputMask.rows() < img.rows(); i += h)
		for(int j = 0; j + inputMask.cols() < img.cols(); j += w) {
			// extract causal neighborhood
			VectorXd input = extractFromImage(
				img.block(i, j, inputMask.rows(), inputMask.cols()), inputIndices);

			// sample output
			VectorXd output = model.sample(input);

			// replace pixels in image by output
			for(int k = 0; k < outputIndices.size(); ++k)
				img(i + outputIndices[k].first, j + outputIndices[k].second) = output[k];
		}

	return img;
}



vector<ArrayXXd> sampleImage(
	vector<ArrayXXd> img,
	const ConditionalDistribution& model,
	ArrayXXb inputMask,
	ArrayXXb outputMask)
{
	if(!img.size())
		throw Exception("Image should have at least one channel.");

	if(inputMask.cols() != outputMask.cols() || inputMask.rows() != outputMask.rows())
		throw Exception("Input and output masks should be of the same size.");

	Tuples inputIndices;
	Tuples outputIndices;

	// boundaries of output region
	int iMin = outputMask.rows();
	int iMax = 0;
	int jMin = outputMask.cols();
	int jMax = 0;

	// precompute indices of active pixels in masks
	for(int i = 0; i < inputMask.rows(); ++i)
		for(int j = 0; j < inputMask.cols(); ++j) {
			if(inputMask(i, j))
				inputIndices.push_back(make_pair(i, j));
			if(outputMask(i, j))
				outputIndices.push_back(make_pair(i, j));
			if(inputMask(i, j) && outputMask(i, j))
				throw Exception("Input and output mask should not overlap.");
		}

	// width and height of output region
	int h = iMax - iMin + 1;
	int w = jMax - jMin + 1;

	if(w < 1)
		throw Exception("There needs to be at least one active pixel in the output mask.");

	int numInputs = inputIndices.size();
	int numOutputs = outputIndices.size();
	int numChannels = img.size();

	if(numInputs * numChannels != model.dimIn() || numOutputs * numChannels != model.dimOut())
		throw Exception("Model and masks are incompatible.");

	for(int i = 0; i + inputMask.rows() < img[0].rows(); i += h)
		for(int j = 0; j + inputMask.cols() < img[0].cols(); j += w) {
			VectorXd input(numInputs * numChannels);

			// extract causal neighborhood
			for(int m = 0; m < numChannels; ++m)
				input.segment(m * numInputs, numInputs) = extractFromImage(
					img[m].block(i, j, inputMask.rows(), inputMask.cols()), inputIndices);

			// sample output
			VectorXd output = model.sample(input);

			// replace pixels in image by output
			for(int m = 0; m < numChannels; ++m)
				for(int k = 0; k < outputIndices.size(); ++k)
					img[m](i + outputIndices[k].first, j + outputIndices[k].second) = output[m * numOutputs + k];
		}

	return img;
}
