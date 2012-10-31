#include <iostream>
#include "tools.h"
#include "exception.h"

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
	Tuples inputIndices;
	Tuples outputIndices;

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

	if(inputIndices.size() != model.dimIn() || outputIndices.size() != model.dimOut())
		throw Exception("Model and masks are incompatible.");

	for(int i = 0; i + inputMask.rows() < img.rows(); ++i)
		for(int j = 0; j + inputMask.cols() < img.cols(); ++j) {
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

	Tuples inputIndices;
	Tuples outputIndices;

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

	int numInputs = inputIndices.size();
	int numOutputs = outputIndices.size();
	int numChannels = img.size();

	if(numInputs * numChannels != model.dimIn() || numOutputs * numChannels != model.dimOut())
		throw Exception("Model and masks are incompatible.");

	for(int i = 0; i + inputMask.rows() < img[0].rows(); ++i)
		for(int j = 0; j + inputMask.cols() < img[0].cols(); ++j) {
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
