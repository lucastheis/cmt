#include <algorithm>
using std::max;
using std::min;

#include <cstdio>
using std::rand;

#include "tools.h"
#include "utils.h"
#include "exception.h"

using Eigen::Block;

VectorXd extractFromImage(ArrayXXd img, Tuples indices) {
	VectorXd pixels(indices.size());

	for(int i = 0; i < indices.size(); ++i)
		pixels[i] = img(indices[i].first, indices[i].second);

	return pixels;
}



pair<ArrayXXd, ArrayXXd> generateDataFromImage(
	ArrayXXd img,
	ArrayXXb inputMask,
	ArrayXXb outputMask,
	int numSamples)
{
	if(inputMask.cols() != outputMask.cols() || inputMask.rows() != outputMask.rows())
		throw Exception("Input and output masks should be of the same size.");

	int w = img.cols() - inputMask.cols() + 1;
	int h = img.rows() - inputMask.rows() + 1;

	if(numSamples > w * h)
		throw Exception("Image not large enough for this many samples.");

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

	// sample random image locations
	set<int> indices = randomSelect(numSamples, w * h);

	pair<ArrayXXd, ArrayXXd> data = make_pair(
		ArrayXXd(inputIndices.size(), numSamples),
		ArrayXXd(outputIndices.size(), numSamples));

	int k = 0;

	for(set<int>::iterator iter = indices.begin(); iter != indices.end(); ++iter, ++k) {
		// compute indices of image location
		int i = *iter / w;
		int j = *iter % w;

		// extract input and output
		MatrixXd patch = img.block(i, j, inputMask.rows(), inputMask.cols());

		data.first.col(k) = extractFromImage(patch, inputIndices);
		data.second.col(k) = extractFromImage(patch, outputIndices);
	}

	return data;
}



pair<ArrayXXd, ArrayXXd> generateDataFromImage(
	vector<ArrayXXd> img,
	ArrayXXb inputMask,
	ArrayXXb outputMask,
	int numSamples)
{
	int numChannels = img.size();

	if(!numChannels)
		throw Exception("Image should have at least one channel.");

	if(inputMask.cols() != outputMask.cols() || inputMask.rows() != outputMask.rows())
		throw Exception("Input and output masks should be of the same size.");

	int w = img[0].cols() - inputMask.cols() + 1;
	int h = img[0].rows() - inputMask.rows() + 1;

	if(numSamples > w * h)
		throw Exception("Image not large enough for this many samples.");

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

	// sample random image locations
	set<int> indices = randomSelect(numSamples, w * h);

	int numInputs = inputIndices.size();
	int numOutputs = outputIndices.size();

	pair<ArrayXXd, ArrayXXd> data = make_pair(
		ArrayXXd(numChannels * numInputs, numSamples),
		ArrayXXd(numChannels * numOutputs, numSamples));

	int k = 0;

	for(set<int>::iterator iter = indices.begin(); iter != indices.end(); ++iter, ++k) {
		// compute indices of image location
		int i = *iter / w;
		int j = *iter % w;

		// extract input and output
		for(int m = 0; m < numChannels; ++m) {
			MatrixXd patch = img[m].block(i, j, inputMask.rows(), inputMask.cols());
			data.first.block(m * numInputs, k, numInputs, 1) = extractFromImage(patch, inputIndices);
			data.second.block(m * numOutputs, k, numOutputs, 1) = extractFromImage(patch, outputIndices);
		}
	}

	return data;
}



pair<ArrayXXd, ArrayXXd> generateDataFromImage(
	vector<ArrayXXd> img,
	vector<ArrayXXb> inputMask,
	vector<ArrayXXb> outputMask,
	int numSamples)
{
	int numChannels = img.size();

	if(!numChannels)
		throw Exception("Image should have at least one channel.");

	if(inputMask.size() != numChannels || outputMask.size() != numChannels)
		throw Exception("Image and masks need to have the same number of channels.");

	int w = img[0].cols() - inputMask[0].cols() + 1;
	int h = img[0].rows() - inputMask[0].rows() + 1;

	if(numSamples > w * h)
		throw Exception("Image not large enough for this many samples.");

	vector<Tuples> inputIndices;
	vector<Tuples> outputIndices;

	int numInputs = 0;
	int numOutputs = 0;

	// precompute indices of active pixels in masks
	for(int m = 0; m < numChannels; ++m) {
		if(inputMask[m].cols() != outputMask[m].cols() || inputMask[m].rows() != outputMask[m].rows())
			throw Exception("Input and output masks should be of the same size.");

		inputIndices.push_back(Tuples());
		outputIndices.push_back(Tuples());

		for(int i = 0; i < inputMask[m].rows(); ++i)
			for(int j = 0; j < inputMask[m].cols(); ++j) {
				if(inputMask[m](i, j))
					inputIndices[m].push_back(make_pair(i, j));
				if(outputMask[m](i, j))
					outputIndices[m].push_back(make_pair(i, j));
				if(inputMask[m](i, j) && outputMask[m](i, j))
					throw Exception("Input and output mask should not overlap.");
			}

		numInputs += inputIndices[m].size();
		numOutputs += outputIndices[m].size();
	}

	// sample random image locations
	set<int> indices = randomSelect(numSamples, w * h);

	pair<ArrayXXd, ArrayXXd> data = make_pair(
		ArrayXXd(numInputs, numSamples),
		ArrayXXd(numOutputs, numSamples));

	int k = 0;

	for(set<int>::iterator iter = indices.begin(); iter != indices.end(); ++iter, ++k) {
		// compute indices of image location
		int i = *iter / w;
		int j = *iter % w;

		int offsetIn = 0;
		int offsetOut = 0;

		// extract input and output
		for(int m = 0; m < numChannels; ++m) {
			MatrixXd patch = img[m].block(i, j, inputMask[m].rows(), inputMask[m].cols());

			data.first.block(offsetIn, k, inputIndices[m].size(), 1) = extractFromImage(patch, inputIndices[m]);
			data.second.block(offsetOut, k, outputIndices[m].size(), 1) = extractFromImage(patch, outputIndices[m]);

			offsetIn += inputIndices[m].size();
			offsetOut += outputIndices[m].size();
		}
	}

	return data;
}



ArrayXXd sampleImage(
	ArrayXXd img,
	const ConditionalDistribution& model,
	ArrayXXb inputMask,
	ArrayXXb outputMask,
	const Transform& preconditioner)
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

	for(int i = 0; i + inputMask.rows() <= img.rows(); i += h)
		for(int j = 0; j + inputMask.cols() <= img.cols(); j += w) {
			// extract causal neighborhood
			VectorXd input = extractFromImage(
				img.block(i, j, inputMask.rows(), inputMask.cols()), inputIndices);

			// sample output
			VectorXd output = model.sample(preconditioner(input));

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
	ArrayXXb outputMask,
	const Transform& preconditioner)
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

	int numInputs = inputIndices.size();
	int numOutputs = outputIndices.size();
	int numChannels = img.size();

	// TODO: allow for non-invertible preconditioners
	if(numInputs * numChannels != model.dimIn() || numOutputs * numChannels != model.dimOut())
		throw Exception("Model and masks are incompatible.");

	for(int i = 0; i + inputMask.rows() <= img[0].rows(); i += h)
		for(int j = 0; j + inputMask.cols() <= img[0].cols(); j += w) {
			VectorXd input(numInputs * numChannels);

			// extract causal neighborhood
			#pragma omp parallel for
			for(int m = 0; m < numChannels; ++m)
				input.segment(m * numInputs, numInputs) = extractFromImage(
					img[m].block(i, j, inputMask.rows(), inputMask.cols()), inputIndices);

			// sample output
			VectorXd output = model.sample(preconditioner(input));

			// replace pixels in image by output
			#pragma omp parallel for
			for(int m = 0; m < numChannels; ++m)
				for(int k = 0; k < outputIndices.size(); ++k)
					img[m](i + outputIndices[k].first, j + outputIndices[k].second) = output[m * numOutputs + k];
		}

	return img;
}



vector<ArrayXXd> sampleImage(
	vector<ArrayXXd> img,
	const ConditionalDistribution& model,
	vector<ArrayXXb> inputMask,
	vector<ArrayXXb> outputMask,
	const Transform& preconditioner)
{
	int numChannels = img.size();

	if(!numChannels)
		throw Exception("Image should have at least one channel.");

	if(inputMask.size() != numChannels || outputMask.size() != numChannels)
		throw Exception("Image and masks need to have the same number of channels.");

	vector<Tuples> inputIndices;
	vector<Tuples> outputIndices;

	// boundaries of output region
	int iMin = outputMask[0].rows();
	int iMax = 0;
	int jMin = outputMask[0].cols();
	int jMax = 0;

	int numInputs = 0;
	int numOutputs = 0;

	for(int m = 0; m < numChannels; ++m) {
		if(inputMask[m].cols() != outputMask[m].cols() || inputMask[m].rows() != outputMask[m].rows())
			throw Exception("Input and output masks should be of the same size.");

		inputIndices.push_back(Tuples());
		outputIndices.push_back(Tuples());

		// precompute indices of active pixels in masks
		for(int i = 0; i < inputMask[m].rows(); ++i)
			for(int j = 0; j < inputMask[m].cols(); ++j) {
				if(inputMask[m](i, j))
					inputIndices[m].push_back(make_pair(i, j));
				if(outputMask[m](i, j)) {
					outputIndices[m].push_back(make_pair(i, j));

					// update boundaries
					iMax = max(iMax, i);
					iMin = min(iMin, i);
					jMax = max(jMax, j);
					jMin = min(jMin, j);
				}

				if(inputMask[m](i, j) && outputMask[m](i, j))
					throw Exception("Input and output mask should not overlap.");
			}

		numInputs += inputIndices[m].size();
		numOutputs += outputIndices[m].size();
	}

	// width and height of output region
	int h = iMax - iMin + 1;
	int w = jMax - jMin + 1;

	if(w < 1)
		throw Exception("There needs to be at least one active pixel in the output mask.");

	// TODO: allow for non-invertible preconditioners
	if(numInputs != model.dimIn() || numOutputs != model.dimOut())
		throw Exception("Model and masks are incompatible.");

	for(int i = 0; i + inputMask[0].rows() <= img[0].rows(); i += h)
		for(int j = 0; j + inputMask[0].cols() <= img[0].cols(); j += w) {
			VectorXd input(numInputs);

			// extract causal neighborhood
			// TODO: parallelize
			for(int m = 0, offset = 0; m < numChannels; ++m) {
				input.segment(offset, inputMask[m].size()) = extractFromImage(
					img[m].block(i, j, inputMask[m].rows(), inputMask[m].cols()), inputIndices[m]);
				offset += inputIndices[m].size();
			}

			// sample output
			VectorXd output = model.sample(preconditioner(input));

			// replace pixels in image by output
			// TODO: parallelize
			for(int m = 0, offset = 0; m < numChannels; ++m) {
				for(int k = 0; k < outputIndices[m].size(); ++k)
					img[m](i + outputIndices[m][k].first, j + outputIndices[m][k].second) = output[offset + k];
				offset += outputIndices[m].size();
			}
		}

	return img;
}
